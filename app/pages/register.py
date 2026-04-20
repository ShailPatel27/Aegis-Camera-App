import cv2
import time
import re
import threading
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QListWidget,
)

from config.settings import (
    ACTIVITY_LOG_COOLDOWNS,
    ACTIVITY_LOG_DEFAULT_COOLDOWN_SECONDS,
    FACE_PREVIEW_INTERVAL_MS,
    FACE_PRIMARY_HOLD_FRAMES,
    FACE_PRIMARY_MATCH_IOU,
    FACE_PRIMARY_MIN_RELATIVE_AREA,
    FACE_REGISTER_SAMPLES_REQUIRED,
    FACE_RECOGNITION_UNKNOWN_LABEL,
    FACE_USERS_DIR,
)
from core.face_engine import FaceEngine
from app.services.auth_client import auth_client


class RegisterPage(QWidget):
    sync_result = pyqtSignal(dict)

    def __init__(self, logger=None):
        super().__init__()
        self.face_engine = FaceEngine()
        self.logger = logger
        self.event_last_logged = {}
        self.live_worker = None
        self.current_frame = None
        self.current_faces = []
        self.primary_face = None
        self.primary_miss_frames = 0
        self.primary_match_name = None
        self.primary_match_score = 0.0
        self.pending_embeddings = []
        self.pending_face_crop = None
        self._save_in_progress = False
        self._last_preview_ts = 0.0
        self._last_match_ts = 0.0
        # Keep register preview lighter than live stream to avoid UI lockups.
        self._preview_interval_s = max(0.10, float(FACE_PREVIEW_INTERVAL_MS) / 1000.0)
        self._match_interval_s = max(0.30, self._preview_interval_s * 2.0)
        self.setStyleSheet(
            """
            QLineEdit {
                background-color: #f8fafc;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 8px 10px;
            }
            QLineEdit:focus {
                border: 1px solid #2563eb;
            }
            QPushButton {
                background-color: #e2e8f0;
                color: #0f172a;
                border: 1px solid #94a3b8;
                border-radius: 8px;
                padding: 8px 10px;
                font-weight: 600;
                outline: none;
            }
            QPushButton:hover {
                background-color: #cbd5e1;
            }
            QPushButton:focus {
                border: 1px solid #2563eb;
                outline: none;
            }
            QListWidget {
                background-color: #0b1224;
                color: #e2e8f0;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 6px;
            }
            QListWidget::item:selected {
                background-color: #1d4ed8;
                color: #ffffff;
            }
            """
        )

        root = QVBoxLayout()
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)
        self.setLayout(root)
        self.sync_result.connect(self._on_sync_result)

        title = QLabel("Face Registration")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(title)

        self.preview = QLabel("Feed not running. Start feed from Live page.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 220)
        self.preview.setStyleSheet("background: #000; border-radius: 10px; color: #94a3b8;")
        root.addWidget(self.preview, 2)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        root.addLayout(row1)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter user name")
        row1.addWidget(self.name_input, 2)

        self.capture_btn = QPushButton("Capture Face Sample")
        self.capture_btn.setFocusPolicy(Qt.NoFocus)
        self.capture_btn.clicked.connect(self.capture_sample)
        row1.addWidget(self.capture_btn, 1)

        self.save_btn = QPushButton("Save Registered User")
        self.save_btn.setFocusPolicy(Qt.NoFocus)
        self.save_btn.clicked.connect(self.save_user)
        row1.addWidget(self.save_btn, 1)


        row2 = QHBoxLayout()
        row2.setSpacing(8)
        root.addLayout(row2)

        self.clear_btn = QPushButton("Clear Captured Samples")
        self.clear_btn.setFocusPolicy(Qt.NoFocus)
        self.clear_btn.clicked.connect(self.clear_samples)
        row2.addWidget(self.clear_btn, 1)

        self.delete_btn = QPushButton("Delete Selected User")
        self.delete_btn.setFocusPolicy(Qt.NoFocus)
        self.delete_btn.clicked.connect(self.delete_selected_user)
        row2.addWidget(self.delete_btn, 1)

        self.samples_label = QLabel(self._sample_text())
        self.samples_label.setStyleSheet("color: #94a3b8;")
        root.addWidget(self.samples_label)

        self.status_label = QLabel("Align face and capture sample.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #94a3b8;")
        root.addWidget(self.status_label)

        users_title = QLabel("Registered Users")
        users_title.setStyleSheet("font-weight: 600;")
        root.addWidget(users_title)

        self.user_list = QListWidget()
        self.user_list.setFocusPolicy(Qt.NoFocus)
        root.addWidget(self.user_list, 1)

        self.refresh_users()

    def _sample_text(self):
        return f"Samples: {len(self.pending_embeddings)}/{FACE_REGISTER_SAMPLES_REQUIRED}"

    @staticmethod
    def _face_area(face):
        _, _, w, h = face
        return max(0, int(w)) * max(0, int(h))

    def set_status(self, text):
        self.status_label.setText(text)

    def _resolve_cooldown(self, key):
        return ACTIVITY_LOG_COOLDOWNS.get(key, ACTIVITY_LOG_DEFAULT_COOLDOWN_SECONDS)

    def _log_activity(self, event_key, message, cooldown_key=None):
        if not self.logger:
            return
        now = time.time()
        cooldown = self._resolve_cooldown(cooldown_key or event_key)
        last_logged = self.event_last_logged.get(event_key, 0.0)
        if now - last_logged < cooldown:
            return
        self.logger.add_log(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.event_last_logged[event_key] = now

    def refresh_users(self):
        self.face_engine.registry = self.face_engine._load_registry()
        self.user_list.clear()
        for user in self.face_engine.list_users():
            profile = self.face_engine.registry.get(user, {}) if isinstance(self.face_engine.registry.get(user), dict) else {}
            role = str(profile.get("role") or "user").lower()
            label = f"{user} [BLACKLIST]" if role == "blacklist" else user
            self.user_list.addItem(label)

    def set_live_worker(self, worker):
        if self.live_worker is not None:
            try:
                self.live_worker.raw_frame_ready.disconnect(self.on_live_frame)
            except Exception:
                pass
        self.live_worker = worker
        if self.live_worker is not None:
            self.live_worker.raw_frame_ready.connect(self.on_live_frame)
            self.set_status("Live feed connected. Align face and capture sample.")
            self._log_activity("register:link:connected", "Register page linked to live feed", "register:link")
        else:
            self.set_status("Feed stopped. Start feed from Live page.")
            self._log_activity("register:link:stopped", "Register page feed link stopped", "register:link")

    def _select_primary_face(self, faces):
        if len(faces) == 0:
            if self.primary_face is not None and self.primary_miss_frames < FACE_PRIMARY_HOLD_FRAMES:
                self.primary_miss_frames += 1
                return self.primary_face
            self.primary_miss_frames = 0
            self.primary_face = None
            self.primary_match_name = None
            self.primary_match_score = 0.0
            return None

        faces_sorted = sorted(faces, key=self._face_area, reverse=True)
        biggest_area = self._face_area(faces_sorted[0])
        stable_faces = [
            f for f in faces_sorted
            if self._face_area(f) >= biggest_area * FACE_PRIMARY_MIN_RELATIVE_AREA
        ]

        chosen = stable_faces[0]
        if self.primary_face is not None:
            best_iou = 0.0
            best_face = None
            for face in stable_faces:
                iou = self.face_engine.box_iou_xywh(face, self.primary_face)
                if iou > best_iou:
                    best_iou = iou
                    best_face = face
            if best_face is not None and best_iou >= FACE_PRIMARY_MATCH_IOU:
                chosen = best_face

        self.primary_face = chosen
        self.primary_miss_frames = 0
        return chosen

    def on_live_frame(self, frame):
        now = time.time()
        # Throttle heavy face processing/rendering on UI thread.
        if (now - self._last_preview_ts) < self._preview_interval_s:
            return
        self._last_preview_ts = now

        self.current_frame = frame.copy()

        faces = list(self.face_engine.detect_faces(self.current_frame))
        self.current_faces = faces
        primary = self._select_primary_face(faces)
        # Matching is more expensive than detect; run less often.
        should_refresh_match = (now - self._last_match_ts) >= self._match_interval_s
        if should_refresh_match:
            self._last_match_ts = now
            self.primary_match_name = None
            self.primary_match_score = 0.0
            if primary is not None:
                emb = self.face_engine.build_embedding(self.current_frame, primary)
                match_name, match_score = self.face_engine.identify_embedding(emb)
                self.primary_match_name = match_name
                self.primary_match_score = match_score

        display = self.current_frame.copy()
        for face in faces:
            x, y, w, h = face
            color = (148, 163, 184)
            thickness = 1
            if primary is not None and tuple(face) == tuple(primary):
                color = (16, 185, 129)
                thickness = 2
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)

        if primary is not None:
            px, py, _, _ = primary
            label_y = py - 10 if py > 30 else py + 26
            primary_text = "Primary face"
            if self.primary_match_name:
                primary_text = f"{self.primary_match_name} ({self.primary_match_score:.2f})"
            cv2.putText(
                display,
                primary_text,
                (px, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (16, 185, 129),
                2,
                cv2.LINE_AA,
            )

        if len(faces) == 0 and primary is None:
            self.set_status("No face detected. Move closer and face camera.")
        elif self.primary_match_name:
            self.set_status(f"User already registered: {self.primary_match_name}")
        elif len(faces) > 1:
            self.set_status("Multiple faces detected. Largest stable face is selected.")
        else:
            self.set_status("Face ready. Capture sample.")

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        img = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        self.preview.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.preview.width(),
                self.preview.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def capture_sample(self):
        if self.current_frame is None:
            self.set_status("Camera frame not available. Start feed first.")
            return

        if self.primary_face is None:
            self.set_status("No stable primary face yet. Hold still and try again.")
            return

        if self.primary_match_name:
            self.set_status(f"User already registered: {self.primary_match_name}")
            self._log_activity(
                f"register:duplicate:{self.primary_match_name}",
                f"Duplicate registration blocked for {self.primary_match_name}",
                "register:duplicate",
            )
            return

        emb = self.face_engine.build_embedding(self.current_frame, self.primary_face)
        if emb is None:
            self.set_status("Could not build face sample. Try again.")
            return
        x, y, w, h = self.primary_face
        crop = self.current_frame[max(0, y):max(0, y) + max(1, h), max(0, x):max(0, x) + max(1, w)]
        self.pending_face_crop = crop.copy() if crop.size > 0 else None

        self.pending_embeddings.append(emb)
        self.samples_label.setText(self._sample_text())
        self.set_status("Sample captured.")
        self._log_activity("register:sample", "Face sample captured", "register:sample")

    def clear_samples(self):
        self.pending_embeddings = []
        self.pending_face_crop = None
        self.samples_label.setText(self._sample_text())
        self.set_status("Captured samples cleared.")
        self._log_activity("register:sample:clear", "Captured face samples cleared", "register:sample")

    @staticmethod
    def _safe_name(name: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip())
        return clean.strip("_") or "user"

    def _save_user_image_local(self, name: str):
        if self.pending_face_crop is None or self.pending_face_crop.size == 0:
            return None
        out_dir = Path(FACE_USERS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self._safe_name(name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), self.pending_face_crop)
        return str(out_path)

    def save_user(self):
        if self._save_in_progress:
            self.set_status("Save already in progress. Please wait...")
            return
        name = self.name_input.text()
        if len(self.pending_embeddings) < FACE_REGISTER_SAMPLES_REQUIRED:
            self.set_status(
                f"Need at least {FACE_REGISTER_SAMPLES_REQUIRED} samples before saving."
            )
            return

        role = "user"
        local_image_path = self._save_user_image_local(name)
        try:
            saved_name = self.face_engine.register_user(
                name,
                self.pending_embeddings,
                role=role,
                image_path=local_image_path,
            )
        except ValueError as exc:
            self.set_status(str(exc))
            return

        self._save_in_progress = True
        self.save_btn.setEnabled(False)

        # Snapshot for async sync before we clear UI buffers.
        face_crop = self.pending_face_crop.copy() if self.pending_face_crop is not None else None
        centroid = None
        if saved_name in self.face_engine.registry:
            centroid = self.face_engine.registry[saved_name].get("centroid")

        self.pending_embeddings = []
        self.pending_face_crop = None
        self.samples_label.setText(self._sample_text())
        self.name_input.setText("")
        self.refresh_users()
        self.set_status(f"User saved: {saved_name} | syncing to monitor...")
        self._log_activity(f"register:save:{saved_name}", f"User registered: {saved_name}", "register:save")

        threading.Thread(
            target=self._sync_face_profile_worker,
            args=(saved_name, role, centroid, face_crop),
            daemon=True,
        ).start()

    def _sync_face_profile_worker(self, saved_name, role, centroid, face_crop):
        result = {"saved_name": saved_name, "ok": False, "message": ""}
        try:
            session = auth_client.load_session()
            if session and centroid is not None:
                image_url = auth_client.upload_face_image(session, saved_name, face_crop)
                auth_client.sync_face_profile(
                    session,
                    saved_name,
                    centroid,
                    role=role,
                    image_url=image_url,
                )
                if (
                    saved_name in self.face_engine.registry
                    and isinstance(self.face_engine.registry[saved_name], dict)
                ):
                    self.face_engine.registry[saved_name]["image_url"] = image_url
                    self.face_engine._save_registry()
                result["ok"] = True
                result["message"] = "synced to monitor"
            else:
                result["message"] = "local save only"
        except Exception as exc:
            result["message"] = f"sync failed: {exc}"
        self.sync_result.emit(result)

    def _on_sync_result(self, payload):
        self._save_in_progress = False
        self.save_btn.setEnabled(True)
        if not isinstance(payload, dict):
            return
        name = payload.get("saved_name") or "user"
        msg = payload.get("message") or ""
        self.set_status(f"User saved: {name} | {msg}")

    def delete_selected_user(self):
        item = self.user_list.currentItem()
        if not item:
            self.set_status("Select a user to delete.")
            return

        display_name = item.text().strip()
        name = display_name.replace(" [BLACKLIST]", "").strip()
        if not name or name == FACE_RECOGNITION_UNKNOWN_LABEL:
            self.set_status("Invalid user selected.")
            return

        deleted = self.face_engine.delete_user(name)
        if deleted:
            self.refresh_users()
            self.set_status(f"Deleted user: {name}")
            self._log_activity(f"register:delete:{name}", f"User deleted: {name}", "register:delete")
            return
        self.set_status("User not found.")
