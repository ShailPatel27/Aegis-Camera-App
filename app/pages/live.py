import time
import threading

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QGridLayout,
)

from app.widgets.toggle import ToggleSwitch
from app.services.auth_client import auth_client
from core.ai_worker import AIWorker
from core.identity_memory import IdentityMemory
from config.settings import *


class LivePage(QWidget):
    worker_changed = pyqtSignal(object)
    session_invalid = pyqtSignal(str)
    session_sync_result = pyqtSignal(object)

    def __init__(self, logger=None, session=None):
        super().__init__()
        self.setStyleSheet(
            """
            QPushButton {
                background-color: #dbeafe;
                color: #0f172a;
                border: 1px solid #93c5fd;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
                outline: none;
                min-width: 110px;
            }
            QPushButton:hover {
                background-color: #bfdbfe;
            }
            QPushButton:disabled {
                background-color: #334155;
                color: #94a3b8;
                border: 1px solid #475569;
            }
            QPushButton:focus {
                border: 1px solid #3b82f6;
                outline: none;
            }
            """
        )

        self.prev_time = time.time()
        self.frame_count = 0
        self.fps_accumulator = 0
        self.fps = 0
        self.logger = logger
        self.session = session if isinstance(session, dict) else auth_client.load_session()
        self.event_last_logged = {}
        self.identity_memory = IdentityMemory()
        self.worker = None
        self.feed_state = "inactive"
        self.toggle_labels = {}
        self.toggle_controls = {}
        self._session_invalid_emitted = False
        self._sync_in_progress = False

        root = QVBoxLayout()
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)
        self.setLayout(root)

        top_bar = QHBoxLayout()
        self.status = QLabel("AI: ACTIVE")
        self.status.setStyleSheet("color: #94a3b8; font-weight: 600;")
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #94a3b8;")
        self.people_label = QLabel("People: 0")
        self.people_label.setStyleSheet("color: #94a3b8;")
        top_bar.addWidget(self.status)
        top_bar.addStretch()
        top_bar.addWidget(self.people_label)
        top_bar.addWidget(self.fps_label)
        root.addLayout(top_bar)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.setStyleSheet("background-color: black; border-radius: 12px;")
        root.addWidget(self.label, 1)

        controls = QGridLayout()
        controls.setHorizontalSpacing(18)
        controls.setVerticalSpacing(10)

        self.intrusion = ToggleSwitch("Intrusion")
        self.crowd = ToggleSwitch("Crowd")
        self.vehicle = ToggleSwitch("Vehicle")
        self.threat = ToggleSwitch("Threat")
        self.motion = ToggleSwitch("Motion")
        self.loiter = ToggleSwitch("Loitering")
        self.emergency = ToggleSwitch("Emergency")
        self.face_recognition = ToggleSwitch("Face Recognition")
        self.screenshot = ToggleSwitch("Screenshot")

        self.intrusion.setChecked(DEFAULT_INTRUSION)
        self.crowd.setChecked(DEFAULT_CROWD)
        self.vehicle.setChecked(DEFAULT_VEHICLE)
        self.threat.setChecked(DEFAULT_THREAT)
        self.motion.setChecked(DEFAULT_MOTION)
        self.loiter.setChecked(DEFAULT_LOITER)
        self.emergency.setChecked(DEFAULT_EMERGENCY)
        self.face_recognition.setChecked(DEFAULT_FACE_RECOGNITION)

        self.toggle_controls = {
            "intrusion": self.intrusion,
            "crowd": self.crowd,
            "vehicle": self.vehicle,
            "threat": self.threat,
            "motion": self.motion,
            "loiter": self.loiter,
            "emergency": self.emergency,
            "face_recognition": self.face_recognition,
            "screenshot": self.screenshot,
        }

        self.toggle_labels = {
            self.intrusion: "Intrusion",
            self.crowd: "Crowd",
            self.vehicle: "Vehicle",
            self.threat: "Threat",
            self.motion: "Motion",
            self.loiter: "Loitering",
            self.emergency: "Emergency",
            self.face_recognition: "Face Recognition",
        }
        self._bind_toggle(self.intrusion, "Intrusion", "intrusion")
        self._bind_toggle(self.crowd, "Crowd", "crowd")
        self._bind_toggle(self.vehicle, "Vehicle", "vehicle")
        self._bind_toggle(self.threat, "Threat", "threat")
        self._bind_toggle(self.motion, "Motion", "motion")
        self._bind_toggle(self.loiter, "Loitering", "loiter")
        self._bind_toggle(self.emergency, "Emergency", "emergency")
        self._bind_toggle(self.face_recognition, "Face Recognition", "face_recognition")
        self._bind_toggle(self.screenshot, "Screenshot", "screenshot")

        controls.addWidget(self.intrusion, 0, 0)
        controls.addWidget(self.crowd, 0, 1)
        controls.addWidget(self.vehicle, 0, 2)
        controls.addWidget(self.threat, 1, 0)
        controls.addWidget(self.motion, 1, 1)
        controls.addWidget(self.loiter, 1, 2)
        controls.addWidget(self.emergency, 2, 0)
        controls.addWidget(self.face_recognition, 2, 1)
        controls.addWidget(self.screenshot, 2, 2)

        action_bar = QHBoxLayout()
        action_bar.setSpacing(10)
        action_bar.addStretch()

        self.start_btn = QPushButton("Start Feed")
        self.start_btn.setFocusPolicy(Qt.NoFocus)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; border: 1px solid #15803d; border-radius: 8px; padding: 8px 14px; font-weight: 700; }"
            "QPushButton:hover { background-color: #15803d; }"
            "QPushButton:disabled { background-color: #14532d; color: #bbf7d0; border: 1px solid #14532d; }"
            "QPushButton:focus { border: 1px solid #22c55e; outline: none; }"
        )
        self.start_btn.clicked.connect(self.start_worker)
        action_bar.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause Feed")
        self.pause_btn.setFocusPolicy(Qt.NoFocus)
        self.pause_btn.clicked.connect(self.pause_worker)
        action_bar.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop Feed")
        self.stop_btn.setFocusPolicy(Qt.NoFocus)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #7f1d1d; color: #fee2e2; border: 1px solid #991b1b; border-radius: 8px; padding: 8px 14px; font-weight: 700; }"
            "QPushButton:hover { background-color: #991b1b; }"
            "QPushButton:disabled { background-color: #450a0a; color: #fca5a5; border: 1px solid #450a0a; }"
            "QPushButton:focus { border: 1px solid #ef4444; outline: none; }"
        )
        self.stop_btn.clicked.connect(self.stop_inactive)
        action_bar.addWidget(self.stop_btn)

        root.addLayout(controls)
        root.addLayout(action_bar)

        self.session_sync_result.connect(self._on_session_sync_result)

        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(5000)
        self.sync_timer.timeout.connect(self._sync_with_backend)
        self.sync_timer.start()

        self._refresh_toggles_from_db()

        self.start_worker()

    def _bind_toggle(self, toggle, label, key):
        toggle.toggled.connect(
            lambda checked, name=label: self._log_activity(
                event_key=f"toggle:{name.lower().replace(' ', '_')}",
                message=f"{name} {'enabled' if checked else 'disabled'}",
                cooldown_key="toggle",
            )
        )
        toggle.toggled.connect(lambda checked, toggle_key=key: self._persist_toggle_change(toggle_key, checked))

    def _refresh_toggles_from_db(self):
        session = self.session if isinstance(self.session, dict) else auth_client.load_session()
        if not isinstance(session, dict):
            return

        toggles = auth_client.get_ai_toggles(session)
        self._apply_toggle_values(toggles)
        self.session = session

    def _sync_with_backend(self):
        if self._sync_in_progress:
            return

        session = self.session if isinstance(self.session, dict) else auth_client.load_session()
        if not isinstance(session, dict):
            return

        self._sync_in_progress = True
        snapshot = dict(session)
        threading.Thread(
            target=self._run_session_sync_worker,
            args=(snapshot,),
            daemon=True,
        ).start()

    def _run_session_sync_worker(self, session_snapshot):
        try:
            refreshed = auth_client.refresh_session(session_snapshot)
            self.session_sync_result.emit({"status": "ok", "session": refreshed})
        except PermissionError as exc:
            self.session_sync_result.emit({"status": "invalid", "reason": str(exc)})
        except Exception:
            # Ignore transient errors; next timer tick retries.
            self.session_sync_result.emit({"status": "error"})

    def _on_session_sync_result(self, payload):
        self._sync_in_progress = False
        if not isinstance(payload, dict):
            return

        status = payload.get("status")
        if status == "ok":
            refreshed = payload.get("session")
            if isinstance(refreshed, dict):
                self.session = refreshed
                self._apply_toggle_values(auth_client.get_ai_toggles(refreshed))
            return
        if status == "invalid":
            self._emit_session_invalid(payload.get("reason") or "Session expired")

    def _emit_session_invalid(self, reason):
        if self._session_invalid_emitted:
            return
        self._session_invalid_emitted = True
        self.stop_worker()
        self.session_invalid.emit(reason or "Session expired")

    def _apply_toggle_values(self, toggles):
        for key, toggle in self.toggle_controls.items():
            if key not in toggles:
                continue
            toggle.blockSignals(True)
            toggle.setChecked(bool(toggles[key]))
            toggle.blockSignals(False)

    def _persist_toggle_change(self, toggle_key, enabled):
        if not isinstance(self.session, dict):
            self.session = auth_client.load_session()
        if not isinstance(self.session, dict):
            return

        try:
            updated_camera = auth_client.update_ai_toggle(self.session, toggle_key, enabled)
            if isinstance(updated_camera, dict):
                self.session["camera"] = updated_camera
        except PermissionError as exc:
            self._emit_session_invalid(str(exc))
        except Exception:
            # Keep UI responsive even if network/db sync fails.
            pass

    def _reset_fps_metrics(self):
        self.prev_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.fps_accumulator = 0
        self.fps_label.setText("FPS: 0")

    def _render_frame_to_label(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        if self.label.width() > 0 and self.label.height() > 0:
            self.label.setPixmap(
                pix.scaled(
                    self.label.width(),
                    self.label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def _inactive_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        white = (240, 240, 240)
        cx, cy = 640, 320
        cv2.rectangle(frame, (cx - 120, cy - 70), (cx + 120, cy + 70), white, 6)
        cv2.rectangle(frame, (cx - 55, cy - 100), (cx + 55, cy - 70), white, 6)
        cv2.circle(frame, (cx, cy), 38, white, 6)
        cv2.putText(
            frame,
            "CAMERA INACTIVE",
            (470, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            white,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press Start Feed to reactivate camera",
            (430, 570),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            white,
            2,
            cv2.LINE_AA,
        )
        return frame

    def update_frame(self):
        pass

    def update_ui(self, frame, person_count, faces, activity):
        if self.feed_state != "active":
            return

        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time
        if delta > 0:
            self.fps_accumulator += 1.0 / delta
            self.frame_count += 1

        if self.frame_count >= FPS_SMOOTHING_WINDOW:
            self.fps = self.fps_accumulator / self.frame_count
            self.fps_label.setText(f"FPS: {int(self.fps)}")
            self.frame_count = 0
            self.fps_accumulator = 0

        self.people_label.setText(f"People: {person_count}")
        if self.intrusion.isChecked() and activity.get("person_count", 0) > 0:
            self._log_activity(
                event_key="intrusion:person",
                message=f"Person detected ({activity['person_count']})",
                cooldown_key="intrusion",
                now=current_time,
            )

        if self.crowd.isChecked() and activity.get("crowd_triggered"):
            self._log_activity(
                event_key="crowd:threshold",
                message=f"Crowd threshold reached ({activity['person_count']})",
                cooldown_key="crowd",
                now=current_time,
            )

        if self.vehicle.isChecked() and activity.get("vehicle_count", 0) > 0:
            self._log_activity(
                event_key="vehicle:detected",
                message=f"Vehicle detected ({activity['vehicle_count']})",
                cooldown_key="vehicle",
                now=current_time,
            )

        if self.threat.isChecked() and activity.get("threat_count", 0) > 0:
            self._log_activity(
                event_key="threat:detected",
                message=f"Threat object detected ({activity['threat_count']})",
                cooldown_key="threat",
                now=current_time,
            )

        if self.motion.isChecked() and activity.get("motion_count", 0) > 0:
            self._log_activity(
                event_key="motion:detected",
                message=f"Motion detected ({activity['motion_count']})",
                cooldown_key="motion",
                now=current_time,
            )

        if self.loiter.isChecked() and activity.get("loiter_count", 0) > 0:
            self._log_activity(
                event_key="loiter:detected",
                message=f"Loitering detected ({activity['loiter_count']})",
                cooldown_key="loiter",
                now=current_time,
            )

        if self.emergency.isChecked() and activity.get("emergency_total", 0) > 0:
            if activity.get("emergency_captured"):
                self._log_activity(
                    event_key=f"emergency:progress:{activity['emergency_progress']}",
                    message=(
                        f"Emergency step captured "
                        f"({activity['emergency_progress']}/{activity['emergency_total']})"
                    ),
                    cooldown_key="emergency",
                    now=current_time,
                )
            if activity.get("emergency_triggered"):
                self._log_activity(
                    event_key="emergency:triggered",
                    message="Emergency pattern sequence completed",
                    cooldown_key="emergency",
                    now=current_time,
                )

        for face in faces:
            if face.get("matched") and face.get("name"):
                self.log_identity_detected(face["name"], face.get("score"))

        self._render_frame_to_label(frame)

    def _toggle_map(self):
        return {
            "intrusion": self.intrusion.isChecked,
            "crowd": self.crowd.isChecked,
            "vehicle": self.vehicle.isChecked,
            "threat": self.threat.isChecked,
            "motion": self.motion.isChecked,
            "loiter": self.loiter.isChecked,
            "emergency": self.emergency.isChecked,
            "face_recognition": self.face_recognition.isChecked,
            "screenshot": self.screenshot.isChecked,
        }

    def start_worker(self):
        if self.worker is not None:
            return
        self._refresh_toggles_from_db()
        self._sync_with_backend()
        self.feed_state = "active"
        self._reset_fps_metrics()
        self.worker = AIWorker(self._toggle_map())
        self.worker.frame_ready.connect(self.update_ui)
        self.worker.start()
        self.status.setText("AI: ACTIVE")
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.worker_changed.emit(self.worker)
        self._log_activity("feed:start", "Feed started", "feed:start")

    def pause_worker(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker = None
        self.feed_state = "paused"
        self._reset_fps_metrics()
        self.status.setText("AI: PAUSED")
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker_changed.emit(None)
        self._log_activity("feed:pause", "Feed paused", "feed:pause")

    def stop_inactive(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.worker_changed.emit(None)
        self.feed_state = "inactive"
        self.status.setText("AI: INACTIVE")
        self.people_label.setText("People: 0")
        self._reset_fps_metrics()
        self._render_frame_to_label(self._inactive_frame())
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self._log_activity("feed:stop", "Feed stopped (camera inactive)", "feed:stop")

    def stop_worker(self):
        # Called by window close handler.
        if hasattr(self, "sync_timer") and self.sync_timer.isActive():
            self.sync_timer.stop()
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.worker_changed.emit(None)
        self.feed_state = "inactive"
        self.status.setText("AI: INACTIVE")

    def _resolve_cooldown(self, key):
        return ACTIVITY_LOG_COOLDOWNS.get(key, ACTIVITY_LOG_DEFAULT_COOLDOWN_SECONDS)

    def _log_with_cooldown(self, event_key, message, cooldown_seconds, now):
        if not self.logger:
            return
        last_logged = self.event_last_logged.get(event_key, 0.0)
        if now - last_logged < cooldown_seconds:
            return
        self.logger.add_log(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.event_last_logged[event_key] = now

    def _log_activity(self, event_key, message, cooldown_key=None, now=None):
        now = time.time() if now is None else now
        cooldown = self._resolve_cooldown(cooldown_key or event_key)
        self._log_with_cooldown(event_key, message, cooldown, now)

    def log_identity_detected(self, user_id, confidence=None):
        now = time.time()
        if not self.identity_memory.should_log_identity(user_id, now):
            return

        confidence_text = f" ({confidence:.2f})" if confidence is not None else ""
        self._log_activity(
            event_key=f"identity:{user_id}",
            message=f"User detected: {user_id}{confidence_text}",
            cooldown_key="identity",
            now=now,
        )
        self.identity_memory.mark_logged(user_id, now)
