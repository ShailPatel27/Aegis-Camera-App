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
    QComboBox,
)

from app.widgets.toggle import ToggleSwitch
from app.services.auth_client import auth_client
from app.services.recorder import ChunkRecorderService
from core.ai_worker import AIWorker
from core.identity_memory import IdentityMemory
from config.settings import *


class LivePage(QWidget):
    worker_changed = pyqtSignal(object)
    session_invalid = pyqtSignal(str)
    session_sync_result = pyqtSignal(object)
    remote_state_result = pyqtSignal(object)

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
        self._applying_remote_state = False
        self._remote_state_update_in_progress = False
        self._recorder_starting = False
        self._recorder_service = ChunkRecorderService()
        self._toggle_sync_in_progress = set()
        self._ignore_remote_apply_until = 0.0
        camera_from_session = None
        if isinstance(self.session, dict) and isinstance(self.session.get("camera"), dict):
            camera_from_session = self.session.get("camera", {}).get("selected_camera")
        try:
            self.selected_camera_index = int(
                CAMERA_INDEX if camera_from_session is None else camera_from_session
            )
        except (TypeError, ValueError):
            self.selected_camera_index = int(CAMERA_INDEX)

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
        self.camera_selector = QComboBox()
        self.camera_selector.setFocusPolicy(Qt.NoFocus)
        self.camera_selector.setStyleSheet(
            "QComboBox { background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 6px 10px; min-width: 130px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background-color: #0f172a; color: #e2e8f0; selection-background-color: #1d4ed8; }"
        )
        self._populate_camera_selector()
        self.camera_selector.currentIndexChanged.connect(self._on_camera_selection_changed)
        top_bar.addWidget(self.status)
        top_bar.addStretch()
        top_bar.addWidget(QLabel("Camera:"))
        top_bar.addWidget(self.camera_selector)
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

        self.stream_btn = QPushButton("Start Feed")
        self.stream_btn.setFocusPolicy(Qt.NoFocus)
        self.stream_btn.clicked.connect(self._on_stream_button_clicked)
        action_bar.addWidget(self.stream_btn)

        self.pause_resume_btn = QPushButton("Pause")
        self.pause_resume_btn.setFocusPolicy(Qt.NoFocus)
        self.pause_resume_btn.clicked.connect(self._on_pause_resume_clicked)
        action_bar.addWidget(self.pause_resume_btn)

        root.addLayout(controls)
        root.addLayout(action_bar)

        self.session_sync_result.connect(self._on_session_sync_result)
        self.remote_state_result.connect(self._on_remote_state_result)

        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(5000)
        self.sync_timer.timeout.connect(self._sync_with_backend)
        self.sync_timer.start()

        self._refresh_toggles_from_db()
        self._update_control_buttons()

        self.start_worker()
        self._sync_with_backend()

    def _bind_toggle(self, toggle, label, key):
        toggle.toggled.connect(
            lambda checked, name=label: self._log_activity(
                event_key=f"toggle:{name.lower().replace(' ', '_')}",
                message=f"{name} {'enabled' if checked else 'disabled'}",
                cooldown_key="toggle",
            )
        )
        toggle.toggled.connect(lambda checked, toggle_key=key: self._persist_toggle_change(toggle_key, checked))

    def _discover_camera_indices(self, max_index=8):
        discovered = []
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx)
            try:
                if cap is None or not cap.isOpened():
                    continue
                ok, _ = cap.read()
                if ok:
                    discovered.append(idx)
            except Exception:
                continue
            finally:
                try:
                    cap.release()
                except Exception:
                    pass

        if not discovered:
            return [self.selected_camera_index]

        if self.selected_camera_index not in discovered:
            discovered.append(self.selected_camera_index)
        return sorted(set(discovered))

    def _populate_camera_selector(self):
        options = self._discover_camera_indices()
        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        for idx in options:
            self.camera_selector.addItem(f"Camera {idx}", idx)
        selected_pos = self.camera_selector.findData(self.selected_camera_index)
        if selected_pos >= 0:
            self.camera_selector.setCurrentIndex(selected_pos)
        self.camera_selector.blockSignals(False)

    def _restart_worker_for_selected_camera(self):
        if self.worker is None:
            return

        self.worker.stop()
        self.worker = None
        self.worker_changed.emit(None)
        self._stop_recorder_service()
        self._reset_fps_metrics()

        self.worker = AIWorker(self._toggle_map(), camera_index=self.selected_camera_index)
        self.worker.raw_frame_ready.connect(self._on_raw_frame_ready)
        self.worker.frame_ready.connect(self.update_ui)
        self.worker.start()
        self._start_recorder_service()
        self.status.setText("AI: ACTIVE")
        self.worker_changed.emit(self.worker)

    def _persist_selected_camera_async(self, selected_camera: int):
        if not isinstance(self.session, dict):
            self.session = auth_client.load_session()
        if not isinstance(self.session, dict):
            return

        snapshot = dict(self.session)

        def _worker():
            try:
                updated = auth_client.update_selected_camera(snapshot, selected_camera)
                if isinstance(updated, dict) and isinstance(self.session, dict):
                    self.session["camera"] = updated
            except Exception as exc:
                self._log_activity(
                    event_key="camera:select:sync_error",
                    message=f"Failed to sync selected camera: {exc}",
                    cooldown_key="toggle",
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_camera_selection_changed(self, _):
        selected = self.camera_selector.currentData()
        try:
            next_index = int(selected)
        except (TypeError, ValueError):
            return

        if next_index == self.selected_camera_index:
            return

        self.selected_camera_index = next_index
        self._log_activity(
            event_key=f"camera:select:{next_index}",
            message=f"Switched to camera {next_index}",
            cooldown_key="toggle",
        )

        if isinstance(self.session, dict) and isinstance(self.session.get("camera"), dict):
            self.session["camera"]["selected_camera"] = next_index

        if self.feed_state == "active":
            self._restart_worker_for_selected_camera()

        self._persist_selected_camera_async(next_index)

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
            try:
                auth_client.sync_faces_registry(refreshed)
            except Exception:
                pass
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
                self._apply_remote_stream_state()
            return
        if status == "invalid":
            self._emit_session_invalid(payload.get("reason") or "Session expired")

    def _emit_session_invalid(self, reason):
        if self._session_invalid_emitted:
            return
        self._session_invalid_emitted = True
        self.stop_worker()
        self.session_invalid.emit(reason or "Session expired")

    def _is_remote_stream_enabled(self):
        camera = self.session.get("camera", {}) if isinstance(self.session, dict) else {}
        if not isinstance(camera, dict):
            return False
        return (camera.get("stream_enabled") is True) or (camera.get("status") == "online")

    def _is_remote_paused(self):
        camera = self.session.get("camera", {}) if isinstance(self.session, dict) else {}
        if not isinstance(camera, dict):
            return False
        config = camera.get("config") if isinstance(camera.get("config"), dict) else {}
        return bool(config.get("feed_paused", False))

    def _apply_remote_stream_state(self):
        if self._applying_remote_state:
            return
        if time.time() < self._ignore_remote_apply_until:
            return
        self._applying_remote_state = True
        try:
            stream_enabled = self._is_remote_stream_enabled()
            paused = self._is_remote_paused()

            if not stream_enabled:
                if self.feed_state != "inactive":
                    self.stop_inactive(sync_remote=False)
                self._update_control_buttons()
                return

            if paused:
                if self.feed_state == "active":
                    self.pause_worker(sync_remote=False)
                elif self.feed_state == "inactive":
                    self.start_worker(sync_remote=False)
                    self.pause_worker(sync_remote=False)
                self._update_control_buttons()
                return

            if self.feed_state != "active":
                self.start_worker(sync_remote=False)
            self._update_control_buttons()
        finally:
            self._applying_remote_state = False

    def _start_recorder_service(self):
        if self._recorder_starting or self._recorder_service.running:
            return

        self._recorder_starting = True

        def _worker():
            try:
                self._recorder_service.start()
                self._log_activity("recorder:start", "Recorder chunk upload service started.", "feed:start")
            except Exception as exc:
                self._log_activity("recorder:start:error", f"Recorder start failed: {exc}", "feed:start")
            finally:
                self._recorder_starting = False

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_recorder_service(self):
        try:
            self._recorder_service.stop()
        except Exception:
            pass
        self._log_activity("recorder:stop", "Recorder chunk upload service stopped.", "feed:stop")

    def _apply_remote_state_async(self, stream_enabled: bool, paused: bool):
        if self._remote_state_update_in_progress:
            return
        session = self.session if isinstance(self.session, dict) else auth_client.load_session()
        if not isinstance(session, dict):
            return

        snapshot = {
            "token": session.get("token"),
            "user": dict(session.get("user", {})),
            "camera": dict(session.get("camera", {})),
        }
        if not snapshot.get("token"):
            return

        self._remote_state_update_in_progress = True

        def _worker():
            try:
                updated = auth_client.set_camera_stream_state(snapshot, bool(stream_enabled))
                merged = {
                    "token": snapshot["token"],
                    "user": snapshot["user"],
                    "camera": updated if isinstance(updated, dict) else snapshot["camera"],
                }
                updated = auth_client.set_camera_paused(merged, bool(paused))
                self.remote_state_result.emit({"status": "ok", "camera": updated})
            except PermissionError as exc:
                self.remote_state_result.emit({"status": "invalid", "reason": str(exc)})
            except Exception as exc:
                self.remote_state_result.emit({"status": "error", "reason": str(exc)})

        threading.Thread(target=_worker, daemon=True).start()

    def _on_remote_state_result(self, payload):
        self._remote_state_update_in_progress = False
        if not isinstance(payload, dict):
            return
        status = payload.get("status")
        if status == "ok":
            camera = payload.get("camera")
            if isinstance(camera, dict):
                if not isinstance(self.session, dict):
                    self.session = auth_client.load_session() or {}
                self.session["camera"] = camera
            return
        if status == "invalid":
            self._emit_session_invalid(payload.get("reason") or "Session expired")

    def _update_control_buttons(self):
        if self.feed_state == "inactive":
            self.stream_btn.setText("Start Feed")
            self.stream_btn.setStyleSheet(
                "QPushButton { background-color: #16a34a; color: white; border: 1px solid #15803d; border-radius: 8px; padding: 8px 14px; font-weight: 700; }"
                "QPushButton:hover { background-color: #15803d; }"
                "QPushButton:disabled { background-color: #14532d; color: #bbf7d0; border: 1px solid #14532d; }"
                "QPushButton:focus { border: 1px solid #22c55e; outline: none; }"
            )
            self.pause_resume_btn.setText("Pause")
            self.pause_resume_btn.setEnabled(False)
        elif self.feed_state == "paused":
            self.stream_btn.setText("Stop Feed")
            self.stream_btn.setStyleSheet(
                "QPushButton { background-color: #7f1d1d; color: #fee2e2; border: 1px solid #991b1b; border-radius: 8px; padding: 8px 14px; font-weight: 700; }"
                "QPushButton:hover { background-color: #991b1b; }"
                "QPushButton:disabled { background-color: #450a0a; color: #fca5a5; border: 1px solid #450a0a; }"
                "QPushButton:focus { border: 1px solid #ef4444; outline: none; }"
            )
            self.pause_resume_btn.setText("Resume")
            self.pause_resume_btn.setEnabled(True)
        else:
            self.stream_btn.setText("Stop Feed")
            self.stream_btn.setStyleSheet(
                "QPushButton { background-color: #7f1d1d; color: #fee2e2; border: 1px solid #991b1b; border-radius: 8px; padding: 8px 14px; font-weight: 700; }"
                "QPushButton:hover { background-color: #991b1b; }"
                "QPushButton:disabled { background-color: #450a0a; color: #fca5a5; border: 1px solid #450a0a; }"
                "QPushButton:focus { border: 1px solid #ef4444; outline: none; }"
            )
            self.pause_resume_btn.setText("Pause")
            self.pause_resume_btn.setEnabled(True)

        if self.feed_state == "paused":
            self.pause_resume_btn.setStyleSheet(
                "QPushButton { background-color: #16a34a; color: #ffffff; border: 1px solid #15803d; border-radius: 8px; padding: 8px 14px; font-weight: 700; }"
                "QPushButton:hover { background-color: #15803d; }"
                "QPushButton:disabled { background-color: #14532d; color: #bbf7d0; border: 1px solid #14532d; }"
                "QPushButton:focus { border: 1px solid #22c55e; outline: none; }"
            )
        else:
            self.pause_resume_btn.setStyleSheet(
                "QPushButton { background-color: #ca8a04; color: #fffbeb; border: 1px solid #a16207; border-radius: 8px; padding: 8px 14px; font-weight: 700; }"
                "QPushButton:hover { background-color: #a16207; }"
                "QPushButton:disabled { background-color: #713f12; color: #fde68a; border: 1px solid #713f12; }"
                "QPushButton:focus { border: 1px solid #f59e0b; outline: none; }"
            )

    def _on_stream_button_clicked(self):
        if self.feed_state == "inactive":
            self.start_worker(sync_remote=True)
        else:
            self.stop_inactive(sync_remote=True)

    def _on_pause_resume_clicked(self):
        if self.feed_state == "active":
            self.pause_worker(sync_remote=True)
        elif self.feed_state == "paused":
            self.start_worker(sync_remote=True)

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

        # Local-first: update UI/runtime state immediately, then sync remote asynchronously.
        if isinstance(self.session.get("camera"), dict):
            camera = dict(self.session.get("camera") or {})
            config = dict(camera.get("config") or {})
            ai_toggles = dict(config.get("ai_toggles") or {})
            ai_toggles[toggle_key] = bool(enabled)
            config["ai_toggles"] = ai_toggles
            camera["config"] = config
            self.session["camera"] = camera

        if toggle_key in self._toggle_sync_in_progress:
            return
        self._toggle_sync_in_progress.add(toggle_key)

        snapshot = dict(self.session)

        def _worker():
            try:
                updated_camera = auth_client.update_ai_toggle(snapshot, toggle_key, enabled)
                if isinstance(updated_camera, dict) and isinstance(self.session, dict):
                    self.session["camera"] = updated_camera
            except PermissionError as exc:
                self._emit_session_invalid(str(exc))
            except Exception as exc:
                self._log_activity(
                    event_key=f"toggle:{toggle_key}:sync_error",
                    message=f"Failed to sync {toggle_key} toggle: {exc}",
                    cooldown_key="toggle",
                )
            finally:
                self._toggle_sync_in_progress.discard(toggle_key)

        threading.Thread(target=_worker, daemon=True).start()

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

    def _on_raw_frame_ready(self, frame):
        if self.feed_state != "active":
            return
        try:
            self._recorder_service.add_frame(frame)
        except Exception:
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
                frame=frame,
            )

        if self.crowd.isChecked() and activity.get("crowd_triggered"):
            self._log_activity(
                event_key="crowd:threshold",
                message=f"Crowd threshold reached ({activity['person_count']})",
                cooldown_key="crowd",
                now=current_time,
                frame=frame,
            )

        if self.vehicle.isChecked() and activity.get("vehicle_count", 0) > 0:
            self._log_activity(
                event_key="vehicle:detected",
                message=f"Vehicle detected ({activity['vehicle_count']})",
                cooldown_key="vehicle",
                now=current_time,
                frame=frame,
            )

        if self.threat.isChecked() and activity.get("threat_count", 0) > 0:
            self._log_activity(
                event_key="threat:detected",
                message=f"Threat object detected ({activity['threat_count']})",
                cooldown_key="threat",
                now=current_time,
                frame=frame,
            )

        if self.motion.isChecked() and activity.get("motion_count", 0) > 0:
            self._log_activity(
                event_key="motion:detected",
                message=f"Motion detected ({activity['motion_count']})",
                cooldown_key="motion",
                now=current_time,
                frame=frame,
            )

        if self.loiter.isChecked() and activity.get("loiter_count", 0) > 0:
            self._log_activity(
                event_key="loiter:detected",
                message=f"Loitering detected ({activity['loiter_count']})",
                cooldown_key="loiter",
                now=current_time,
                frame=frame,
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
                    frame=frame,
                )
            if activity.get("emergency_triggered"):
                self._log_activity(
                    event_key="emergency:triggered",
                    message="Emergency pattern sequence completed",
                    cooldown_key="emergency",
                    now=current_time,
                    frame=frame,
                )

        for face in faces:
            if face.get("matched") and face.get("name"):
                self.log_identity_detected(
                    face["name"],
                    face.get("score"),
                    frame=frame,
                    role=face.get("role"),
                )

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

    def start_worker(self, sync_remote=True):
        if self.worker is not None:
            return
        self._refresh_toggles_from_db()
        self.feed_state = "active"
        self._reset_fps_metrics()
        self.worker = AIWorker(self._toggle_map(), camera_index=self.selected_camera_index)
        self.worker.raw_frame_ready.connect(self._on_raw_frame_ready)
        self.worker.frame_ready.connect(self.update_ui)
        self.worker.start()
        self._start_recorder_service()
        self.status.setText("AI: ACTIVE")
        self._update_control_buttons()
        self.worker_changed.emit(self.worker)
        self._log_activity("feed:start", "Feed started", "feed:start")
        if sync_remote:
            self._ignore_remote_apply_until = time.time() + 4.0
            self._apply_remote_state_async(True, False)

    def pause_worker(self, sync_remote=True):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker = None
        self._stop_recorder_service()
        self.feed_state = "paused"
        self._reset_fps_metrics()
        self.status.setText("AI: PAUSED")
        self._update_control_buttons()
        self.worker_changed.emit(None)
        self._log_activity("feed:pause", "Feed paused", "feed:pause")
        if sync_remote:
            self._ignore_remote_apply_until = time.time() + 4.0
            self._apply_remote_state_async(True, True)

    def stop_inactive(self, sync_remote=True):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.worker_changed.emit(None)
        self._stop_recorder_service()
        self.feed_state = "inactive"
        self.status.setText("AI: INACTIVE")
        self.people_label.setText("People: 0")
        self._reset_fps_metrics()
        self._render_frame_to_label(self._inactive_frame())
        self._update_control_buttons()
        self._log_activity("feed:stop", "Feed stopped (camera inactive)", "feed:stop")
        if sync_remote:
            self._ignore_remote_apply_until = time.time() + 4.0
            self._apply_remote_state_async(False, False)

    def stop_worker(self):
        # Called by window close handler.
        if hasattr(self, "sync_timer") and self.sync_timer.isActive():
            self.sync_timer.stop()
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.worker_changed.emit(None)
        self._stop_recorder_service()
        self.feed_state = "inactive"
        self.status.setText("AI: INACTIVE")
        self._update_control_buttons()

    def _resolve_cooldown(self, key):
        return ACTIVITY_LOG_COOLDOWNS.get(key, ACTIVITY_LOG_DEFAULT_COOLDOWN_SECONDS)

    def _log_with_cooldown(self, event_key, message, cooldown_seconds, now):
        if not self.logger:
            return False
        last_logged = self.event_last_logged.get(event_key, 0.0)
        if now - last_logged < cooldown_seconds:
            return False
        self.logger.add_log(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.event_last_logged[event_key] = now
        return True

    @staticmethod
    def _alert_type_from_event_key(event_key: str):
        if event_key.startswith("intrusion:"):
            return "intrusion"
        if event_key.startswith("crowd:"):
            return "crowd"
        if event_key.startswith("vehicle:"):
            return "vehicle"
        if event_key.startswith("threat:"):
            return "threat"
        if event_key.startswith("motion:"):
            return "motion"
        if event_key.startswith("loiter:"):
            return "loiter"
        if event_key == "emergency:triggered":
            return "emergency"
        return None

    def _push_alert_async(self, alert_type: str, message: str, confidence=None, face_name=None, frame=None):
        session_snapshot = self.session if isinstance(self.session, dict) else auth_client.load_session()
        if not isinstance(session_snapshot, dict):
            return
        frame_snapshot = frame.copy() if frame is not None else None

        def _worker():
            try:
                auth_client.create_alert(
                    session_snapshot,
                    alert_type=alert_type,
                    message=message,
                    confidence=confidence,
                    face_name=face_name,
                    frame_bgr=frame_snapshot,
                )
            except Exception as exc:
                self._log_with_cooldown(
                    event_key=f"alerts:sync_error:{alert_type}",
                    message=f"DB alert sync failed ({alert_type}): {exc}",
                    cooldown_seconds=self._resolve_cooldown("toggle"),
                    now=time.time(),
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _log_activity(self, event_key, message, cooldown_key=None, now=None, frame=None):
        now = time.time() if now is None else now
        cooldown = self._resolve_cooldown(cooldown_key or event_key)
        did_log = self._log_with_cooldown(event_key, message, cooldown, now)
        if not did_log:
            return
        alert_type = self._alert_type_from_event_key(event_key)
        if alert_type:
            snapshot = frame if self.screenshot.isChecked() else None
            self._push_alert_async(alert_type=alert_type, message=message, frame=snapshot)

    def log_identity_detected(self, user_id, confidence=None, frame=None, role="user"):
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
        if str(role or "user").lower() == "blacklist":
            self._push_alert_async(
                alert_type="threat",
                message=f"Blacklisted person detected: {user_id}{confidence_text}",
                confidence=confidence,
                face_name=user_id,
                frame=frame if self.screenshot.isChecked() else None,
            )
        self.identity_memory.mark_logged(user_id, now)
