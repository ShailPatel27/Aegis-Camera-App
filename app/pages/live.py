from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import time
import cv2
from core.ai_worker import AIWorker
from core.identity_memory import IdentityMemory
from app.widgets.toggle import ToggleSwitch

from config.settings import *


class LivePage(QWidget):
    def __init__(self, logger=None):
        super().__init__()

        self.prev_time = time.time()
        self.frame_count = 0
        self.fps_accumulator = 0
        self.fps = 0
        
        self.logger = logger
        self.event_last_logged = {}
        self.identity_memory = IdentityMemory()

        # FPS tracking
        self.prev_time = time.time()
        self.fps = 0

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        self.setLayout(layout)

        # ===== TOP STATUS BAR =====
        top_bar = QHBoxLayout()

        self.status = QLabel("AI: ACTIVE")
        self.status.setStyleSheet("color: #94a3b8;")

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #94a3b8;")

        self.people_label = QLabel("People: 0")
        self.people_label.setStyleSheet("color: #94a3b8;")

        top_bar.addWidget(self.status)
        top_bar.addStretch()
        top_bar.addWidget(self.people_label)
        top_bar.addWidget(self.fps_label)

        layout.addLayout(top_bar)

        # ===== VIDEO =====
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.setStyleSheet("background-color: black; border-radius: 10px;")
        layout.addWidget(self.label)

        # ===== CONTROLS =====
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()

        self.intrusion = ToggleSwitch("Intrusion")
        self.crowd = ToggleSwitch("Crowd")
        self.vehicle = ToggleSwitch("Vehicle")

        self.threat = ToggleSwitch("Threat")
        self.motion = ToggleSwitch("Motion")
        self.loiter = ToggleSwitch("Loitering")
        self.emergency = ToggleSwitch("Emergency")

        self.intrusion.setChecked(DEFAULT_INTRUSION)
        self.crowd.setChecked(DEFAULT_CROWD)
        self.vehicle.setChecked(DEFAULT_VEHICLE)
        self.threat.setChecked(DEFAULT_THREAT)
        self.motion.setChecked(DEFAULT_MOTION)
        self.loiter.setChecked(DEFAULT_LOITER)
        self.emergency.setChecked(DEFAULT_EMERGENCY)

        for t in [self.intrusion, self.crowd, self.vehicle]:
            row1.addWidget(t)

        for t in [self.threat, self.motion, self.loiter, self.emergency]:
            row2.addWidget(t)

        layout.addLayout(row1)
        layout.addLayout(row2)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(FRAME_INTERVAL_MS)
        
        self.worker = AIWorker({
            "intrusion": self.intrusion.isChecked,
            "crowd": self.crowd.isChecked,
            "vehicle": self.vehicle.isChecked,
            "threat": self.threat.isChecked,
            "motion": self.motion.isChecked,
        })

        self.worker.frame_ready.connect(self.update_ui)
        self.worker.start()

    def update_frame(self):
        # This function is now ONLY for UI timing / fallback
        # AI processing is handled in AIWorker

        current_time = time.time()
            
    def update_ui(self, frame, person_count):
        current_time = time.time()

        # ===== FPS SMOOTHING =====
        delta = current_time - self.prev_time
        self.prev_time = current_time

        if delta > 0:
            inst_fps = 1.0 / delta
            self.fps_accumulator += inst_fps
            self.frame_count += 1

        if self.frame_count >= FPS_SMOOTHING_WINDOW:
            self.fps = self.fps_accumulator / self.frame_count
            self.fps_label.setText(f"FPS: {int(self.fps)}")

            self.frame_count = 0
            self.fps_accumulator = 0

        # ===== PEOPLE COUNT =====
        self.people_label.setText(f"People: {person_count}")
        if self.intrusion.isChecked() and person_count > 0:
            self._log_with_cooldown(
                event_key="intrusion:person",
                message=f"[{time.strftime('%H:%M:%S')}] Person detected ({person_count})",
                cooldown_seconds=INTRUSION_EVENT_COOLDOWN_SECONDS,
                now=current_time,
            )

        # ===== RENDER =====
        if DISPLAY_MIRROR_FEED:
            frame = cv2.flip(frame, 1)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape

        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)

        if self.label.width() > 0 and self.label.height() > 0:
            self.label.setPixmap(
                pix.scaled(
                    self.label.width(),
                    self.label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def _log_with_cooldown(self, event_key, message, cooldown_seconds, now):
        if not self.logger:
            return
        last_logged = self.event_last_logged.get(event_key, 0.0)
        if now - last_logged < cooldown_seconds:
            return
        self.logger.add_log(message)
        self.event_last_logged[event_key] = now

    def log_identity_detected(self, user_id, confidence=None):
        now = time.time()
        if not self.identity_memory.should_log_identity(user_id, now):
            return

        confidence_text = ""
        if confidence is not None:
            confidence_text = f" ({confidence:.2f})"
        self._log_with_cooldown(
            event_key=f"identity:{user_id}",
            message=f"[{time.strftime('%H:%M:%S')}] User detected: {user_id}{confidence_text}",
            cooldown_seconds=IDENTITY_EVENT_COOLDOWN_SECONDS,
            now=now,
        )
        self.identity_memory.mark_logged(user_id, now)
