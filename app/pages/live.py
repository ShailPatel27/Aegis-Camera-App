from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import time
import cv2

from core.camera import Camera
from core.detector import Detector
from core.motion import MotionDetector
from app.widgets.toggle import ToggleSwitch

from config.settings import *


class LivePage(QWidget):
    def __init__(self, logger=None):
        super().__init__()

        self.logger = logger
        self.last_log_time = 0

        self.camera = Camera()
        self.detector = Detector()
        self.motion_detector = MotionDetector()

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

    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return

        current_time = time.time()

        # ===== FPS CALCULATION =====
        delta = current_time - self.prev_time
        if delta > 0:
            self.fps = 1.0 / delta
        self.prev_time = current_time

        self.fps_label.setText(f"FPS: {self.fps:.1f}")

        # ===== YOLO =====
        run_yolo = any([
            self.intrusion.isChecked(),
            self.crowd.isChecked(),
            self.vehicle.isChecked(),
            self.threat.isChecked()
        ])

        detections = []
        if run_yolo:
            detections = self.detector.detect(frame)

        person_count = 0

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cls = det["class_name"]

            if cls == "person":
                person_count += 1

                if self.intrusion.isChecked():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if cls in ["car", "truck", "bus"] and self.vehicle.isChecked():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            if cls in ["knife", "scissors"] and self.threat.isChecked():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ===== PEOPLE COUNT UI =====
        self.people_label.setText(f"People: {person_count}")

        # ===== MOTION =====
        if self.motion.isChecked():
            motion_boxes = self.motion_detector.detect(frame)

            for (x1, y1, x2, y2) in motion_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # ===== LOGGING =====
        if self.intrusion.isChecked() and person_count > 0:
            if self.logger and current_time - self.last_log_time > LOG_COOLDOWN:
                self.logger.add_log(
                    f"[{time.strftime('%H:%M:%S')}] Person detected ({person_count})"
                )
                self.last_log_time = current_time

        # ===== RENDER =====
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