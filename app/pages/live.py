from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import time
import cv2

from core.camera import Camera
from core.detector import Detector
from app.widgets.toggle import ToggleSwitch


class LivePage(QWidget):
    def __init__(self, logger=None):
        super().__init__()

        self.logger = logger
        self.last_log_time = 0

        self.camera = Camera()
        self.detector = Detector()

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        self.setLayout(layout)

        # Status
        self.status = QLabel("AI: ACTIVE | Camera: ONLINE")
        self.status.setStyleSheet("padding: 5px; color: #94a3b8;")
        layout.addWidget(self.status)

        # Video
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.setStyleSheet("""
            background-color: black;
            border-radius: 10px;
        """)
        layout.addWidget(self.label)

        # Controls (2 rows)
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()

        self.intrusion = ToggleSwitch("Intrusion")
        self.crowd = ToggleSwitch("Crowd")
        self.vehicle = ToggleSwitch("Vehicle")

        self.threat = ToggleSwitch("Threat")
        self.motion = ToggleSwitch("Motion")
        self.loiter = ToggleSwitch("Loitering")

        # 🔴 Emergency toggle (special)
        self.emergency = ToggleSwitch("Emergency")
        self.emergency.setStyleSheet("""
        QCheckBox {
            color: #ef4444;
            font-size: 13px;
            spacing: 10px;
        }
        QCheckBox::indicator {
            width: 40px;
            height: 20px;
            border-radius: 10px;
            background-color: #7f1d1d;
        }
        QCheckBox::indicator:checked {
            background-color: #dc2626;
        }
        """)

        # defaults
        self.intrusion.setChecked(True)
        self.crowd.setChecked(True)

        for t in [self.intrusion, self.crowd, self.vehicle]:
            row1.addWidget(t)

        for t in [self.threat, self.motion, self.loiter, self.emergency]:
            row2.addWidget(t)

        layout.addLayout(row1)
        layout.addLayout(row2)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        frame = self.camera.get_frame()
        if frame is None:
            return

        current_time = time.time()

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
            conf = det["confidence"]

            if cls == "person":
                person_count += 1

                if self.intrusion.isChecked():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"PERSON {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if cls in ["car", "truck", "bus"] and self.vehicle.isChecked():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"{cls.upper()} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            if cls in ["knife", "scissors"] and self.threat.isChecked():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"THREAT {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if self.crowd.isChecked():
            cv2.putText(frame, f"People: {person_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Intrusion logging
        if self.intrusion.isChecked() and person_count > 0:
            if self.logger and current_time - self.last_log_time > 2:
                self.logger.add_log(
                    f"[{time.strftime('%H:%M:%S')}] Person detected ({person_count})"
                )
                self.last_log_time = current_time

        # 🔴 Emergency placeholder
        if self.emergency.isChecked():
            # you will plug MediaPipe here
            pass

        # FPS
        cv2.putText(frame, "FPS: 18", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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