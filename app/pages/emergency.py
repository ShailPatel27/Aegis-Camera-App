import cv2

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
)

from config.settings import EMERGENCY_FINGER_NAMES, EMERGENCY_PATTERN_STEPS
from core.emergency import EmergencyPatternStore, HandPatternDetector


class EmergencyPage(QWidget):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger
        self.live_worker = None
        self.detector = HandPatternDetector()
        self.store = EmergencyPatternStore()

        self.current_pattern = None
        self.captured_steps = []
        self.saved_steps = self.store.load_steps()

        self.setStyleSheet(
            """
            QPushButton {
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 700;
                outline: none;
            }
            QPushButton:focus {
                outline: none;
            }
            """
        )

        root = QVBoxLayout()
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)
        self.setLayout(root)

        title = QLabel("Emergency Pattern Setup")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(title)

        subtitle = QLabel(
            "Show one hand to camera. We detect open/closed fingers from landmarks.\n"
            "If it looks right, click Confirm. Confirm 4 times to auto-save global pattern."
        )
        subtitle.setStyleSheet("color: #94a3b8;")
        root.addWidget(subtitle)

        self.preview = QLabel("Feed not running. Start feed from Live page.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 380)
        self.preview.setStyleSheet("background: #000; border-radius: 10px; color: #94a3b8;")
        root.addWidget(self.preview, 1)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        root.addLayout(controls)

        self.confirm_btn = QPushButton("Confirm Step")
        self.confirm_btn.setFocusPolicy(Qt.NoFocus)
        self.confirm_btn.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; border: 1px solid #15803d; }"
            "QPushButton:hover { background-color: #15803d; }"
            "QPushButton:disabled { background-color: #14532d; color: #bbf7d0; border: 1px solid #14532d; }"
        )
        self.confirm_btn.clicked.connect(self.confirm_step)
        controls.addWidget(self.confirm_btn)

        self.redo_btn = QPushButton("Redo Pattern")
        self.redo_btn.setFocusPolicy(Qt.NoFocus)
        self.redo_btn.setStyleSheet(
            "QPushButton { background-color: #7f1d1d; color: #fee2e2; border: 1px solid #991b1b; }"
            "QPushButton:hover { background-color: #991b1b; }"
        )
        self.redo_btn.clicked.connect(self.redo_pattern)
        controls.addWidget(self.redo_btn)
        controls.addStretch()

        self.detected_label = QLabel("")
        self.detected_label.setStyleSheet("color: #e2e8f0;")
        root.addWidget(self.detected_label)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #94a3b8;")
        root.addWidget(self.progress_label)

        self.current_seq_label = QLabel("")
        self.current_seq_label.setWordWrap(True)
        self.current_seq_label.setStyleSheet("color: #e2e8f0;")
        root.addWidget(self.current_seq_label)

        self.saved_seq_label = QLabel("")
        self.saved_seq_label.setWordWrap(True)
        self.saved_seq_label.setStyleSheet("color: #94a3b8;")
        root.addWidget(self.saved_seq_label)

        self.status_label = QLabel("Waiting for hand input.")
        self.status_label.setStyleSheet("color: #94a3b8;")
        root.addWidget(self.status_label)

        self._refresh_labels()

    def _log(self, message):
        if self.logger:
            self.logger.add_log(message)

    @staticmethod
    def _state_text(state_tuple):
        if state_tuple is None:
            return "(no hand)"
        return "".join("1" if value else "0" for value in state_tuple)

    def _named_state_text(self, state_tuple):
        if state_tuple is None:
            return "No hand detected"
        parts = []
        for i, name in enumerate(EMERGENCY_FINGER_NAMES):
            parts.append(f"{name}:{'open' if state_tuple[i] else 'closed'}")
        return " | ".join(parts)

    def _refresh_labels(self):
        self.progress_label.setText(
            f"Emergency setup progress: {len(self.captured_steps)}/{EMERGENCY_PATTERN_STEPS}"
        )
        self.confirm_btn.setEnabled(len(self.captured_steps) < EMERGENCY_PATTERN_STEPS)

        if self.captured_steps:
            current = ", ".join(
                f"S{i + 1}:{self._state_text(step)}" for i, step in enumerate(self.captured_steps)
            )
            self.current_seq_label.setText(f"Current Pattern: {current}")
        else:
            self.current_seq_label.setText("Current Pattern: (empty)")

        if self.saved_steps:
            saved = ", ".join(
                f"S{i + 1}:{self._state_text(step)}" for i, step in enumerate(self.saved_steps)
            )
            self.saved_seq_label.setText(f"Saved Global Pattern: {saved}")
        else:
            self.saved_seq_label.setText("Saved Global Pattern: (none)")

        self.detected_label.setText(
            f"Detected Hand State: {self._named_state_text(self.current_pattern)}"
        )

    def set_live_worker(self, worker):
        if self.live_worker is not None:
            try:
                self.live_worker.raw_frame_ready.disconnect(self.on_live_frame)
            except Exception:
                pass

        self.live_worker = worker
        if self.live_worker is not None:
            self.live_worker.raw_frame_ready.connect(self.on_live_frame)
            self.status_label.setText("Live feed connected. Show hand and press Confirm.")
        else:
            self.status_label.setText("Feed stopped. Start feed from Live page.")
            self.current_pattern = None
            self._refresh_labels()

    def on_live_frame(self, frame):
        display = frame.copy()
        self.current_pattern = self.detector.detect(display, draw=True)
        self._refresh_labels()

        cv2.putText(
            display,
            f"Detected: {self._state_text(self.current_pattern)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"Setup {len(self.captured_steps)}/{EMERGENCY_PATTERN_STEPS}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )

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

    def confirm_step(self):
        if self.current_pattern is None:
            self.status_label.setText("No hand detected. Show one hand and try again.")
            return
        if len(self.captured_steps) >= EMERGENCY_PATTERN_STEPS:
            self.status_label.setText("All steps already captured. Redo to start again.")
            return

        self.captured_steps.append(tuple(self.current_pattern))
        self.status_label.setText(
            f"Captured Step {len(self.captured_steps)}/{EMERGENCY_PATTERN_STEPS}."
        )

        if len(self.captured_steps) == EMERGENCY_PATTERN_STEPS:
            self.store.save_steps(self.captured_steps)
            self.saved_steps = list(self.captured_steps)
            self.status_label.setText("Pattern captured and saved globally.")
            self._log("[EMERGENCY] Global emergency pattern saved.")

        self._refresh_labels()

    def redo_pattern(self):
        self.captured_steps = []
        self.status_label.setText("Pattern reset. Start confirming from Step 1.")
        self._refresh_labels()
