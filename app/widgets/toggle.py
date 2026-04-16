from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtCore import Qt

class ToggleSwitch(QCheckBox):
    def __init__(self, label=""):
        super().__init__(label)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)

        self.setStyleSheet("""
        QCheckBox {
            color: white;
            font-size: 13px;
            spacing: 10px;
        }

        QCheckBox::indicator {
            width: 40px;
            height: 20px;
            border-radius: 10px;
            background-color: #475569;
        }

        QCheckBox::indicator:checked {
            background-color: #2563eb;
        }

        QCheckBox::indicator:unchecked {
            background-color: #475569;
        }
        """)
