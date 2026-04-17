from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame


class AccountPage(QWidget):
    logout_requested = pyqtSignal()

    def __init__(self, user=None):
        super().__init__()
        self.user = user or {}
        self._build_ui()
        self.set_user(self.user)

    def _build_ui(self):
        self.setStyleSheet(
            """
            QLabel { color: #e2e8f0; }
            QPushButton {
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: 700;
                outline: none;
            }
            QPushButton:focus { outline: none; }
            """
        )

        root = QVBoxLayout()
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)
        self.setLayout(root)

        title = QLabel("Account")
        title.setStyleSheet("font-size: 20px; font-weight: 800;")
        root.addWidget(title)

        self.name_label = QLabel()
        self.email_label = QLabel()
        self.phone_label = QLabel()
        for label in (self.name_label, self.email_label, self.phone_label):
            label.setStyleSheet("font-size: 14px; color: #cbd5e1;")
            root.addWidget(label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #334155;")
        root.addWidget(line)

        self.logout_btn = QPushButton("Logout")
        self.logout_btn.setStyleSheet(
            "QPushButton { background-color: #991b1b; color: #fee2e2; border: 1px solid #7f1d1d; }"
            "QPushButton:hover { background-color: #7f1d1d; }"
        )
        self.logout_btn.clicked.connect(self.logout_requested.emit)
        root.addWidget(self.logout_btn, alignment=Qt.AlignLeft)
        root.addStretch()

    def set_user(self, user):
        self.user = user or {}
        self.name_label.setText(f"Name: {self.user.get('name', '-')}")
        self.email_label.setText(f"Email: {self.user.get('email', '-')}")
        phone = self.user.get("phone") or self.user.get("alternate_contact") or "-"
        self.phone_label.setText(f"Contact: {phone}")
