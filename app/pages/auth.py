from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QInputDialog,
)

from app.services.auth_client import auth_client


class AuthPage(QWidget):
    auth_success = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(
            """
            QLineEdit {
                background-color: #f8fafc;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 10px 12px;
            }
            QLineEdit:focus {
                border: 1px solid #2563eb;
            }
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
        root.setSpacing(14)
        self.setLayout(root)

        title = QLabel("AEGIS CAMERA LOGIN")
        title.setStyleSheet("font-size: 24px; font-weight: 800;")
        subtitle = QLabel("Sign in or create an account to continue.")
        subtitle.setStyleSheet("color: #94a3b8;")
        root.addWidget(title, alignment=Qt.AlignHCenter)
        root.addWidget(subtitle, alignment=Qt.AlignHCenter)

        tabs = QTabWidget()
        tabs.setStyleSheet(
            """
            QTabBar::tab {
                padding: 10px 18px;
                background: #0b1224;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #1d4ed8;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #334155;
                border-radius: 10px;
                background: #0b1224;
                top: -1px;
            }
            """
        )
        root.addWidget(tabs, 1)

        self.login_tab = QWidget()
        self.signup_tab = QWidget()
        tabs.addTab(self.login_tab, "Login")
        tabs.addTab(self.signup_tab, "Sign Up")

        self._build_login_tab()
        self._build_signup_tab()

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #94a3b8;")
        root.addWidget(self.status)

    def _build_login_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        self.login_tab.setLayout(layout)

        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("Email")
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.login_email)
        layout.addWidget(self.login_password)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.login_btn = QPushButton("Login")
        self.login_btn.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; border: 1px solid #15803d; }"
            "QPushButton:hover { background-color: #15803d; }"
        )
        self.login_btn.clicked.connect(self._do_login)
        btn_row.addWidget(self.login_btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    def _build_signup_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        self.signup_tab.setLayout(layout)

        self.signup_name = QLineEdit()
        self.signup_name.setPlaceholderText("Full name")
        self.signup_email = QLineEdit()
        self.signup_email.setPlaceholderText("Email")
        self.signup_password = QLineEdit()
        self.signup_password.setPlaceholderText("Password")
        self.signup_password.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.signup_name)
        layout.addWidget(self.signup_email)
        layout.addWidget(self.signup_password)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.signup_btn = QPushButton("Create Account")
        self.signup_btn.setStyleSheet(
            "QPushButton { background-color: #1d4ed8; color: white; border: 1px solid #1e40af; }"
            "QPushButton:hover { background-color: #1e40af; }"
        )
        self.signup_btn.clicked.connect(self._do_signup)
        btn_row.addWidget(self.signup_btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    def _set_status(self, text, ok=False):
        self.status.setStyleSheet(f"color: {'#86efac' if ok else '#fca5a5'};")
        self.status.setText(text)

    def _do_login(self):
        email = self.login_email.text().strip()
        password = self.login_password.text().strip()
        if not email or not password:
            self._set_status("Please enter email and password.")
            return
        try:
            session = auth_client.login(email, password)
            session = self._ensure_camera_named(session)
            if not session:
                return
            self._set_status("Login successful.", ok=True)
            self.auth_success.emit(session)
        except Exception as exc:
            self._set_status(str(exc))

    def _do_signup(self):
        name = self.signup_name.text().strip()
        email = self.signup_email.text().strip()
        password = self.signup_password.text().strip()
        if not name or not email or not password:
            self._set_status("Please fill name, email, and password.")
            return
        try:
            session = auth_client.register(name, email, password)
            session = self._ensure_camera_named(session)
            if not session:
                return
            self._set_status("Account created.", ok=True)
            self.auth_success.emit(session)
        except Exception as exc:
            self._set_status(str(exc))

    def _ensure_camera_named(self, session):
        if not session.get("needs_camera_name"):
            return session

        token = session.get("token")
        user = session.get("user") or {}
        suggested = f"{user.get('name', '').strip()} Camera".strip()
        if suggested == "Camera":
            suggested = ""

        dialog = QInputDialog(self)
        dialog.setWindowTitle("Create Camera")
        dialog.setLabelText("No camera found for this account.\nEnter a camera name:")
        dialog.setTextValue(suggested)
        dialog.setStyleSheet(
            """
            QInputDialog {
                background-color: #0f172a;
            }
            QLabel {
                color: #e2e8f0;
            }
            QLineEdit {
                background-color: #f8fafc;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 8px;
            }
            QPushButton {
                background-color: #1d4ed8;
                color: white;
                border: 1px solid #1e40af;
                border-radius: 6px;
                padding: 6px 10px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #1e40af;
            }
            """
        )
        ok = dialog.exec_()
        camera_name = dialog.textValue().strip()

        if not ok:
            self._set_status("Camera setup canceled. Please provide a camera name to continue.")
            return None
        if not camera_name:
            self._set_status("Camera name is required.")
            return None

        return auth_client.complete_camera_setup(token, user, camera_name)
