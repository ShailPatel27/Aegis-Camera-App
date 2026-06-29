from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
)

from app.pages.auth import AuthPage
from app.pages.live import LivePage
from app.pages.logs import LogsPage
from app.pages.register import RegisterPage
from app.pages.emergency import EmergencyPage
from app.pages.settings import SettingsPage
from app.pages.account import AccountPage
from app.services.auth_client import auth_client


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Aegis Camera")
        self.resize(1200, 700)
        self.session = None
        self.auth_page = None
        self.live_page = None
        self.logs_page = None
        self.register_page = None
        self.emergency_page = None
        self.settings_page = None
        self.account_page = None
        self.sidebar = None
        self.stack = None

        session = auth_client.load_session()
        if session:
            self._show_app(session)
        else:
            self._show_auth()

    def _show_auth(self):
        self._teardown_app_pages()
        self.auth_page = AuthPage()
        self.auth_page.auth_success.connect(self._on_auth_success)
        self.setCentralWidget(self.auth_page)

    def _show_app(self, session):
        self.session = session
        user = dict(session.get("user", {}))
        user["camera"] = session.get("camera", {})

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_widget.setLayout(main_layout)

        title = QLabel(f"AEGIS CAMERA SYSTEM  |  {user.get('name', 'User')}")
        title.setStyleSheet("font-size: 18px; padding: 12px;")
        main_layout.addWidget(title)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        main_layout.addLayout(content)

        self.sidebar = QListWidget()
        self.sidebar.addItems(["Live Feed", "Logs", "Register", "Emergency", "Settings", "Account"])
        self.sidebar.setFixedWidth(220)
        self.sidebar.setFocusPolicy(Qt.NoFocus)
        self.sidebar.setStyleSheet(
            """
            QListWidget::item:selected {
                outline: none;
                border: none;
            }
            QListWidget::item:selected:active {
                outline: none;
                border: none;
            }
            QListWidget::item:selected:!active {
                outline: none;
                border: none;
            }
            QListWidget::item:focus {
                outline: none;
                border: none;
            }
            """
        )

        stack = QStackedWidget()
        logs_page = None
        live_page = None
        register_page = None
        emergency_page = None
        settings_page = None
        account_page = None
        try:
            logs_page = LogsPage()
            live_page = LivePage(logs_page, session=session)
            register_page = RegisterPage(logs_page)
            emergency_page = EmergencyPage(logs_page)
            settings_page = SettingsPage()
            account_page = AccountPage(user)

            stack.addWidget(live_page)
            stack.addWidget(logs_page)
            stack.addWidget(register_page)
            stack.addWidget(emergency_page)
            stack.addWidget(settings_page)
            stack.addWidget(account_page)

            live_page.worker_changed.connect(register_page.set_live_worker)
            live_page.worker_changed.connect(emergency_page.set_live_worker)
            live_page.session_invalid.connect(self._handle_session_invalid)
            register_page.set_live_worker(live_page.worker)
            emergency_page.set_live_worker(live_page.worker)
            account_page.logout_requested.connect(self._logout)

            self.sidebar.currentRowChanged.connect(self.on_nav_changed)
            self.sidebar.setCurrentRow(0)

            content.addWidget(self.sidebar)
            content.addWidget(stack)
        except Exception:
            if live_page is not None:
                live_page.stop_worker()
            raise

        self.logs_page = logs_page
        self.live_page = live_page
        self.register_page = register_page
        self.emergency_page = emergency_page
        self.settings_page = settings_page
        self.account_page = account_page
        self.stack = stack
        self.setCentralWidget(main_widget)

    def _on_auth_success(self, session):
        QTimer.singleShot(0, lambda: self._finish_auth_success(session))

    def _finish_auth_success(self, session):
        try:
            self._show_app(session)
        except Exception as exc:
            if self.auth_page is not None:
                self.auth_page._set_status(f"Login succeeded, but the app failed to open: {exc}")
            else:
                raise

    def _logout(self):
        auth_client.clear_session()
        self._show_auth()

    def _handle_session_invalid(self, _reason):
        auth_client.clear_session()
        self._show_auth()

    def _teardown_app_pages(self):
        if self.register_page is not None:
            self.register_page.set_live_worker(None)
        if self.emergency_page is not None:
            self.emergency_page.set_live_worker(None)
        if self.live_page is not None:
            self.live_page.stop_worker()

        self.live_page = None
        self.logs_page = None
        self.register_page = None
        self.emergency_page = None
        self.settings_page = None
        self.account_page = None
        self.sidebar = None
        self.stack = None

    def on_nav_changed(self, index):
        if self.stack is not None:
            self.stack.setCurrentIndex(index)

    def closeEvent(self, event):
        self._teardown_app_pages()
        super().closeEvent(event)
