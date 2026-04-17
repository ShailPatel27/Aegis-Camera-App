from PyQt5.QtCore import Qt
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
        user = session.get("user", {})

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

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

        self.stack = QStackedWidget()

        self.logs_page = LogsPage()
        self.live_page = LivePage(self.logs_page)
        self.register_page = RegisterPage(self.logs_page)
        self.emergency_page = EmergencyPage(self.logs_page)
        self.settings_page = SettingsPage()
        self.account_page = AccountPage(user)

        self.stack.addWidget(self.live_page)
        self.stack.addWidget(self.logs_page)
        self.stack.addWidget(self.register_page)
        self.stack.addWidget(self.emergency_page)
        self.stack.addWidget(self.settings_page)
        self.stack.addWidget(self.account_page)

        self.live_page.worker_changed.connect(self.register_page.set_live_worker)
        self.live_page.worker_changed.connect(self.emergency_page.set_live_worker)
        self.register_page.set_live_worker(self.live_page.worker)
        self.emergency_page.set_live_worker(self.live_page.worker)
        self.account_page.logout_requested.connect(self._logout)

        self.sidebar.currentRowChanged.connect(self.on_nav_changed)
        self.sidebar.setCurrentRow(0)

        content.addWidget(self.sidebar)
        content.addWidget(self.stack)

    def _on_auth_success(self, session):
        self._show_app(session)

    def _logout(self):
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
