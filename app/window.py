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

from app.pages.live import LivePage
from app.pages.logs import LogsPage
from app.pages.register import RegisterPage
from app.pages.emergency import EmergencyPage
from app.pages.settings import SettingsPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Aegis Camera")
        self.resize(1200, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_widget.setLayout(main_layout)

        title = QLabel("AEGIS CAMERA SYSTEM")
        title.setStyleSheet("font-size: 18px; padding: 12px;")
        main_layout.addWidget(title)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        main_layout.addLayout(content)

        self.sidebar = QListWidget()
        self.sidebar.addItems(["Live Feed", "Logs", "Register", "Emergency", "Settings"])
        self.sidebar.setFixedWidth(200)
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

        self.stack.addWidget(self.live_page)
        self.stack.addWidget(self.logs_page)
        self.stack.addWidget(self.register_page)
        self.stack.addWidget(self.emergency_page)
        self.stack.addWidget(self.settings_page)

        self.live_page.worker_changed.connect(self.register_page.set_live_worker)
        self.live_page.worker_changed.connect(self.emergency_page.set_live_worker)
        self.register_page.set_live_worker(self.live_page.worker)
        self.emergency_page.set_live_worker(self.live_page.worker)

        self.sidebar.currentRowChanged.connect(self.on_nav_changed)
        self.sidebar.setCurrentRow(0)

        content.addWidget(self.sidebar)
        content.addWidget(self.stack)

    def on_nav_changed(self, index):
        self.stack.setCurrentIndex(index)

    def closeEvent(self, event):
        self.register_page.set_live_worker(None)
        self.emergency_page.set_live_worker(None)
        self.live_page.stop_worker()
        super().closeEvent(event)
