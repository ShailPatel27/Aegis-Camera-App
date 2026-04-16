from PyQt5.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QStackedWidget

from app.pages.live import LivePage
from app.pages.logs import LogsPage
from app.pages.register import RegisterPage
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

        # 🔹 Title
        title = QLabel("AEGIS CAMERA SYSTEM")
        title.setStyleSheet("font-size: 18px; padding: 12px;")
        main_layout.addWidget(title)

        # 🔹 Content Layout
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        main_layout.addLayout(content)

        # 🔹 Sidebar (CREATE FIRST)
        self.sidebar = QListWidget()
        self.sidebar.addItems(["🏠 Live Feed", "📜 Logs", "👤 Register", "⚙ Settings"])
        self.sidebar.setFixedWidth(200)

        # 🔹 Pages
        self.stack = QStackedWidget()
        
        # Create pages FIRST
        self.logs_page = LogsPage()
        self.live_page = LivePage(self.logs_page)

        self.register_page = RegisterPage()
        self.settings_page = SettingsPage()

        # Add to stack
        self.stack.addWidget(self.live_page)
        self.stack.addWidget(self.logs_page)
        self.stack.addWidget(self.register_page)
        self.stack.addWidget(self.settings_page)

        # 🔹 Connect
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

        # 🔹 Add to layout
        content.addWidget(self.sidebar)
        content.addWidget(self.stack)