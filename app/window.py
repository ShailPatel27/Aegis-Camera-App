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
        self.stack.addWidget(LivePage())
        self.stack.addWidget(LogsPage())
        self.stack.addWidget(RegisterPage())
        self.stack.addWidget(SettingsPage())

        # 🔹 Connect
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

        # 🔹 Add to layout
        content.addWidget(self.sidebar)
        content.addWidget(self.stack)