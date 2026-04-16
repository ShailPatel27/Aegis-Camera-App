from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout

class LogsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Logs"))
        self.setLayout(layout)