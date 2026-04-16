from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout

class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Settings"))
        self.setLayout(layout)