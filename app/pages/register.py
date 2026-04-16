from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout

class RegisterPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Face Registration"))
        self.setLayout(layout)