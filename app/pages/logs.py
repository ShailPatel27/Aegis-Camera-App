from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget

class LogsPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.log_list = QListWidget()
        layout.addWidget(self.log_list)

    def add_log(self, message):
        self.log_list.insertItem(0, message)