from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget
from pathlib import Path
from config.settings import LOCAL_LOG_FILE_PATH, LOCAL_LOG_UI_MAX_ITEMS

class LogsPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.log_list = QListWidget()
        layout.addWidget(self.log_list)

    def add_log(self, message):
        self.log_list.insertItem(0, message)
        while self.log_list.count() > LOCAL_LOG_UI_MAX_ITEMS:
            self.log_list.takeItem(self.log_list.count() - 1)

        log_path = Path(LOCAL_LOG_FILE_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(message + "\n")
