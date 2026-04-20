from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget
from pathlib import Path
from config.settings import LOCAL_LOG_FILE_PATH, LOCAL_LOG_UI_MAX_ITEMS, LOCAL_LOG_FILE_MAX_LINES

class LogsPage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.log_list = QListWidget()
        self.log_list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.log_list)

    def add_log(self, message):
        self.log_list.insertItem(0, message)
        while self.log_list.count() > LOCAL_LOG_UI_MAX_ITEMS:
            self.log_list.takeItem(self.log_list.count() - 1)

        log_path = Path(LOCAL_LOG_FILE_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(message + "\n")

        # Keep file bounded to prevent unbounded growth.
        if LOCAL_LOG_FILE_MAX_LINES > 0:
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
                if len(lines) > LOCAL_LOG_FILE_MAX_LINES:
                    lines = lines[-LOCAL_LOG_FILE_MAX_LINES:]
                    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except Exception:
                pass
