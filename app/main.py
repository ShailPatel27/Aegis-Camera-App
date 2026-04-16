import sys
from PyQt5.QtWidgets import QApplication
from app.window import MainWindow

def run_app():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QMainWindow {
        background-color: #0f172a;
    }

    QWidget {
        color: white;
        font-size: 14px;
    }

    QListWidget {
        background-color: #020617;
        border: none;
        padding: 10px;
    }

    QListWidget::item {
        padding: 12px;
        border-radius: 6px;
    }

    QListWidget::item:selected {
        background-color: #2563eb;
    }

    QLabel {
        color: white;
    }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())