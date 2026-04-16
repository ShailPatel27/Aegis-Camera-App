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
        outline: none;
    }

    QListWidget::item {
        padding: 12px;
        border-radius: 6px;
        outline: none;
        border: none;
    }

    QListWidget::item:selected {
        background-color: #2563eb;
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

    QListView {
        outline: none;
    }

    QListView::item {
        outline: none;
        border: none;
    }

    QListView::item:selected {
        outline: none;
        border: none;
    }

    QListView::item:selected:active {
        outline: none;
        border: none;
    }

    QListView::item:selected:!active {
        outline: none;
        border: none;
    }

    QLabel {
        color: white;
    }

    QPushButton {
        outline: none;
    }

    QPushButton:focus {
        outline: none;
    }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
