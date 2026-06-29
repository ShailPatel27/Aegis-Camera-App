import sys
import faulthandler
import traceback
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QApplication
from app.window import MainWindow

_FAULT_LOG = None


def _write_crash_log(exc_type, exc_value, exc_traceback):
    try:
        log_path = Path("data/logs/crash.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write("\n" + "=" * 80 + "\n")
            fp.write(datetime.now().isoformat(timespec="seconds") + "\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=fp)
    except Exception:
        pass


def _install_crash_logger():
    global _FAULT_LOG
    try:
        fault_path = Path("data/logs/crash_native.log")
        fault_path.parent.mkdir(parents=True, exist_ok=True)
        _FAULT_LOG = fault_path.open("a", encoding="utf-8")
        faulthandler.enable(file=_FAULT_LOG, all_threads=True)
    except Exception:
        _FAULT_LOG = None

    def _hook(exc_type, exc_value, exc_traceback):
        _write_crash_log(exc_type, exc_value, exc_traceback)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _hook


def run_app():
    _install_crash_logger()
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
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception:
        _write_crash_log(*sys.exc_info())
        raise
