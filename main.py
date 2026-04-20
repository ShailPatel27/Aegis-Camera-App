import os
import sys

from app.main import run_app

if __name__ == "__main__":
    # When packaged with PyInstaller, run relative to the executable directory
    # so paths like models/, data/, and bin/ resolve correctly on double-click.
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    run_app()
