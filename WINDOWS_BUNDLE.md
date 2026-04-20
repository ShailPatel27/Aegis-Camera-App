# Windows Bundle (AEGIS Camera)

This project can be packaged into a double-clickable Windows app using PyInstaller.

## 1) Prerequisites

- Windows 10/11
- Python virtual environment at `venv`
- App dependencies installed (`requirements.txt`)

## 2) Build

Run:

```bat
build_windows.bat
```

Output:

- `dist\AEGIS Camera\AEGIS Camera.exe`

## 3) Run

Double-click:

- `dist\AEGIS Camera\AEGIS Camera.exe`

## 4) Notes

- The bundle includes `models/`, `assets/`, `bin/` and required `data` folders.
- `main.py` auto-switches working directory to the executable path when frozen, so relative paths keep working.
- If you change model files or assets, rebuild.

