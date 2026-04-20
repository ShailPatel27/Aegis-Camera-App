@echo off
setlocal

echo Building AEGIS Camera Windows app...

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

if not exist venv\Scripts\python.exe (
  echo ERROR: venv not found. Create it first.
  exit /b 1
)

venv\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1

venv\Scripts\python.exe -m PyInstaller --noconfirm aegis_camera.spec
if errorlevel 1 exit /b 1

echo.
echo Build complete.
echo Run this file to launch:
echo dist\AEGIS Camera\AEGIS Camera.exe

endlocal
