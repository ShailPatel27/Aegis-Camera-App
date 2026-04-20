# AEGIS Camera App

AEGIS Camera is the desktop capture and AI node for the AEGIS system.  
It runs on Windows, controls local camera feed, performs detections, and syncs events/chunks to the Monitor backend.

## What It Does

- Live camera feed with start/pause/stop
- AI toggles (intrusion, crowd, vehicle, threat, motion, loiter, emergency, face recognition)
- Face registration and local face database
- Chunk recording/upload pipeline
- Alert and analytics sync to monitor backend

## Project Layout

- `main.py`: app entry point
- `app/`: UI pages and app services
- `core/`: AI worker, detector, motion, face engine
- `config/settings.py`: runtime settings and defaults
- `models/`: YOLO / landmark models
- `bin/`: ffmpeg binary
- `data/`: local runtime data (faces, logs, chunks, auth)

## Run (Development)

1. Create and activate venv
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies
```powershell
pip install -r requirements.txt
```

3. Configure `.env` (backend URL, Supabase keys if used)

4. Start app
```powershell
python main.py
```

## Build Windows EXE (Double-click App)

This repo includes PyInstaller setup.

1. Ensure dependencies are installed in `venv`
2. Build:
```powershell
build_windows.bat
```

3. Output:
- `dist\AEGIS Camera\AEGIS Camera.exe`

4. Run:
- Double-click `AEGIS Camera.exe`

## Create Desktop Icon (Shortcut)

After build:

1. Open `dist\AEGIS Camera\`
2. Right-click `AEGIS Camera.exe` -> `Send to` -> `Desktop (create shortcut)`
3. On Desktop, rename shortcut to `AEGIS Camera`
4. (Optional) Right-click shortcut -> `Properties` -> `Change Icon` and choose:
- `assets\icons\icon.ico`

Now you can start AEGIS Camera with one desktop double-click.

## Notes

- When packaged, app auto-sets working directory to executable location, so relative paths keep working.
- If models/assets/config change, rebuild the EXE.

