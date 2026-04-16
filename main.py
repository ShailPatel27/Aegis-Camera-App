import cv2
import time
import os
import subprocess
import threading
from queue import Queue
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
BUCKET = os.getenv("BUCKET")

USER_ID = "demo-user"
CAM_ID = "main-cam"

FPS = 10
CHUNK_SECONDS = 5
FRAMES_PER_CHUNK = FPS * CHUNK_SECONDS

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

os.makedirs("chunks", exist_ok=True)

# 🔥 QUEUE FOR BACKGROUND PROCESSING
task_queue = Queue()

def worker():
    while True:
        avi_path, timestamp = task_queue.get()

        mp4_path = avi_path.replace(".avi", ".mp4")

        # 🔁 encode
        subprocess.run([
            "./ffmpeg.exe",
            "-y",
            "-i", avi_path,
            "-r", str(FPS),
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-vcodec", "libx264",
            mp4_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(mp4_path):
            remote_path = f"{USER_ID}/{CAM_ID}/video_{timestamp}.mp4"

            try:
                with open(mp4_path, "rb") as f:
                    supabase.storage.from_(BUCKET).upload(
                        remote_path,
                        f,
                        {"content-type": "video/mp4"}
                    )
                print("Uploaded:", remote_path)
            except Exception as e:
                print("Upload failed:", e)

        # cleanup
        try:
            os.remove(avi_path)
            os.remove(mp4_path)
        except:
            pass

        task_queue.task_done()

# 🔥 START WORKER THREAD
threading.Thread(target=worker, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Recording Started...")

while True:
    timestamp = int(time.time())
    avi_path = f"chunks/video_{timestamp}.avi"

    out = cv2.VideoWriter(
        avi_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        FPS,
        (width, height)
    )

    frame_count = 0

    # 🎥 RECORD (NO BLOCKING NOW)
    frame_interval = 1.0 / FPS
    next_frame_time = time.time()

    while frame_count < FRAMES_PER_CHUNK:
        ret, frame = cap.read()
        if not ret:
            continue

        out.write(frame)
        frame_count += 1

        # 🔥 enforce real-time pacing
        next_frame_time += frame_interval
        sleep_time = next_frame_time - time.time()

        if sleep_time > 0:
            time.sleep(sleep_time)

    out.release()

    print("Chunk recorded:", avi_path)

    # 🔥 SEND TO BACKGROUND
    task_queue.put((avi_path, timestamp))

cap.release()