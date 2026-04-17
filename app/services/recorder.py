import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Tuple
from io import BytesIO

import cv2
from dotenv import load_dotenv
from supabase import create_client

from config.settings import FFMPEG_PATH, RECORD_CHUNK_SECONDS, RECORD_FPS


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

FPS = RECORD_FPS
CHUNK_SECONDS = RECORD_CHUNK_SECONDS
FRAMES_PER_CHUNK = FPS * CHUNK_SECONDS
RETENTION_SECONDS = 5 * 60
LOCAL_CHUNKS_DIR = Path("chunks")
SESSION_PATH = Path("data/auth/session.json")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
task_queue: "Queue[Tuple[Path, int, str, str]]" = Queue()


def _safe_path_part(value: str, fallback: str) -> str:
    cleaned = (value or "").strip().replace("\\", "_").replace("/", "_")
    return cleaned or fallback


def _load_session_context() -> Tuple[str, int]:
    if not SESSION_PATH.exists():
        raise RuntimeError(
            "Session file missing. Login in the camera app first so uploader can resolve camera context."
        )

    payload = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    camera = payload.get("camera", {}) if isinstance(payload, dict) else {}

    camera_id = _safe_path_part(str(camera.get("id") or ""), "camera")
    # Supabase bucket names should be lowercase and URL-safe.
    bucket_name = re.sub(r"[^a-z0-9_-]", "-", camera_id.lower()).strip("-")
    if not bucket_name:
        raise RuntimeError("Invalid camera id for bucket name. Ensure camera has a valid id.")

    camera_index = int(camera.get("selected_camera", 0) or 0)
    return bucket_name, camera_index


def _chunk_prefix() -> str:
    return "stream-chunks"


def _ensure_bucket_and_prefix(bucket_name: str, prefix: str) -> None:
    storage = supabase.storage
    try:
        existing = storage.list_buckets()
    except Exception as exc:
        print("Bucket discovery failed:", exc)
        existing = []

    bucket_exists = False
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, dict) and item.get("name") == bucket_name:
                bucket_exists = True
                break

    if not bucket_exists:
        try:
            storage.create_bucket(
                bucket_name,
                name=bucket_name,
                options={"public": True, "file_size_limit": 52428800},
            )
            print(f"Created bucket: {bucket_name}")
        except Exception as exc:
            # Ignore "already exists" races.
            if "exists" not in str(exc).lower():
                raise

    try:
        storage.from_(bucket_name).upload(
            f"{prefix}/.keep",
            BytesIO(b""),
            {"content-type": "application/octet-stream", "upsert": "true"},
        )
    except Exception:
        # Marker is optional.
        pass


def _cleanup_remote_chunks(bucket_name: str, prefix: str, current_ts: int) -> None:
    cutoff_ts = current_ts - RETENTION_SECONDS
    storage = supabase.storage.from_(bucket_name)
    try:
        entries = storage.list(
            prefix,
            {"limit": 1000, "offset": 0, "sortBy": {"column": "name", "order": "asc"}},
        )
    except Exception as exc:
        print("Remote cleanup skipped (list failed):", exc)
        return

    if not isinstance(entries, list):
        return

    delete_paths = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", ""))
        match = re.match(r"^chunk_(\d+)\.mp4$", name)
        if not match:
            continue
        ts = int(match.group(1))
        if ts <= cutoff_ts:
            delete_paths.append(f"{prefix}/{name}")

    if not delete_paths:
        return

    try:
        storage.remove(delete_paths)
        print(f"Cleanup removed {len(delete_paths)} old chunk(s)")
    except Exception as exc:
        print("Remote cleanup skipped (remove failed):", exc)


def _encode_and_upload_worker() -> None:
    while True:
        avi_path, timestamp, bucket_name, prefix = task_queue.get()
        mp4_path = avi_path.with_suffix(".mp4")

        try:
            subprocess.run(
                [
                    FFMPEG_PATH,
                    "-y",
                    "-i",
                    str(avi_path),
                    "-r",
                    str(FPS),
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                    "-profile:v",
                    "baseline",
                    "-level",
                    "3.0",
                    "-movflags",
                    "frag_keyframe+empty_moov+default_base_moof",
                    "-vcodec",
                    "libx264",
                    str(mp4_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            if mp4_path.exists():
                remote_path = f"{prefix}/chunk_{timestamp}.mp4"
                try:
                    with mp4_path.open("rb") as fp:
                        supabase.storage.from_(bucket_name).upload(
                            remote_path,
                            fp,
                            {"content-type": "video/mp4"},
                        )
                except Exception as exc:
                    # Self-heal once if bucket/prefix is missing.
                    if "bucket" in str(exc).lower() or "not found" in str(exc).lower():
                        _ensure_bucket_and_prefix(bucket_name, prefix)
                        with mp4_path.open("rb") as fp:
                            supabase.storage.from_(bucket_name).upload(
                                remote_path,
                                fp,
                                {"content-type": "video/mp4"},
                            )
                    else:
                        raise
                print("Uploaded:", remote_path)
                _cleanup_remote_chunks(bucket_name, prefix, timestamp)
        except Exception as exc:
            print("Chunk processing failed:", exc)
        finally:
            try:
                if avi_path.exists():
                    avi_path.unlink()
                if mp4_path.exists():
                    mp4_path.unlink()
            except Exception:
                pass
            task_queue.task_done()


def run() -> None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY in .env")

    bucket_name, camera_index = _load_session_context()
    prefix = _chunk_prefix()
    _ensure_bucket_and_prefix(bucket_name, prefix)

    LOCAL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_encode_and_upload_worker, daemon=True).start()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Recording started for bucket '{bucket_name}' / {prefix} at camera index {camera_index}")

    try:
        while True:
            timestamp = int(time.time())
            avi_path = LOCAL_CHUNKS_DIR / f"chunk_{timestamp}.avi"
            writer = cv2.VideoWriter(
                str(avi_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                FPS,
                (width, height),
            )

            frame_count = 0
            frame_interval = 1.0 / FPS
            next_frame_time = time.time()

            while frame_count < FRAMES_PER_CHUNK:
                ok, frame = cap.read()
                if not ok:
                    continue

                writer.write(frame)
                frame_count += 1

                next_frame_time += frame_interval
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            writer.release()
            print("Chunk recorded:", avi_path.as_posix())
            task_queue.put((avi_path, timestamp, bucket_name, prefix))
    finally:
        cap.release()


if __name__ == "__main__":
    run()
