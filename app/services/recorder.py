import json
import os
import re
import subprocess
import threading
import time
from io import BytesIO
from pathlib import Path
from queue import Queue
from typing import Optional, Tuple

import cv2
from dotenv import load_dotenv
from supabase import create_client

from config.settings import FFMPEG_PATH, RECORD_CHUNK_SECONDS, RECORD_FPS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

FPS = RECORD_FPS
CHUNK_SECONDS = RECORD_CHUNK_SECONDS
FRAMES_PER_CHUNK = FPS * CHUNK_SECONDS
RETENTION_SECONDS = 5 * 60
LOCAL_CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
SESSION_PATH = PROJECT_ROOT / "data" / "auth" / "session.json"


def _resolve_ffmpeg_path() -> str:
    ffmpeg_path = Path(FFMPEG_PATH)
    if ffmpeg_path.is_absolute():
        return str(ffmpeg_path)
    return str((PROJECT_ROOT / ffmpeg_path).resolve())


def _safe_path_part(value: str, fallback: str) -> str:
    cleaned = (value or "").strip().replace("\\", "_").replace("/", "_")
    return cleaned or fallback


def _load_session_context() -> Tuple[str, str]:
    if not SESSION_PATH.exists():
        raise RuntimeError("Session file missing. Login first.")

    payload = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    camera = payload.get("camera", {}) if isinstance(payload, dict) else {}
    raw_camera_id = str(camera.get("id") or "").strip()
    if not raw_camera_id:
        raise RuntimeError("Camera id missing in session.")

    camera_id = _safe_path_part(raw_camera_id, "camera")
    bucket_name = re.sub(r"[^a-z0-9_-]", "-", camera_id.lower()).strip("-")
    if not bucket_name:
        raise RuntimeError("Invalid camera id for bucket name.")

    return raw_camera_id, bucket_name


class ChunkRecorderService:
    def __init__(self):
        self.supabase = None
        self.bucket_name: Optional[str] = None
        self.camera_id: Optional[str] = None
        self.prefix = "stream-chunks"
        self.running = False

        self.writer = None
        self.frame_count = 0
        self.current_chunk_ts = 0
        self.current_avi_path: Optional[Path] = None
        self.current_size = (0, 0)

        self.task_queue: "Queue[Tuple[Path, int, str, str]]" = Queue()
        self.worker_thread = threading.Thread(target=self._encode_and_upload_worker, daemon=True)
        self.worker_thread.start()

        LOCAL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_supabase(self):
        if self.supabase is not None:
            return self.supabase
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY in .env")
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        return self.supabase

    def _ensure_bucket_and_prefix(self, bucket_name: str, prefix: str) -> None:
        storage = self._get_supabase().storage
        bucket_exists = False
        try:
            existing = storage.list_buckets()
            if isinstance(existing, list):
                bucket_exists = any(isinstance(b, dict) and b.get("name") == bucket_name for b in existing)
        except Exception as exc:
            print("Bucket discovery failed:", exc)

        if not bucket_exists:
            try:
                storage.create_bucket(
                    bucket_name,
                    name=bucket_name,
                    options={"public": True, "file_size_limit": 52428800},
                )
                print("Created bucket:", bucket_name)
            except Exception as exc:
                if "exists" not in str(exc).lower():
                    print("Bucket create failed:", exc)

        try:
            storage.from_(bucket_name).upload(
                f"{prefix}/_init.txt",
                BytesIO(b"stream-chunks initialized"),
                {"content-type": "application/octet-stream", "upsert": "true"},
            )
            print("Initialized folder marker:", f"{bucket_name}/{prefix}/_init.txt")
        except Exception as exc:
            print("Folder marker upload skipped:", exc)

    def _cleanup_remote_chunks(self, bucket_name: str, prefix: str, current_ts: int) -> None:
        cutoff_ts = current_ts - RETENTION_SECONDS
        storage = self._get_supabase().storage.from_(bucket_name)
        try:
            entries = storage.list(
                prefix,
                {"limit": 1000, "offset": 0, "sortBy": {"column": "name", "order": "asc"}},
            )
        except Exception:
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
        except Exception:
            pass

    def start(self) -> None:
        if self.running:
            return
        self.camera_id, self.bucket_name = _load_session_context()
        self._ensure_bucket_and_prefix(self.bucket_name, self.prefix)
        self.running = True
        print(f"Recorder started for camera {self.camera_id} -> bucket {self.bucket_name}/{self.prefix}")

    def stop(self) -> None:
        if not self.running:
            return
        self._finalize_current_chunk()
        self.running = False
        print("Recorder stopped")

    def add_frame(self, frame) -> None:
        if not self.running or self.bucket_name is None:
            return
        if frame is None:
            return

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return

        size_changed = self.current_size != (w, h)
        if self.writer is None or size_changed:
            self._finalize_current_chunk()
            self._start_new_chunk((w, h))

        self.writer.write(frame)
        self.frame_count += 1

        if self.frame_count >= FRAMES_PER_CHUNK:
            self._finalize_current_chunk()
            self._start_new_chunk((w, h))

    def _start_new_chunk(self, size: Tuple[int, int]) -> None:
        self.current_chunk_ts = int(time.time())
        self.current_avi_path = LOCAL_CHUNKS_DIR / f"chunk_{self.current_chunk_ts}.avi"
        self.current_size = size
        self.frame_count = 0
        self.writer = cv2.VideoWriter(
            str(self.current_avi_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            FPS,
            size,
        )
        print("Chunk started:", self.current_avi_path.as_posix())

    def _finalize_current_chunk(self) -> None:
        if self.writer is None or self.current_avi_path is None or self.bucket_name is None:
            return

        self.writer.release()
        self.writer = None

        if self.frame_count > 0:
            print("Chunk recorded:", self.current_avi_path.as_posix())
            self.task_queue.put((self.current_avi_path, self.current_chunk_ts, self.bucket_name, self.prefix))
        else:
            try:
                self.current_avi_path.unlink(missing_ok=True)
            except Exception:
                pass

        self.current_avi_path = None
        self.frame_count = 0

    def _encode_and_upload_worker(self) -> None:
        while True:
            avi_path, timestamp, bucket_name, prefix = self.task_queue.get()
            mp4_path = avi_path.with_suffix(".mp4")
            remote_path = f"{prefix}/chunk_{timestamp}.mp4"
            try:
                subprocess.run(
                    [
                        _resolve_ffmpeg_path(),
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
                    with mp4_path.open("rb") as fp:
                        self._get_supabase().storage.from_(bucket_name).upload(
                            remote_path,
                            fp,
                            {"content-type": "video/mp4"},
                        )
                    print("Uploaded:", f"{bucket_name}/{remote_path}")
                    self._cleanup_remote_chunks(bucket_name, prefix, timestamp)
                else:
                    print("Encode failed; mp4 missing for:", avi_path.as_posix())
            except Exception as exc:
                print("Chunk processing failed:", exc)
            finally:
                try:
                    avi_path.unlink(missing_ok=True)
                    mp4_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self.task_queue.task_done()
