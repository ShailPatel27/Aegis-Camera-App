import json
from pathlib import Path

import cv2
import numpy as np

from config.settings import (
    FACE_DB_PATH,
    FACE_DETECT_MIN_NEIGHBORS,
    FACE_DETECT_SCALE_FACTOR,
    FACE_EMBED_SIZE,
    FACE_REGISTRATION_ALLOW_OVERWRITE,
    FACE_REGISTRATION_CASE_SENSITIVE,
    FACE_REGISTRATION_DUPLICATE_THRESHOLD,
    FACE_MIN_SIZE,
    FACE_RECOGNITION_THRESHOLD,
)


class FaceEngine:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.db_path = Path(FACE_DB_PATH)
        self._last_registry_mtime = None
        self.registry = self._load_registry()

    def _load_registry(self):
        current_mtime = None
        if self.db_path.exists():
            try:
                current_mtime = self.db_path.stat().st_mtime
            except Exception:
                current_mtime = None

        if not self.db_path.exists():
            self._last_registry_mtime = None
            return {}
        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._last_registry_mtime = current_mtime
                return data
        except Exception:
            pass
        self._last_registry_mtime = current_mtime
        return {}

    def _save_registry(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")
        try:
            self._last_registry_mtime = self.db_path.stat().st_mtime
        except Exception:
            self._last_registry_mtime = None

    def reload_if_changed(self):
        # Runtime sync: keep face identities current without app restart.
        if not self.db_path.exists():
            if self.registry:
                self.registry = {}
            self._last_registry_mtime = None
            return

        try:
            current_mtime = self.db_path.stat().st_mtime
        except Exception:
            return

        if self._last_registry_mtime is None or current_mtime != self._last_registry_mtime:
            self.registry = self._load_registry()

    def list_users(self):
        self.reload_if_changed()
        return sorted(self.registry.keys())

    def delete_user(self, name):
        if name not in self.registry:
            return False
        del self.registry[name]
        self._save_registry()
        return True

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECT_SCALE_FACTOR,
            minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
            minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE),
        )
        return faces

    @staticmethod
    def box_iou_xywh(box_a, box_b):
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0, aw) * max(0, ah)
        area_b = max(0, bw) * max(0, bh)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def _cosine_similarity(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _sanitize_name(name):
        return " ".join(name.strip().split())

    @staticmethod
    def _name_key(name):
        return name if FACE_REGISTRATION_CASE_SENSITIVE else name.lower()

    def identify_embedding(self, embedding, threshold=None):
        self.reload_if_changed()
        if embedding is None:
            return None, 0.0
        emb_vec = np.array(embedding, dtype=np.float32)
        best_name = None
        best_score = 0.0
        for name, data in self.registry.items():
            centroid = np.array(data["centroid"], dtype=np.float32)
            score = self._cosine_similarity(emb_vec, centroid)
            if score > best_score:
                best_score = score
                best_name = name

        threshold = FACE_RECOGNITION_THRESHOLD if threshold is None else threshold
        if best_name is None or best_score < threshold:
            return None, float(best_score)
        return best_name, float(best_score)

    def build_embedding(self, frame, face_box):
        x, y, w, h = face_box
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            return None

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (FACE_EMBED_SIZE, FACE_EMBED_SIZE))
        vec = resized.astype(np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return (vec / norm).tolist()

    def register_user(self, name, embeddings, role="user", image_path=None, image_url=None):
        clean_name = self._sanitize_name(name)
        if not clean_name:
            raise ValueError("Name cannot be empty.")
        if not embeddings:
            raise ValueError("At least one face sample is required.")

        existing_name = None
        clean_key = self._name_key(clean_name)
        for candidate in self.registry.keys():
            if self._name_key(candidate) == clean_key:
                existing_name = candidate
                break

        if existing_name is not None and not FACE_REGISTRATION_ALLOW_OVERWRITE:
            raise ValueError(f"User already exists: {existing_name}")

        check_vec = np.mean(np.array(embeddings, dtype=np.float32), axis=0)
        check_norm = np.linalg.norm(check_vec)
        if check_norm == 0:
            raise ValueError("Invalid samples.")
        check_vec = (check_vec / check_norm).tolist()
        dup_name, dup_score = self.identify_embedding(
            check_vec,
            threshold=FACE_REGISTRATION_DUPLICATE_THRESHOLD,
        )
        if dup_name is not None and existing_name is None:
            raise ValueError(f"Face already registered as {dup_name} ({dup_score:.2f})")

        vectors = np.array(embeddings, dtype=np.float32)
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            raise ValueError("Invalid samples.")
        centroid = (centroid / norm).tolist()

        key_name = existing_name if existing_name is not None else clean_name
        self.registry[key_name] = {
            "centroid": centroid,
            "samples": len(embeddings),
            "role": str(role or "user"),
            "image_path": image_path,
            "image_url": image_url,
        }
        self._save_registry()
        return key_name

    def recognize(self, frame):
        self.reload_if_changed()
        results = []
        faces = self.detect_faces(frame)
        for (x, y, w, h) in faces:
            emb = self.build_embedding(frame, (x, y, w, h))
            if emb is None:
                continue

            best_name, best_score = self.identify_embedding(emb, FACE_RECOGNITION_THRESHOLD)
            is_match = best_name is not None
            role = "user"
            image_url = None
            if is_match:
                profile = self.registry.get(best_name, {}) if isinstance(self.registry.get(best_name), dict) else {}
                role = str(profile.get("role") or "user")
                image_url = profile.get("image_url")
            results.append(
                {
                    "box": (int(x), int(y), int(x + w), int(y + h)),
                    "name": best_name if is_match else None,
                    "score": float(best_score),
                    "matched": bool(is_match),
                    "role": role if is_match else None,
                    "image_url": image_url if is_match else None,
                }
            )
        return results
