import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from supabase import create_client

from config.settings import (
    CAMERA_INDEX,
    DEFAULT_CROWD,
    DEFAULT_EMERGENCY,
    DEFAULT_FACE_RECOGNITION,
    DEFAULT_INTRUSION,
    DEFAULT_LOITER,
    DEFAULT_MOTION,
    DEFAULT_THREAT,
    DEFAULT_VEHICLE,
)


load_dotenv()


class AuthClient:
    def __init__(self):
        self.base_url = os.getenv("MONITOR_BACKEND_URL", "http://localhost:8000").rstrip("/")
        self.api_base = f"{self.base_url}/api/v1/auth"
        self.session_path = Path("data/auth/session.json")
        self.supabase = None

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if supabase_url and supabase_key:
            try:
                self.supabase = create_client(supabase_url, supabase_key)
            except Exception:
                self.supabase = None

    @staticmethod
    def _ai_toggle_defaults() -> Dict[str, bool]:
        return {
            "intrusion": bool(DEFAULT_INTRUSION),
            "crowd": bool(DEFAULT_CROWD),
            "vehicle": bool(DEFAULT_VEHICLE),
            "threat": bool(DEFAULT_THREAT),
            "motion": bool(DEFAULT_MOTION),
            "loiter": bool(DEFAULT_LOITER),
            "emergency": bool(DEFAULT_EMERGENCY),
            "face_recognition": bool(DEFAULT_FACE_RECOGNITION),
            "screenshot": False,
        }

    def _normalize_ai_toggles(self, raw: Optional[Dict]) -> Dict[str, bool]:
        normalized = self._ai_toggle_defaults()
        if isinstance(raw, dict):
            for key in normalized.keys():
                if key in raw:
                    normalized[key] = bool(raw[key])
        return normalized

    def _merge_camera_with_defaults(self, camera: Dict) -> Dict:
        merged = dict(camera or {})
        config = merged.get("config")
        if not isinstance(config, dict):
            config = {}
        config = dict(config)
        config["ai_toggles"] = self._normalize_ai_toggles(config.get("ai_toggles"))
        merged["config"] = config
        return merged

    def _save_session(self, token: str, user: Dict, camera: Optional[Dict] = None):
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token": token, "user": user, "camera": camera or {}}
        self.session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear_session(self):
        if self.session_path.exists():
            self.session_path.unlink()

    def load_session(self) -> Optional[Dict]:
        if not self.session_path.exists():
            return None
        try:
            data = json.loads(self.session_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("token") and data.get("user"):
                return data
        except Exception:
            pass
        return None

    def _extract_auth_payload(self, payload: Dict) -> Tuple[str, Dict]:
        if payload.get("success") and isinstance(payload.get("data"), dict):
            data = payload["data"]
        else:
            data = payload
        token = data.get("access_token")
        user = data.get("user")
        if not token or not isinstance(user, dict):
            raise ValueError(payload.get("message") or payload.get("detail") or "Invalid auth response")
        return token, user

    def _auth_headers(self, token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def _get_existing_camera(self, token: str) -> Optional[Dict]:
        get_resp = requests.get(
            f"{self.base_url}/api/cameras",
            headers=self._auth_headers(token),
            timeout=20,
        )
        if get_resp.status_code in (401, 403):
            raise PermissionError("Session expired")

        if get_resp.ok:
            cameras = get_resp.json()
            if isinstance(cameras, list) and len(cameras) > 0:
                preferred = next(
                    (c for c in cameras if int(c.get("selected_camera", -1)) == int(CAMERA_INDEX)),
                    cameras[0],
                )
                return self._merge_camera_with_defaults(preferred)
        return None

    def get_current_user(self, token: str) -> Dict:
        response = requests.get(
            f"{self.api_base}/me",
            headers=self._auth_headers(token),
            timeout=20,
        )
        if response.status_code in (401, 403):
            raise PermissionError("Session expired")
        if not response.ok:
            raise ValueError("Failed to validate user session")
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("id"):
            raise ValueError("Invalid user payload from backend")
        return payload

    def refresh_session(self, session: Dict) -> Dict:
        if not isinstance(session, dict):
            raise ValueError("Missing session")
        token = session.get("token")
        if not token:
            raise ValueError("Missing session token")

        user = self.get_current_user(token)
        camera = self._get_existing_camera(token) or {}
        self._save_session(token, user, camera)
        return {"token": token, "user": user, "camera": camera}

    def _create_camera(self, token: str, user: Dict, camera_name: str) -> Dict:
        name = (camera_name or "").strip()
        if not name:
            raise ValueError("Camera name cannot be empty.")

        register_resp = requests.post(
            f"{self.base_url}/api/cameras/register",
            headers={**self._auth_headers(token), "Content-Type": "application/json"},
            json={
                "name": name,
                "selected_camera": int(CAMERA_INDEX),
                "type": "webcam",
                "location": None,
                "config": {"ai_toggles": self._ai_toggle_defaults()},
            },
            timeout=20,
        )
        if not register_resp.ok:
            try:
                payload = register_resp.json()
                message = payload.get("detail") or payload.get("message") or "Failed to create camera"
            except Exception:
                message = "Failed to create camera"
            raise ValueError(message)

        payload = register_resp.json()
        camera_id = payload.get("camera_id")
        resolved_name = name
        camera_data = {}
        if isinstance(payload.get("camera"), list) and payload["camera"]:
            camera_data = payload["camera"][0]
            resolved_name = camera_data.get("name", resolved_name)
        created = {
            "id": camera_id,
            "name": resolved_name,
            "selected_camera": int(CAMERA_INDEX),
            "config": {"ai_toggles": self._ai_toggle_defaults()},
        }
        created.update(camera_data)
        return self._merge_camera_with_defaults(created)

    def _finalize_session(self, token: str, user: Dict, camera: Dict) -> Dict:
        user["camera_id"] = camera.get("id")
        self._save_session(token, user, camera)
        return {"token": token, "user": user, "camera": camera}

    def get_ai_toggles(self, session: Dict) -> Dict[str, bool]:
        camera = session.get("camera") if isinstance(session, dict) else None
        if not isinstance(camera, dict):
            return self._ai_toggle_defaults()
        config = camera.get("config")
        if not isinstance(config, dict):
            return self._ai_toggle_defaults()
        return self._normalize_ai_toggles(config.get("ai_toggles"))

    def refresh_camera(self, session: Dict) -> Optional[Dict]:
        if not isinstance(session, dict):
            return None
        token = session.get("token")
        if not token:
            return None
        latest = self._get_existing_camera(token)
        if not latest:
            return None
        self._save_session(token, session.get("user", {}), latest)
        return latest

    def update_ai_toggle(self, session: Dict, toggle_key: str, enabled: bool) -> Dict:
        valid_keys = self._ai_toggle_defaults().keys()
        if toggle_key not in valid_keys:
            raise ValueError(f"Unknown AI toggle key: {toggle_key}")

        if not isinstance(session, dict):
            raise ValueError("Missing session")

        token = session.get("token")
        user = dict(session.get("user", {}))
        camera = dict(session.get("camera", {}))
        if not token or not camera.get("id"):
            raise ValueError("Missing authenticated camera session")

        camera = self._merge_camera_with_defaults(camera)
        config = dict(camera.get("config", {}))
        ai_toggles = self._normalize_ai_toggles(config.get("ai_toggles"))
        ai_toggles[toggle_key] = bool(enabled)
        config["ai_toggles"] = ai_toggles
        camera["config"] = config

        camera_id = camera.get("id")
        user_id = user.get("id")
        updated_camera = None
        if self.supabase and camera_id and user_id:
            payload = {
                "config": config,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            response = (
                self.supabase.table("cameras")
                .update(payload)
                .eq("id", camera_id)
                .eq("user_id", user_id)
                .execute()
            )
            if isinstance(response.data, list) and response.data:
                updated_camera = response.data[0]

        if isinstance(updated_camera, dict):
            camera = self._merge_camera_with_defaults(updated_camera)

        self._save_session(token, user, camera)
        return camera

    def register(self, name: str, email: str, password: str) -> Dict:
        response = requests.post(
            f"{self.api_base}/register",
            json={"name": name, "email": email, "password": password},
            timeout=20,
        )
        payload = response.json()
        if not response.ok or not payload.get("success", True):
            raise ValueError(payload.get("message") or payload.get("detail") or "Registration failed")
        token, user = self._extract_auth_payload(payload)
        camera = self._get_existing_camera(token)
        if camera:
            return self._finalize_session(token, user, camera)
        return {"token": token, "user": user, "camera": None, "needs_camera_name": True}

    def login(self, email: str, password: str) -> Dict:
        response = requests.post(
            f"{self.api_base}/login",
            json={"email": email, "password": password},
            timeout=20,
        )
        payload = response.json()
        if not response.ok or not payload.get("success", True):
            raise ValueError(payload.get("message") or payload.get("detail") or "Login failed")
        token, user = self._extract_auth_payload(payload)
        camera = self._get_existing_camera(token)
        if camera:
            return self._finalize_session(token, user, camera)
        return {"token": token, "user": user, "camera": None, "needs_camera_name": True}

    def complete_camera_setup(self, token: str, user: Dict, camera_name: str) -> Dict:
        existing = self._get_existing_camera(token)
        camera = existing or self._create_camera(token, user, camera_name)
        return self._finalize_session(token, user, camera)


auth_client = AuthClient()
