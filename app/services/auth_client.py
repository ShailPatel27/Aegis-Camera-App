import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse

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

    @staticmethod
    def _normalize_embedding(values, target_dim: int = 9216):
        if not isinstance(values, (list, tuple)):
            return [0.0] * target_dim
        casted = []
        for v in values:
            try:
                casted.append(float(v))
            except Exception:
                casted.append(0.0)
        if len(casted) < target_dim:
            casted.extend([0.0] * (target_dim - len(casted)))
        if len(casted) > target_dim:
            casted = casted[:target_dim]
        return casted

    def _auth_headers(self, token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def _candidate_base_urls(self):
        urls = [self.base_url]
        try:
            parsed = urlparse(self.base_url)
            if parsed.port == 8000:
                alt = parsed._replace(netloc=f"{parsed.hostname}:8001")
                urls.append(urlunparse(alt))
        except Exception:
            pass
        # Keep order and uniqueness.
        out = []
        seen = set()
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _request_with_port_fallback(self, method: str, path: str, **kwargs):
        last_exc = None
        for base in self._candidate_base_urls():
            url = f"{base}{path}"
            try:
                return requests.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise ValueError("Request failed")

    def _get_existing_camera(self, token: str) -> Optional[Dict]:
        for attempt in range(2):
            get_resp = self._request_with_port_fallback(
                "GET",
                "/api/cameras",
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

            # Occasionally backend routes may not be ready yet on first request.
            if get_resp.status_code == 404 and attempt == 0:
                time.sleep(0.35)
                continue
            break
        return None

    def get_current_user(self, token: str) -> Dict:
        response = self._request_with_port_fallback(
            "GET",
            "/api/v1/auth/me",
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

        register_resp = self._request_with_port_fallback(
            "POST",
            "/api/cameras/register",
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
        updated_camera = None

        # Primary write path: backend API, so monitor and camera stay in sync.
        response = self._request_with_port_fallback(
            "PATCH",
            f"/api/cameras/{camera_id}/config",
            headers={**self._auth_headers(token), "Content-Type": "application/json"},
            json={"config": config},
            timeout=20,
        )
        if response.status_code in (401, 403):
            raise PermissionError("Session expired")
        if response.ok:
            payload = response.json()
            candidate = payload.get("camera") if isinstance(payload, dict) else None
            if isinstance(candidate, dict):
                updated_camera = candidate
        elif self.supabase and camera_id and user.get("id"):
            # Fallback for transient backend issues.
            user_id = user.get("id")
            payload = {
                "config": config,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            supa_response = (
                self.supabase.table("cameras")
                .update(payload)
                .eq("id", camera_id)
                .eq("user_id", user_id)
                .execute()
            )
            if isinstance(supa_response.data, list) and supa_response.data:
                updated_camera = supa_response.data[0]
            else:
                raise ValueError("Failed to update AI toggles")
        else:
            raise ValueError("Failed to update AI toggles")

        if isinstance(updated_camera, dict):
            camera = self._merge_camera_with_defaults(updated_camera)

        self._save_session(token, user, camera)
        return camera

    def set_camera_stream_state(self, session: Dict, enabled: bool) -> Dict:
        if not isinstance(session, dict):
            raise ValueError("Missing session")

        token = session.get("token")
        user = dict(session.get("user", {}))
        camera = dict(session.get("camera", {}))
        camera_id = camera.get("id")
        if not token or not camera_id:
            raise ValueError("Missing authenticated camera session")

        response = self._request_with_port_fallback(
            "PATCH",
            f"/api/cameras/{camera_id}/stream",
            headers={**self._auth_headers(token), "Content-Type": "application/json"},
            json={"enabled": bool(enabled)},
            timeout=20,
        )
        if response.status_code in (401, 403):
            raise PermissionError("Session expired")
        if not response.ok:
            raise ValueError("Failed to update camera stream state")

        payload = response.json()
        updated = payload.get("camera") if isinstance(payload, dict) else None
        if isinstance(updated, dict):
            camera = self._merge_camera_with_defaults(updated)
        else:
            camera["status"] = "online" if enabled else "offline"
            camera = self._merge_camera_with_defaults(camera)

        self._save_session(token, user, camera)
        return camera

    def set_camera_paused(self, session: Dict, paused: bool) -> Dict:
        if not isinstance(session, dict):
            raise ValueError("Missing session")

        token = session.get("token")
        user = dict(session.get("user", {}))
        camera = dict(session.get("camera", {}))
        camera_id = camera.get("id")
        if not token or not camera_id:
            raise ValueError("Missing authenticated camera session")

        current_config = camera.get("config") if isinstance(camera.get("config"), dict) else {}
        config = dict(current_config)
        config["feed_paused"] = bool(paused)

        response = self._request_with_port_fallback(
            "PATCH",
            f"/api/cameras/{camera_id}/config",
            headers={**self._auth_headers(token), "Content-Type": "application/json"},
            json={"config": config},
            timeout=20,
        )
        if response.status_code in (401, 403):
            raise PermissionError("Session expired")
        if not response.ok:
            raise ValueError("Failed to update camera pause state")

        payload = response.json()
        updated = payload.get("camera") if isinstance(payload, dict) else None
        if isinstance(updated, dict):
            camera = self._merge_camera_with_defaults(updated)
        else:
            camera["config"] = config
            camera = self._merge_camera_with_defaults(camera)

        self._save_session(token, user, camera)
        return camera

    def register(self, name: str, email: str, password: str) -> Dict:
        response = self._request_with_port_fallback(
            "POST",
            "/api/v1/auth/register",
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
        response = self._request_with_port_fallback(
            "POST",
            "/api/v1/auth/login",
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

    def create_alert(
        self,
        session: Dict,
        alert_type: str,
        message: str,
        confidence: Optional[float] = None,
        image_url: Optional[str] = None,
        metadata: Optional[Dict] = None,
        face_name: Optional[str] = None,
    ) -> Dict:
        if not isinstance(session, dict):
            raise ValueError("Missing session")

        token = session.get("token")
        camera = session.get("camera") if isinstance(session.get("camera"), dict) else {}
        camera_id = camera.get("id")
        if not token or not camera_id:
            raise ValueError("Missing authenticated camera session")

        payload = {
            "camera_id": camera_id,
            "alert_type": alert_type,
            "message": message,
            "confidence": confidence,
            "image_url": image_url,
            "metadata": metadata or {},
            "face_name": face_name,
        }
        response = self._request_with_port_fallback(
            "POST",
            "/api/v1/monitor/alerts",
            headers={**self._auth_headers(token), "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        if response.status_code in (401, 403):
            raise PermissionError("Session expired")
        if not response.ok:
            detail = None
            try:
                err = response.json()
                detail = err.get("detail") or err.get("message")
            except Exception:
                detail = response.text.strip() or None
            raise ValueError(detail or f"Failed to create alert (HTTP {response.status_code})")
        return response.json()

    def sync_face_profile(self, session: Dict, name: str, embedding, role: str = "user", image_url: Optional[str] = None) -> Dict:
        if not isinstance(session, dict):
            raise ValueError("Missing session")
        user = session.get("user") if isinstance(session.get("user"), dict) else {}
        user_id = user.get("id")
        if not user_id:
            raise ValueError("Missing user id")

        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("Face name is required")

        if embedding is None:
            raise ValueError("Missing face embedding")
        normalized_embedding = self._normalize_embedding(embedding, 9216)

        if not self.supabase:
            raise ValueError("Supabase client unavailable for face sync")

        existing = (
            self.supabase.table("faces")
            .select("id")
            .eq("user_id", user_id)
            .eq("name", clean_name)
            .limit(1)
            .execute()
        )
        existing_rows = existing.data if isinstance(existing.data, list) else []

        payload = {
            "user_id": user_id,
            "name": clean_name,
            "role": role or "user",
            "embedding": normalized_embedding,
            "image_url": image_url,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if existing_rows:
            face_id = existing_rows[0].get("id")
            response = self.supabase.table("faces").update(payload).eq("id", face_id).execute()
        else:
            payload["created_at"] = datetime.now(timezone.utc).isoformat()
            response = self.supabase.table("faces").insert(payload).execute()

        rows = response.data if isinstance(response.data, list) else []
        return rows[0] if rows else {}


auth_client = AuthClient()
