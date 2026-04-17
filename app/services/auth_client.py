import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from dotenv import load_dotenv


load_dotenv()


class AuthClient:
    def __init__(self):
        self.base_url = os.getenv("MONITOR_BACKEND_URL", "http://localhost:8000").rstrip("/")
        self.api_base = f"{self.base_url}/api/v1/auth"
        self.session_path = Path("data/auth/session.json")

    def _save_session(self, token: str, user: Dict):
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token": token, "user": user}
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
        self._save_session(token, user)
        return {"token": token, "user": user}

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
        self._save_session(token, user)
        return {"token": token, "user": user}


auth_client = AuthClient()
