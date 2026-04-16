import time
from collections import OrderedDict, deque
from config.settings import (
    IDENTITY_CACHE_MAX_USERS,
    IDENTITY_EMBEDDING_HISTORY_PER_USER,
    IDENTITY_EVENT_COOLDOWN_SECONDS,
)


class IdentityMemory:
    def __init__(self):
        self._users = OrderedDict()

    def _ensure_user(self, user_id, now):
        if user_id in self._users:
            self._users.move_to_end(user_id)
            self._users[user_id]["last_seen"] = now
            return self._users[user_id]

        if len(self._users) >= IDENTITY_CACHE_MAX_USERS:
            self._users.popitem(last=False)

        user = {
            "embeddings": deque(maxlen=IDENTITY_EMBEDDING_HISTORY_PER_USER),
            "last_seen": now,
            "last_logged": 0.0,
        }
        self._users[user_id] = user
        return user

    def add_embedding(self, user_id, embedding, now=None):
        now = now or time.time()
        user = self._ensure_user(user_id, now)
        user["embeddings"].append(embedding)
        return len(user["embeddings"])

    def should_log_identity(self, user_id, now=None):
        now = now or time.time()
        user = self._ensure_user(user_id, now)
        elapsed = now - user["last_logged"]
        return elapsed >= IDENTITY_EVENT_COOLDOWN_SECONDS

    def mark_logged(self, user_id, now=None):
        now = now or time.time()
        user = self._ensure_user(user_id, now)
        user["last_logged"] = now

