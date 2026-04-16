import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config.settings import (
    DISPLAY_MIRROR_FEED,
    EMERGENCY_FINGER_NAMES,
    EMERGENCY_HAND_MAX_NUM,
    EMERGENCY_HAND_LANDMARKER_MODEL_PATH,
    EMERGENCY_MIN_DETECTION_CONFIDENCE,
    EMERGENCY_MIN_PRESENCE_CONFIDENCE,
    EMERGENCY_MIN_TRACKING_CONFIDENCE,
    EMERGENCY_PATTERN_PATH,
    EMERGENCY_PATTERN_STEPS,
    EMERGENCY_RESET_TIMEOUT_SECONDS,
)


class EmergencyPatternStore:
    def __init__(self, path=EMERGENCY_PATTERN_PATH):
        self.path = Path(path)

    def load_steps(self):
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            return []

        normalized = []
        for step in steps:
            if not isinstance(step, list) or len(step) != len(EMERGENCY_FINGER_NAMES):
                return []
            normalized.append(tuple(bool(v) for v in step))
        return normalized

    def save_steps(self, steps):
        payload = {
            "steps": [[bool(v) for v in step] for step in steps],
            "finger_order": list(EMERGENCY_FINGER_NAMES),
            "step_count": EMERGENCY_PATTERN_STEPS,
            "updated_at": int(time.time()),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class EmergencySequenceMatcher:
    def __init__(self, steps=None, timeout_s=EMERGENCY_RESET_TIMEOUT_SECONDS):
        self.steps = list(steps or [])
        self.timeout_s = timeout_s
        self.progress = 0
        self.last_match_ts = 0.0

    def set_steps(self, steps):
        normalized = [tuple(bool(v) for v in step) for step in (steps or [])]
        if normalized != self.steps:
            self.steps = normalized
            self.reset()

    def reset(self):
        self.progress = 0
        self.last_match_ts = 0.0

    def update(self, pattern, now=None):
        now = time.time() if now is None else now
        if not self.steps:
            self.reset()
            return {
                "progress": 0,
                "total": 0,
                "remaining_reset_s": 0.0,
                "captured": False,
                "triggered": False,
            }

        if self.progress > 0 and (now - self.last_match_ts) > self.timeout_s:
            self.progress = 0
            self.last_match_ts = 0.0

        captured = False
        triggered = False
        if pattern is not None and self.progress < len(self.steps):
            expected = self.steps[self.progress]
            if tuple(pattern) == tuple(expected):
                self.progress += 1
                self.last_match_ts = now
                captured = True
                if self.progress >= len(self.steps):
                    triggered = True
                    self.progress = 0
                    self.last_match_ts = 0.0

        if self.progress > 0 and self.last_match_ts > 0:
            remaining = max(0.0, self.timeout_s - (now - self.last_match_ts))
        else:
            remaining = self.timeout_s

        return {
            "progress": self.progress,
            "total": len(self.steps),
            "remaining_reset_s": float(remaining),
            "captured": captured,
            "triggered": triggered,
        }


class HandPatternDetector:
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    def __init__(self):
        base_options = mp_python.BaseOptions(
            model_asset_path=EMERGENCY_HAND_LANDMARKER_MODEL_PATH
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=EMERGENCY_HAND_MAX_NUM,
            min_hand_detection_confidence=EMERGENCY_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=EMERGENCY_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=EMERGENCY_MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _finger_pattern(self, hand_landmarks, handedness_label):
        lm = hand_landmarks

        # index, middle, ring, pinky: tip above pip => open.
        index_open = lm[8].y < lm[6].y
        middle_open = lm[12].y < lm[10].y
        ring_open = lm[16].y < lm[14].y
        pinky_open = lm[20].y < lm[18].y

        # thumb: handedness-aware x comparison.
        effective_handedness = handedness_label.lower()
        if DISPLAY_MIRROR_FEED:
            if effective_handedness == "right":
                effective_handedness = "left"
            elif effective_handedness == "left":
                effective_handedness = "right"

        if effective_handedness == "right":
            thumb_open = lm[4].x < lm[3].x
        else:
            thumb_open = lm[4].x > lm[3].x

        return (
            bool(thumb_open),
            bool(index_open),
            bool(middle_open),
            bool(ring_open),
            bool(pinky_open),
        )

    @staticmethod
    def _to_px(landmark, width, height):
        x = int(max(0, min(width - 1, landmark.x * width)))
        y = int(max(0, min(height - 1, landmark.y * height)))
        return x, y

    def _draw_landmarks(self, frame_bgr, hand_landmarks):
        h, w = frame_bgr.shape[:2]
        for a, b in self.HAND_CONNECTIONS:
            x1, y1 = self._to_px(hand_landmarks[a], w, h)
            x2, y2 = self._to_px(hand_landmarks[b], w, h)
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (80, 220, 255), 2, cv2.LINE_AA)

        for lm in hand_landmarks:
            x, y = self._to_px(lm, w, h)
            cv2.circle(frame_bgr, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def detect(self, frame_bgr, draw=True):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return None

        hand_landmarks = results.hand_landmarks[0]
        handedness_label = "Right"
        if results.handedness and results.handedness[0]:
            handedness_label = results.handedness[0][0].category_name

        if draw:
            self._draw_landmarks(frame_bgr, hand_landmarks)

        return self._finger_pattern(hand_landmarks, handedness_label)
