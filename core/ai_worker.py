import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal

from config.settings import (
    CROWD_THRESHOLD,
    DETECTION_CONFIRM_FRAMES,
    DETECTION_MATCH_IOU,
    DISPLAY_MIRROR_FEED,
    FACE_RECOGNITION_ENABLED,
    FACE_RECOGNITION_CONFIRM_FRAMES,
    FACE_RECOGNITION_FRAME_SKIP,
    FACE_RECOGNITION_MATCH_IOU,
    FACE_RECOGNITION_UNKNOWN_LABEL,
    EMERGENCY_FRAME_SKIP,
    FRAME_INTERVAL_MS,
    IDLE_FRAME_INTERVAL_MS,
    LOITER_DWELL_SECONDS,
    LOITER_MATCH_IOU,
    LOITER_TRACK_STALE_SECONDS,
    THREAT_CLASS_NAMES,
    THREAT_MIN_CONFIDENCE,
    YOLO_INFERENCE_FRAME_SKIP,
)
from core.camera import Camera
from core.detector import Detector
from core.emergency import (
    EmergencyPatternStore,
    EmergencySequenceMatcher,
    HandPatternDetector,
)
from core.face_engine import FaceEngine
from core.motion import MotionDetector


class AIWorker(QThread):
    frame_ready = pyqtSignal(object, int, object, object)
    raw_frame_ready = pyqtSignal(object)

    def __init__(self, toggles, camera_index=None):
        super().__init__()
        self.running = True
        self._emergency_flash_until = 0.0

        self.camera = Camera(camera_index=camera_index)
        self.detector = Detector()
        self.motion = MotionDetector()
        self.face_engine = FaceEngine()
        self.hand_detector = HandPatternDetector()
        self.emergency_store = EmergencyPatternStore()
        self.emergency_matcher = EmergencySequenceMatcher(self.emergency_store.load_steps())
        self._last_emergency_status = {
            "progress": 0,
            "total": 0,
            "remaining_reset_s": 0.0,
            "captured": False,
            "triggered": False,
        }

        self.toggles = toggles
        self._previous_raw_detections = []
        self._confirmed_streaks = {}
        self._previous_raw_faces = []
        self._confirmed_face_streaks = {}
        self._loiter_tracks = {}
        self._next_loiter_track_id = 1

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0

        return inter_area / union

    def _build_confirmed_detections(self, detections):
        confirmed = []
        updated_streaks = {}

        for det in detections:
            class_name = det["class_name"]
            box = tuple(det["box"])

            best_key = None
            best_iou = 0.0
            for prev_idx, prev_det in enumerate(self._previous_raw_detections):
                if prev_det["class_name"] != class_name:
                    continue
                iou = self._iou(box, tuple(prev_det["box"]))
                if iou > best_iou:
                    best_iou = iou
                    best_key = f"{class_name}:{prev_idx}"

            if best_key and best_iou >= DETECTION_MATCH_IOU:
                streak = self._confirmed_streaks.get(best_key, 0) + 1
            else:
                streak = 1

            curr_key = f"{class_name}:{len(updated_streaks)}"
            updated_streaks[curr_key] = streak

            if streak >= DETECTION_CONFIRM_FRAMES:
                confirmed.append(det)

        self._previous_raw_detections = detections
        self._confirmed_streaks = updated_streaks
        return confirmed

    def _build_confirmed_faces(self, faces):
        confirmed = []
        updated_streaks = {}

        for face in faces:
            x1, y1, x2, y2 = face["box"]
            box = (x1, y1, x2, y2)

            best_key = None
            best_iou = 0.0
            for prev_idx, prev_face in enumerate(self._previous_raw_faces):
                px1, py1, px2, py2 = prev_face["box"]
                iou = self._iou(box, (px1, py1, px2, py2))
                if iou > best_iou:
                    best_iou = iou
                    best_key = f"face:{prev_idx}"

            if best_key and best_iou >= FACE_RECOGNITION_MATCH_IOU:
                streak = self._confirmed_face_streaks.get(best_key, 0) + 1
            else:
                streak = 1

            curr_key = f"face:{len(updated_streaks)}"
            updated_streaks[curr_key] = streak

            if streak >= FACE_RECOGNITION_CONFIRM_FRAMES:
                confirmed.append(face)

        self._previous_raw_faces = faces
        self._confirmed_face_streaks = updated_streaks
        return confirmed

    def run(self):
        frame_index = 0
        last_detections = []
        last_person_count = 0
        last_faces = []
        frame_interval_s = max(0.0, FRAME_INTERVAL_MS / 1000.0)
        next_emit_at = time.perf_counter()

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            if DISPLAY_MIRROR_FEED:
                frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()

            frame_index += 1

            person_count = last_person_count
            detections = last_detections
            faces = last_faces

            run_yolo = any(
                [
                    self.toggles["intrusion"](),
                    self.toggles["crowd"](),
                    self.toggles["vehicle"](),
                    self.toggles["threat"](),
                    self.toggles["loiter"](),
                ]
            )

            should_run_yolo = (
                YOLO_INFERENCE_FRAME_SKIP <= 1
                or frame_index % YOLO_INFERENCE_FRAME_SKIP == 0
            )
            if run_yolo and should_run_yolo:
                raw_detections = self.detector.detect(frame)
                detections = self._build_confirmed_detections(raw_detections)
                last_detections = detections

                person_count = 0
                for det in detections:
                    if det["class_name"] == "person":
                        person_count += 1
                last_person_count = person_count
            elif not run_yolo:
                self._previous_raw_detections = []
                self._confirmed_streaks = {}
                self._loiter_tracks = {}

            run_face = FACE_RECOGNITION_ENABLED and (
                self.toggles.get("face_recognition") is None
                or self.toggles["face_recognition"]()
            )
            if run_face:
                should_run_face = (
                    FACE_RECOGNITION_FRAME_SKIP <= 1
                    or frame_index % FACE_RECOGNITION_FRAME_SKIP == 0
                )
                if should_run_face:
                    raw_faces = self.face_engine.recognize(frame)
                    faces = self._build_confirmed_faces(raw_faces)
                    last_faces = faces
            else:
                self._previous_raw_faces = []
                self._confirmed_face_streaks = {}
                last_faces = []
                faces = []

            for det in detections:
                x1, y1, x2, y2 = map(int, det["box"])
                cls = det["class_name"]
                conf = float(det.get("confidence", 0.0))

                if cls == "person" and self.toggles["intrusion"]():
                    # Translucent border for person boxes.
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

                if cls in ["car", "truck", "bus"] and self.toggles["vehicle"]():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(
                        frame,
                        f"{cls} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if cls in THREAT_CLASS_NAMES and self.toggles["threat"]() and conf >= THREAT_MIN_CONFIDENCE:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"{cls} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            motion_count = 0
            if self.toggles["motion"]():
                boxes = self.motion.detect(frame)
                motion_count = len(boxes)
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if self.toggles["crowd"]() and person_count >= CROWD_THRESHOLD:
                cv2.putText(
                    frame,
                    f"CROWD ALERT ({person_count})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 140, 255),
                    2,
                    cv2.LINE_AA,
                )

            loiter_count = 0
            if self.toggles["loiter"]():
                now = time.time()
                person_boxes = [
                    tuple(map(int, det["box"]))
                    for det in detections
                    if det["class_name"] == "person"
                ]
                loitering_boxes = self._update_loiter_tracks(person_boxes, now)
                loiter_count = len(loitering_boxes)

                for box in loitering_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 105, 255), 2)
                    cv2.putText(
                        frame,
                        "LOITERING",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (180, 105, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                self._loiter_tracks = {}

            for face in faces:
                x1, y1, x2, y2 = face["box"]
                matched = face["matched"]
                label = face["name"] if matched else FACE_RECOGNITION_UNKNOWN_LABEL
                score = face["score"]
                color = (52, 211, 153) if matched else (148, 163, 184)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {score:.2f}" if matched else label
                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            emergency_progress = 0
            emergency_total = 0
            emergency_remaining = 0.0
            emergency_captured = False
            emergency_triggered = False
            if self.toggles.get("emergency") and self.toggles["emergency"]():
                # Reload in case pattern changed in Emergency page.
                self.emergency_matcher.set_steps(self.emergency_store.load_steps())
                now = time.time()
                status = self.emergency_matcher.update(None, now=now)
                if EMERGENCY_FRAME_SKIP <= 1 or frame_index % EMERGENCY_FRAME_SKIP == 0:
                    # Keep emergency hand landmarks exclusive to Emergency tab preview.
                    hand_pattern = self.hand_detector.detect(frame, draw=False)
                    now = time.time()
                    status = self.emergency_matcher.update(hand_pattern, now=now)

                self._last_emergency_status = status
                emergency_progress = status["progress"]
                emergency_total = status["total"]
                emergency_remaining = status["remaining_reset_s"]
                emergency_captured = status["captured"]
                emergency_triggered = status["triggered"]
                if emergency_triggered:
                    self._emergency_flash_until = now + 1.0

            if time.time() < self._emergency_flash_until:
                cv2.putText(
                    frame,
                    "EMERGENCY!",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            self.raw_frame_ready.emit(raw_frame)
            vehicle_count = sum(1 for d in detections if d["class_name"] in ["car", "truck", "bus"])
            threat_count = sum(
                1
                for d in detections
                if d["class_name"] in THREAT_CLASS_NAMES and float(d.get("confidence", 0.0)) >= THREAT_MIN_CONFIDENCE
            )
            matched_faces = [f["name"] for f in faces if f.get("matched") and f.get("name")]
            activity = {
                "person_count": person_count,
                "crowd_triggered": person_count >= CROWD_THRESHOLD,
                "vehicle_count": vehicle_count,
                "threat_count": threat_count,
                "motion_count": motion_count,
                "loiter_count": loiter_count,
                "matched_faces": matched_faces,
                "emergency_progress": emergency_progress,
                "emergency_total": emergency_total,
                "emergency_remaining_reset_s": emergency_remaining,
                "emergency_captured": emergency_captured,
                "emergency_triggered": emergency_triggered,
            }
            self.frame_ready.emit(frame, person_count, faces, activity)

            # Dynamic pacing: use idle frame interval when no notable activity is present.
            has_activity = any(
                [
                    person_count > 0,
                    vehicle_count > 0,
                    threat_count > 0,
                    loiter_count > 0,
                    len(matched_faces) > 0,
                    emergency_captured,
                    emergency_triggered,
                ]
            )
            interval_ms = FRAME_INTERVAL_MS if has_activity else IDLE_FRAME_INTERVAL_MS
            frame_interval_s = max(0.0, interval_ms / 1000.0)

            if frame_interval_s > 0:
                next_emit_at += frame_interval_s
                sleep_for = next_emit_at - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_emit_at = time.perf_counter()

        self.camera.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def _update_loiter_tracks(self, person_boxes, now):
        matched_track_ids = set()
        loitering_boxes = []

        for box in person_boxes:
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self._loiter_tracks.items():
                iou = self._iou(box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= LOITER_MATCH_IOU:
                track = self._loiter_tracks[best_track_id]
                track["box"] = box
                track["last_seen"] = now
                matched_track_ids.add(best_track_id)
            else:
                track_id = self._next_loiter_track_id
                self._next_loiter_track_id += 1
                self._loiter_tracks[track_id] = {
                    "box": box,
                    "first_seen": now,
                    "last_seen": now,
                }
                matched_track_ids.add(track_id)

        stale_ids = []
        for track_id, track in self._loiter_tracks.items():
            if now - track["last_seen"] > LOITER_TRACK_STALE_SECONDS:
                stale_ids.append(track_id)
                continue
            dwell = now - track["first_seen"]
            if track_id in matched_track_ids and dwell >= LOITER_DWELL_SECONDS:
                loitering_boxes.append(track["box"])

        for track_id in stale_ids:
            self._loiter_tracks.pop(track_id, None)

        return loitering_boxes
