import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from config.settings import (
    DETECTION_CONFIRM_FRAMES,
    DETECTION_MATCH_IOU,
    DISPLAY_MIRROR_FEED,
    FACE_RECOGNITION_ENABLED,
    FACE_RECOGNITION_CONFIRM_FRAMES,
    FACE_RECOGNITION_FRAME_SKIP,
    FACE_RECOGNITION_MATCH_IOU,
    FACE_RECOGNITION_UNKNOWN_LABEL,
    YOLO_INFERENCE_FRAME_SKIP,
)
from core.camera import Camera
from core.detector import Detector
from core.face_engine import FaceEngine
from core.motion import MotionDetector


class AIWorker(QThread):
    frame_ready = pyqtSignal(object, int, object)
    raw_frame_ready = pyqtSignal(object)

    def __init__(self, toggles):
        super().__init__()
        self.running = True

        self.camera = Camera()
        self.detector = Detector()
        self.motion = MotionDetector()
        self.face_engine = FaceEngine()

        self.toggles = toggles
        self._previous_raw_detections = []
        self._confirmed_streaks = {}
        self._previous_raw_faces = []
        self._confirmed_face_streaks = {}

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

                if cls == "person" and self.toggles["intrusion"]():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if cls in ["car", "truck", "bus"] and self.toggles["vehicle"]():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                if cls in ["knife", "scissors"] and self.toggles["threat"]():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if self.toggles["motion"]():
                boxes = self.motion.detect(frame)
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

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

            self.raw_frame_ready.emit(raw_frame)
            self.frame_ready.emit(frame, person_count, faces)

        self.camera.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
