import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from core.camera import Camera
from core.detector import Detector
from core.motion import MotionDetector
from config.settings import (
    DETECTION_CONFIRM_FRAMES,
    DETECTION_MATCH_IOU,
    YOLO_INFERENCE_FRAME_SKIP,
)


class AIWorker(QThread):
    frame_ready = pyqtSignal(object, int)

    def __init__(self, toggles):
        super().__init__()
        self.running = True

        self.camera = Camera()
        self.detector = Detector()
        self.motion = MotionDetector()

        self.toggles = toggles
        self._previous_raw_detections = []
        self._confirmed_streaks = {}

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

    def run(self):
        frame_index = 0
        last_detections = []
        last_person_count = 0

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            frame_index += 1

            person_count = last_person_count
            detections = last_detections

            # ===== YOLO (every Nth frame) =====
            run_yolo = any([
                self.toggles["intrusion"](),
                self.toggles["crowd"](),
                self.toggles["vehicle"](),
                self.toggles["threat"]()
            ])

            should_run_yolo = YOLO_INFERENCE_FRAME_SKIP <= 1 or frame_index % YOLO_INFERENCE_FRAME_SKIP == 0
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

            # ===== DRAW FROM LAST DETECTIONS =====
            for det in detections:
                x1, y1, x2, y2 = map(int, det["box"])
                cls = det["class_name"]

                if cls == "person":
                    if self.toggles["intrusion"]():
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if cls in ["car", "truck", "bus"] and self.toggles["vehicle"]():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                if cls in ["knife", "scissors"] and self.toggles["threat"]():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # ===== MOTION (can stay every frame, it's cheap) =====
            if self.toggles["motion"]():
                boxes = self.motion.detect(frame)
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # ===== EMERGENCY (every frame) =====
            if self.toggles.get("emergency") and self.toggles["emergency"]():
                # placeholder → your mediapipe logic here
                pass

            self.frame_ready.emit(frame, person_count)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
