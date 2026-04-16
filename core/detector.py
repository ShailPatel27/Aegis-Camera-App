from ultralytics import YOLO
from config.settings import YOLO_MODEL_PATH


class Detector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class_name": results.names[cls_id],
                "confidence": conf,
                "box": xyxy
            })

        return detections