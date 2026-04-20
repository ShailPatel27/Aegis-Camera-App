from ultralytics import YOLO
from config.settings import YOLO_MODEL_PATH, YOLO_MIN_CONFIDENCE


class Detector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < YOLO_MIN_CONFIDENCE:
                continue
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class_name": results.names[cls_id],
                "confidence": conf,
                "box": xyxy
            })

        return detections

    def detect_with_botsort(self, frame):
        """
        Runs YOLO once with BOT-SORT tracking and returns:
        - detections: same shape as detect()
        - person_track_ids: set[int] for tracked person objects in this frame
        """
        results = self.model.track(
            frame,
            verbose=False,
            persist=True,
            tracker="botsort.yaml",
        )[0]

        detections = []
        person_track_ids = set()

        boxes = results.boxes
        ids_tensor = getattr(boxes, "id", None)

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < YOLO_MIN_CONFIDENCE:
                continue

            xyxy = box.xyxy[0].tolist()
            class_name = results.names[cls_id]
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": conf,
                    "box": xyxy,
                }
            )

            if class_name == "person" and ids_tensor is not None:
                try:
                    track_id = int(ids_tensor[idx].item())
                    person_track_ids.add(track_id)
                except Exception:
                    pass

        return detections, person_track_ids
