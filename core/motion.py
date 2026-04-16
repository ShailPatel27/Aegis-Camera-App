import cv2
from config.settings import MOTION_THRESHOLD, MOTION_MIN_AREA


class MotionDetector:
    def __init__(self):
        self.prev_gray = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_boxes = []

        if self.prev_gray is None:
            self.prev_gray = gray
            return motion_boxes

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < MOTION_MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)
            motion_boxes.append((x, y, x + w, y + h))

        self.prev_gray = gray

        return motion_boxes