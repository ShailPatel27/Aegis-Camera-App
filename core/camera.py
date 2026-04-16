import cv2
from config.settings import CAMERA_INDEX

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
