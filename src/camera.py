# src/camera.py
import cv2

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
        
    def release(self):
        self.cap.release()