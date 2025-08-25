# src/camera.py
import cv2

class Camera:
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"⚠️ カメラインデックス {camera_index} でカメラを開けませんでした")
            # 他のカメラインデックスを試す
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"✅ カメラインデックス {i} でカメラを開きました")
                    break
        
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
        
    def is_opened(self):
        """カメラが開いているか確認"""
        return self.cap.isOpened()
    
    def get_camera_info(self):
        """カメラ情報を取得"""
        if self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'is_opened': True
            }
        return {'is_opened': False}
    
    def release(self):
        self.cap.release()