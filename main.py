from src.camera import Camera
from src.hand_tracker import HandTracker
from src.keyboard_tracker import KeyboardTracker
from src.data_collector import DataCollector
from src.keyboard_detector import KeyboardDetector
import cv2
import os

def main():
    # データ保存用ディレクトリの作成
    os.makedirs('data', exist_ok=True)
    camera = Camera()
    hand_tracker = HandTracker()
    keyboard_tracker = KeyboardTracker()
    data_collector = DataCollector()
    keyboard_detector = KeyboardDetector()
    
    keyboard_tracker.start()
    
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                break
            
            # キーボード検出
            # debug_frame = keyboard_detector.detect_keyboard(frame)

            results = hand_tracker.detect_hands(frame)
            hand_tracker.draw_landmarks(frame, results)
            
            key = keyboard_tracker.get_key_event()
            if key and results.multi_hand_landmarks:
                # データを収集
                data_collector.add_sample(key, results.multi_hand_landmarks[0])
                print(f"Collected data for key: {key}")
            
            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 終了時にデータを保存
        data_collector.save_to_file()
        camera.release()
        keyboard_tracker.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()