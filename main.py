from src.camera import Camera
from src.hand_tracker import HandTracker
from src.keyboard_tracker import KeyboardTracker
from src.data_collector import DataCollector
from src.keyboard_detector import KeyboardDetector
import cv2
import os

def main():
    # データ保存用ディレクトリの作成
    print(0)
    os.makedirs('data', exist_ok=True)
    print(1)
    camera = Camera()
    print(12)
    hand_tracker = HandTracker()
    print(13)
    keyboard_tracker = KeyboardTracker()
    print(14)
    data_collector = DataCollector()
    print(15)
    keyboard_detector = KeyboardDetector()
    print(16)
    
    keyboard_tracker.start()
    
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                break
            
            # キーボード検出
            debug_frame = keyboard_detector.detect_keyboard(frame)

            results = hand_tracker.detect_hands(debug_frame)
            hand_tracker.draw_landmarks(debug_frame, results)
            
            key = keyboard_tracker.get_key_event()
            if key and results.multi_hand_landmarks:
                # データを収集
                data_collector.add_sample(key, results.multi_hand_landmarks[0])
                print(f"Collected data for key: {key}")
            
            # cv2.imshow('Hand Tracking', frame)
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