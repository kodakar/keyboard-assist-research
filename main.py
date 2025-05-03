from src.camera import Camera
from src.hand_tracker import HandTracker
from src.keyboard_tracker import KeyboardTracker
from src.data_collector import DataCollector
from src.keyboard_detector import KeyboardDetector
from src.keyboard_map import *
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
    
     # キーボードマップの初期化
    keyboard_map = KeyboardMap()
    
    # キーボードマップが存在しない場合、OCR検出を試みる
    if not os.path.exists('keyboard_map.json'):
        print("キーボードマップが見つかりません。手動キャリブレーションを開始します")
        keyboard_map.setup_manual_calibration()
        keyboard_tracker.start()

    # ウィンドウ作成
    cv2.namedWindow('Hand Tracking')
    
    
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                break
            
            # キーボード検出
            # debug_frame = keyboard_detector.detect_keyboard(frame)

            results = hand_tracker.detect_hands(frame)
            hand_tracker.draw_landmarks(frame, results)
            
            # キーボードマップを描画
            if keyboard_map:
                keyboard_map.visualize(frame)
            
            # 手のランドマークが検出された場合、最も近いキーを特定
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                # 人差し指の先端（ランドマークインデックス8）を取得
                index_finger = results.multi_hand_landmarks[0].landmark[8]
                nearest_key, distance = keyboard_map.get_nearest_key(index_finger.x, index_finger.y)
                
                # 最も近いキーを表示
                cv2.putText(frame, f"Nearest: {nearest_key}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            key = keyboard_tracker.get_key_event()
            if key and results.multi_hand_landmarks:
                # データを収集
                data_collector.add_sample(key, results.multi_hand_landmarks[0])
                print(f"Collected data for key: {key}")
                # キー入力を画面に表示
                cv2.putText(frame, f"Pressed: {key}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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