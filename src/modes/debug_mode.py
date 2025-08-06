from src.core.camera import Camera
from src.core.hand_tracker import HandTracker
from src.input.keyboard_tracker import KeyboardTracker
from src.input.data_collector import DataCollector
from src.input.keyboard_map import KeyboardMap
from src.processing.filters.moving_average import MovingAverageFilter
from src.ui.display_manager import DisplayManager
import cv2
import os
import json
import time

def run_debug_mode():
    # データ保存用ディレクトリの作成
    os.makedirs('data', exist_ok=True)
    camera = Camera()
    hand_tracker = HandTracker()
    keyboard_tracker = KeyboardTracker()
    data_collector = DataCollector()
    
    # キーボードマップの初期化
    keyboard_map = KeyboardMap()

    # 震え補正フィルターの初期化
    tremor_filter = MovingAverageFilter(window_size=8)
    
    # DisplayManagerの初期化
    display_manager = DisplayManager()
    
    # キーボードマップが存在しない場合、OCR検出を試みる
    if not os.path.exists('keyboard_map.json'):
        print("キーボードマップが見つかりません。手動キャリブレーションを開始します")
        keyboard_map.setup_manual_calibration()
    
    keyboard_tracker.start()

    # 履歴用の変数を追加
    raw_keystrokes = []
    predicted_keys = []  # 予測キーの履歴
    
    # 最後の予測キーとその持続表示用タイマー
    last_predicted_key = None
    predicted_key_timer = 0
    predicted_key_duration = 60  # 60フレーム（約2秒）表示し続ける
    
    # FPS計算用
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    
    # ウィンドウ作成
    cv2.namedWindow('Debug Mode - Hand Tracking')
    
    try:
        while True:
            frame_start_time = time.time()
            
            frame = camera.read_frame()
            if frame is None:
                break
            
            results = hand_tracker.detect_hands(frame)
            hand_tracker.draw_landmarks(frame, results)
            
            # キーボードマップを描画
            if keyboard_map:
                keyboard_map.visualize(frame)

            current_nearest_key = None
            
            # 手のランドマークが検出された場合、最も近いキーを特定
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                # 人差し指の先端（ランドマークインデックス8）を取得
                index_finger = results.multi_hand_landmarks[0].landmark[8]
                
                # 震え補正フィルターを適用
                filtered_x, filtered_y, filtered_z = tremor_filter.filter(index_finger)
                
                # 元の位置を表示（赤色）
                raw_x, raw_y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
                cv2.circle(frame, (raw_x, raw_y), 5, (0, 0, 255), -1)
                
                # フィルタリング後の位置を表示（緑色）
                filtered_pos_x = int(filtered_x * frame.shape[1])
                filtered_pos_y = int(filtered_y * frame.shape[0])
                cv2.circle(frame, (filtered_pos_x, filtered_pos_y), 8, (0, 255, 0), -1)
                
                # 線で結ぶ
                cv2.line(frame, (raw_x, raw_y), (filtered_pos_x, filtered_pos_y), (255, 0, 0), 2)
                
                # 最も近いキーを取得（フィルタリング後の位置から）
                nearest_key, distance = keyboard_map.get_nearest_key(filtered_x, filtered_y)
                current_nearest_key = nearest_key

            # 実際のキー入力を処理
            key = keyboard_tracker.get_key_event()
            if key and results.multi_hand_landmarks and current_nearest_key:
                # データを収集
                data_collector.add_sample(key, results.multi_hand_landmarks[0])
                print(f"Pressed: {key}, Predicted: {current_nearest_key}")
                
                # 履歴に追加
                raw_keystrokes.append(key)
                predicted_keys.append(current_nearest_key)
                
                # 予測キーを表示用に保存
                last_predicted_key = current_nearest_key
                predicted_key_timer = predicted_key_duration

            # タイマーを減らす
            if predicted_key_timer > 0:
                predicted_key_timer -= 1

            # FPS計算
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            # 表示情報を辞書で準備
            info = {
                'current_prediction': current_nearest_key if current_nearest_key else "None",
                'last_prediction': last_predicted_key if predicted_key_timer > 0 else None,
                'actual_history': ''.join(str(k) for k in raw_keystrokes[-10:]) if raw_keystrokes else "",
                'predicted_history': ''.join(str(k) for k in predicted_keys[-10:]) if predicted_keys else "",
                'hand_detected': results.multi_hand_landmarks is not None,
                'system_status': "Running",
                'fps': current_fps
            }
            
            # DisplayManagerで統一表示
            display_manager.render_frame(frame, info, mode='debug')
            
            cv2.imshow('Debug Mode - Hand Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # データを保存
        data_collector.save_to_file()
        camera.release()
        keyboard_tracker.stop()
        cv2.destroyAllWindows()