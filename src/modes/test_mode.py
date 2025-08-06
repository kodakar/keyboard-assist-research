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

def run_test_mode(test_text):
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

    # 実験設定
    EXPERIMENT_TEXT = test_text
    experiment_index = 0
    experiment_results = []

    # 履歴用の変数を追加
    raw_keystrokes = []
    predicted_keys = []  # 予測キーの履歴
    
    # 最後の予測キーとその持続表示用タイマー
    last_predicted_key = None
    predicted_key_timer = 0
    predicted_key_duration = 60  # 60フレーム（約2秒）表示し続ける
    
    # 結果表示用タイマー
    result_display_timer = 0
    result_display_duration = 90  # 90フレーム（約3秒）表示
    last_result = None
    
    # FPS計算用
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    
    # ウィンドウ作成
    cv2.namedWindow('Test Mode - Keyboard Input Test')
    
    print(f"テストを開始します。「{test_text}」を入力してください。")
    
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
                
                if experiment_index < len(EXPERIMENT_TEXT):
                    intended_key = EXPERIMENT_TEXT[experiment_index]
                    
                    # 実験結果を記録
                    is_prediction_correct = (str(current_nearest_key) == str(intended_key))
                    
                    experiment_results.append({
                        'intended': intended_key,
                        'actual': key,
                        'predicted': current_nearest_key,
                        'prediction_correct': is_prediction_correct,
                        'timestamp': time.time()
                    })
                    
                    # 結果を保存してタイマー設定
                    last_result = "CORRECT" if is_prediction_correct else "WRONG"
                    result_display_timer = result_display_duration
                    
                    experiment_index += 1
                    
                    # テスト完了チェック
                    if experiment_index >= len(EXPERIMENT_TEXT):
                        print("テスト完了！結果を確認してください。")
                
                # 履歴に追加
                raw_keystrokes.append(key)
                predicted_keys.append(current_nearest_key)
                
                # 予測キーを表示用に保存
                last_predicted_key = current_nearest_key
                predicted_key_timer = predicted_key_duration

            # タイマーを減らす
            if predicted_key_timer > 0:
                predicted_key_timer -= 1
            if result_display_timer > 0:
                result_display_timer -= 1

            # FPS計算
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            # 現在の精度を計算
            current_accuracy = 0.0
            if experiment_results:
                correct_predictions = sum(1 for r in experiment_results if r['prediction_correct'])
                current_accuracy = correct_predictions / len(experiment_results) * 100

            # 表示情報を辞書で準備
            info = {
                'current_prediction': current_nearest_key if current_nearest_key else "None",
                'last_prediction': last_predicted_key if predicted_key_timer > 0 else None,
                'actual_history': ''.join(str(k) for k in raw_keystrokes[-10:]) if raw_keystrokes else "",
                'predicted_history': ''.join(str(k) for k in predicted_keys[-10:]) if predicted_keys else "",
                'hand_detected': results.multi_hand_landmarks is not None,
                'system_status': "Testing",
                'fps': current_fps
            }
            
            # テスト専用の情報を追加
            if experiment_index < len(EXPERIMENT_TEXT):
                info['test_instruction'] = EXPERIMENT_TEXT[experiment_index]
                info['test_progress'] = f"{experiment_index + 1}/{len(EXPERIMENT_TEXT)}"
            else:
                info['test_instruction'] = "Complete!"
                info['test_progress'] = f"{len(EXPERIMENT_TEXT)}/{len(EXPERIMENT_TEXT)}"
            
            # 結果表示
            if result_display_timer > 0 and last_result:
                info['test_result'] = last_result
            
            # 精度表示
            if experiment_results:
                info['accuracy'] = current_accuracy
            
            # DisplayManagerで統一表示
            display_manager.render_frame(frame, info, mode='test')
            
            cv2.imshow('Test Mode - Keyboard Input Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # テスト完了で自動終了（オプション）
            # if experiment_index >= len(EXPERIMENT_TEXT):
            #     time.sleep(3)  # 3秒結果を表示してから終了
            #     break
    
    finally:
        # 実験結果を保存
        if experiment_results:
            # タイムスタンプ付きのファイル名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'data/test_results_{timestamp}.json'
            
            # 結果データに追加情報を含める
            result_data = {
                'test_text': test_text,
                'total_characters': len(test_text),
                'completed_characters': len(experiment_results),
                'timestamp': timestamp,
                'results': experiment_results
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            # 実験結果のサマリーを表示
            correct_predictions = sum(1 for r in experiment_results if r['prediction_correct'])
            total_predictions = len(experiment_results)
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions * 100
                print(f"\n{'='*50}")
                print(f"テスト結果サマリー")
                print(f"{'='*50}")
                print(f"テスト文字列: '{test_text}'")
                print(f"完了文字数: {total_predictions}/{len(test_text)}")
                print(f"予測精度: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
                print(f"結果保存先: {filename}")
                
                # 間違った予測の詳細を表示
                wrong_predictions = [r for r in experiment_results if not r['prediction_correct']]
                if wrong_predictions:
                    print(f"\n間違った予測 ({len(wrong_predictions)}件):")
                    for i, wp in enumerate(wrong_predictions[:5], 1):  # 最初の5件のみ表示
                        print(f"  {i}. 意図: '{wp['intended']}', 実際: '{wp['actual']}', 予測: '{wp['predicted']}'")
                    if len(wrong_predictions) > 5:
                        print(f"  ... 他{len(wrong_predictions)-5}件")
                print(f"{'='*50}")
        
        # データを保存
        data_collector.save_to_file()
        camera.release()
        keyboard_tracker.stop()
        cv2.destroyAllWindows()