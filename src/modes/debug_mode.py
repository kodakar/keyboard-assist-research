from src.core.camera import Camera
from src.core.hand_tracker import HandTracker
from src.input.keyboard_tracker import KeyboardTracker
from src.input.data_collector import DataCollector
from src.input.keyboard_map import KeyboardMap
from src.processing.filters.moving_average import MovingAverageFilter
from src.ui.display_manager import DisplayManager
from src.ui.key_formatter import KeyFormatter
from src.utils.error_handler import ErrorHandler, SystemStatus, ErrorType
import cv2
import os
import json
import time

def run_debug_mode():
    # エラーハンドラーの初期化
    error_handler = ErrorHandler()
    error_handler.set_status(SystemStatus.INITIALIZING, "Starting debug mode")
    
    # 変数を事前に定義（スコープ問題対策）
    camera = None
    keyboard_tracker = None
    total_keys_pressed = 0
    supported_keys_pressed = 0
    support_rate = 0
    raw_keystrokes = []
    
    try:
        # データ保存用ディレクトリの作成
        os.makedirs('data', exist_ok=True)
        
        # コンポーネントの初期化
        print("Initializing system components...")
        
        # カメラ初期化
        error_handler.set_status(SystemStatus.INITIALIZING, "Initializing camera")
        camera = Camera()
        if not error_handler.check_camera_status(camera):
            print("Failed to initialize camera. Please check camera connection.")
            error_handler.save_error_log()
            return
        
        # その他コンポーネント初期化
        hand_tracker = HandTracker()
        keyboard_tracker = KeyboardTracker()
        data_collector = DataCollector()
        
        # キーボードマップの初期化
        error_handler.set_status(SystemStatus.INITIALIZING, "Loading keyboard map")
        keyboard_map = KeyboardMap()
        
        if not error_handler.check_keyboard_map_status(keyboard_map):
            print("Keyboard map error detected. Please check keyboard mapping.")
        
        # 震え補正フィルターの初期化
        tremor_filter = MovingAverageFilter(window_size=8)
        
        # DisplayManagerの初期化
        display_manager = DisplayManager()
        
        # キーボードマップが存在しない場合、手動キャリブレーション
        if not os.path.exists('keyboard_map.json'):
            error_handler.set_status(SystemStatus.CALIBRATING, "Starting manual calibration")
            print("キーボードマップが見つかりません。手動キャリブレーションを開始します")
            try:
                keyboard_map.setup_manual_calibration()
                error_handler.set_status(SystemStatus.RUNNING, "Calibration completed")
            except Exception as e:
                error_handler.add_error(ErrorType.KEYBOARD_MAP_ERROR, f"Calibration failed: {str(e)}")
                return
        
        # キーボードトラッカー開始
        try:
            keyboard_tracker.start()
        except Exception as e:
            error_handler.add_error(ErrorType.SYSTEM_ERROR, f"Failed to start keyboard tracker: {str(e)}")
            return

        # システム開始
        error_handler.set_status(SystemStatus.RUNNING, "Debug mode ready")

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
        
        # サポートされているキーの統計
        supported_keys_pressed = 0
        total_keys_pressed = 0
        
        # 手検出失敗カウンター
        hand_detection_failures = 0
        
        # ウィンドウ作成
        cv2.namedWindow('Debug Mode - Hand Tracking')
        
        print("デバッグモードを開始しました")
        print(f"サポートキー: 英字(26), 数字(10), スペース(1) = 計37キー")
        print("ESCキーで終了します")
        
        try:
            while True:
                frame_start_time = time.time()
                
                # フレーム読み取り
                frame = camera.read_frame()
                if frame is None:
                    error_handler.add_error(ErrorType.CAMERA_ERROR, "Failed to read frame")
                    break
                
                # 手検出
                try:
                    results = hand_tracker.detect_hands(frame)
                    hand_tracker.draw_landmarks(frame, results)
                    
                    # 手検出状態チェック
                    if not error_handler.check_hand_detection_status(results, hand_detection_failures):
                        hand_detection_failures += 1
                    else:
                        hand_detection_failures = 0  # 成功時はリセット
                        
                except Exception as e:
                    error_handler.add_error(ErrorType.HAND_DETECTION_ERROR, f"Hand detection failed: {str(e)}")
                    continue
                
                # キーボードマップを描画
                try:
                    if keyboard_map:
                        keyboard_map.visualize(frame)
                except Exception as e:
                    error_handler.add_warning(f"Keyboard visualization error: {str(e)}")

                current_nearest_key = None
                
                # 手のランドマークが検出された場合、最も近いキーを特定
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    try:
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
                        
                    except Exception as e:
                        error_handler.add_warning(f"Hand tracking error: {str(e)}")

                # 実際のキー入力を処理
                key = keyboard_tracker.get_key_event()
                if key:
                    total_keys_pressed += 1
                    
                    try:
                        # サポートキーの統計
                        if KeyFormatter.is_supported_key(key):
                            supported_keys_pressed += 1
                            
                            # サポートされているキーの場合のみ処理
                            if results.multi_hand_landmarks and current_nearest_key:
                                # データを収集
                                data_collector.add_sample(key, results.multi_hand_landmarks[0])
                                
                                # コンソール出力も整形
                                key_display = KeyFormatter.format_for_display(key)
                                predicted_display = KeyFormatter.format_for_display(current_nearest_key)
                                print(f"Pressed: {key_display}, Predicted: {predicted_display}")
                                
                                # 履歴に追加
                                raw_keystrokes.append(key)
                                predicted_keys.append(current_nearest_key)
                                
                                # 予測キーを表示用に保存
                                last_predicted_key = current_nearest_key
                                predicted_key_timer = predicted_key_duration
                        else:
                            # サポートされていないキーの場合
                            print(f"Unsupported key: {key} (ignored)")
                            
                    except Exception as e:
                        error_handler.add_warning(f"Key processing error: {str(e)}")

                # タイマーを減らす
                if predicted_key_timer > 0:
                    predicted_key_timer -= 1

                # FPS計算
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

                # サポート率の計算
                support_rate = (supported_keys_pressed / total_keys_pressed * 100) if total_keys_pressed > 0 else 0

                # エラーハンドラーの表示タイマー更新
                error_handler.update_display_timer()
                
                # エラー・状態情報を取得
                error_info = error_handler.get_display_info()
                health_info = error_handler.get_system_health()

                # 表示情報を辞書で準備（KeyFormatterで整形）
                info = {
                    'current_prediction': KeyFormatter.format_for_display(current_nearest_key) if current_nearest_key else "None",
                    'last_prediction': KeyFormatter.format_for_display(last_predicted_key) if predicted_key_timer > 0 and last_predicted_key else None,
                    'actual_history': ''.join(KeyFormatter.format_for_test_history(k) for k in raw_keystrokes[-10:]) if raw_keystrokes else "",
                    'predicted_history': ''.join(KeyFormatter.format_for_test_history(k) for k in predicted_keys[-10:]) if predicted_keys else "",
                    'hand_detected': results.multi_hand_landmarks is not None,
                    'system_status': f"Debug ({error_info['system_status']}, Support: {support_rate:.0f}%)",
                    'fps': current_fps,
                    # エラー情報を追加
                    'error_message': error_info['status_message'] if error_info['show_message'] else None,
                    'error_count': error_info['error_count'],
                    'warning_count': error_info['warning_count'],
                    'system_health': f"Errors: {health_info['recent_errors']}, Warnings: {health_info['recent_warnings']}"
                }
                
                # DisplayManagerで統一表示
                display_manager.render_frame(frame, info, mode='debug')
                
                cv2.imshow('Debug Mode - Hand Tracking', frame)
                
                # ESCキーで終了
                if cv2.waitKey(1) & 0xFF == 27:
                    error_handler.set_status(SystemStatus.COMPLETED, "User requested exit")
                    break
            
        except KeyboardInterrupt:
            error_handler.set_status(SystemStatus.COMPLETED, "Interrupted by user")
            print("\nDebug mode interrupted by user")
        
        except Exception as e:
            error_handler.add_error(ErrorType.SYSTEM_ERROR, f"Unexpected error: {str(e)}")
            print(f"Unexpected error occurred: {e}")
    
    finally:
        try:
            # 統計情報を表示
            print(f"\n{'='*50}")
            print(f"デバッグセッション終了")
            print(f"{'='*50}")
            print(f"最終状態: {error_handler.current_status.value}")
            print(f"総キー入力数: {total_keys_pressed}")
            print(f"サポートキー入力数: {supported_keys_pressed}")
            print(f"サポート率: {support_rate:.1f}%")
            
            if raw_keystrokes:
                # 整形された履歴を表示
                formatted_history = ''.join(KeyFormatter.format_for_test_history(k) for k in raw_keystrokes)
                print(f"入力履歴: '{formatted_history}'")
            
            # システムヘルス情報
            health = error_handler.get_system_health()
            print(f"システムヘルス:")
            print(f"  稼働時間: {health['uptime']:.1f}秒")
            print(f"  総エラー数: {health['total_errors']}")
            print(f"  総警告数: {health['total_warnings']}")
            
            print(f"サポートキー詳細: {KeyFormatter.get_supported_keys_info()}")
            
            # エラーログを保存
            if error_handler.error_log or error_handler.warnings:
                error_handler.save_error_log()
            
            # データを保存
            data_collector.save_to_file()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        finally:
            # リソース解放
            try:
                if camera:
                    camera.release()
                if keyboard_tracker:
                    keyboard_tracker.stop()
                cv2.destroyAllWindows()
                print("Resources cleaned up successfully")
            except Exception as e:
                print(f"Error during resource cleanup: {e}")