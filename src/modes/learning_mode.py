# src/modes/learning_mode.py
"""
学習モード
ユーザーに指定テキストを入力してもらい、手の軌跡データを収集して学習を実行
"""

import cv2
import numpy as np
import time
from datetime import datetime
import os
import sys

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.camera import Camera
from src.core.hand_tracker import HandTracker
from src.input.keyboard_map import KeyboardMap
from src.input.keyboard_tracker import KeyboardTracker
from src.processing.enhanced_data_collector import EnhancedDataCollector
from src.processing.models.hand_lstm import BasicHandLSTM, HandLSTMTrainer
from src.ui.display_manager import DisplayManager

class LearningMode:
    def __init__(self, user_id: str = "user_001"):
        """
        学習モードの初期化
        
        Args:
            user_id: ユーザーID
        """
        self.user_id = user_id
        
        # コンポーネントの初期化
        self.camera = Camera()
        self.hand_tracker = HandTracker()
        self.keyboard_map = KeyboardMap()
        self.keyboard_tracker = KeyboardTracker()
        self.data_collector = EnhancedDataCollector(user_id=user_id)
        self.display_manager = DisplayManager()
        
        # 学習モデル
        self.model = None
        self.trainer = None
        
        # 学習状態
        self.is_learning = False
        self.current_target_text = ""
        self.current_input_text = ""
        
        # 表示設定
        self.show_trajectory = True
        self.show_predictions = True
        
        # 進捗管理
        self.progress = {
            'current': 0,
            'total': 0,
            'start_time': None
        }
        
        print(f"🎓 学習モードを初期化しました")
        print(f"   ユーザーID: {user_id}")
    
    def run_learning_mode(self, target_text: str = "hello world"):
        """
        学習モードを実行
        
        Args:
            target_text: 目標とする入力テキスト
        """
        self.current_target_text = target_text
        self.current_input_text = ""
        
        print(f"\n🎯 学習モード開始")
        print(f"   目標テキスト: {target_text}")
        print(f"   操作説明:")
        print(f"     - 手をカメラに映して、目標テキストを入力してください")
        print(f"     - 手の軌跡データが自動的に収集されます")
        print(f"     - SPACE: データ収集開始/停止")
        print(f"     - L: 学習実行")
        print(f"     - T: 軌跡表示切り替え")
        print(f"     - P: 予測表示切り替え")
        print(f"     - ESC: 終了")
        
        # データ収集セッション開始
        self.data_collector.start_collection_session(target_text)
        
        # メインループ
        self._main_loop()
    
    def _main_loop(self):
        """メインループ"""
        frame_count = 0
        start_time = time.time()
        
        while True:
            # フレーム取得
            frame = self.camera.read_frame()
            if frame is None:
                print("⚠️ カメラからのフレーム取得に失敗しました")
                break
            
            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time) if current_time > start_time else 0
            
            # 手の検出
            results = self.hand_tracker.detect_hands(frame)
            
            # 手の軌跡データを収集
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # 最初の手
                
                # 軌跡データを追加
                self.data_collector.add_hand_position(hand_landmarks, current_time)
                
                # 手の位置からキーを推定
                if self.keyboard_map.key_positions:
                    # 手の中心位置を取得（中指の付け根）
                    center_landmark = hand_landmarks.landmark[9]
                    h, w = frame.shape[:2]
                    hand_x = int(center_landmark.x * w)
                    hand_y = int(center_landmark.y * h)
                    
                    # 最も近いキーを取得
                    nearest_key, distance = self.keyboard_map.get_nearest_key(hand_x, hand_y)
                    
                    # 予測表示
                    if self.show_predictions and nearest_key:
                        self._display_prediction(frame, nearest_key, distance)
                
                # 手のランドマークを描画
                self.hand_tracker.draw_landmarks(frame, results)
            
            # キーボードマップを可視化
            frame = self.keyboard_map.visualize(frame)
            
            # 軌跡を可視化
            if self.show_trajectory:
                frame = self.data_collector.visualize_trajectory(frame)
            
            # 情報表示
            self._display_info(frame, fps)
            
            # フレーム表示
            cv2.imshow('Learning Mode', frame)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE
                self._toggle_data_collection()
            elif key == ord('l') or key == ord('L'):  # L
                self._run_training()
            elif key == ord('t') or key == ord('T'):  # T
                self.show_trajectory = not self.show_trajectory
                print(f"軌跡表示: {'ON' if self.show_trajectory else 'OFF'}")
            elif key == ord('p') or key == ord('P'):  # P
                self.show_predictions = not self.show_predictions
                print(f"予測表示: {'ON' if self.show_predictions else 'OFF'}")
        
        # クリーンアップ
        self._cleanup()
    
    def _toggle_data_collection(self):
        """データ収集の開始/停止を切り替え"""
        if self.data_collector.is_collecting:
            self.data_collector.stop_collection_session()
            print("⏸️ データ収集を一時停止しました")
        else:
            self.data_collector.start_collection_session(self.current_target_text)
            print("▶️ データ収集を再開しました")
    
    def _run_training(self):
        """学習を実行"""
        print(f"\n🚀 学習を開始します")
        
        # データセット情報を取得
        dataset_info = self.data_collector.get_training_dataset_info()
        if dataset_info['total_samples'] < 10:
            print(f"⚠️ 学習データが不足しています（必要: 10サンプル、現在: {dataset_info['total_samples']}サンプル）")
            return
        
        print(f"📊 学習データ: {dataset_info['total_samples']}サンプル")
        
        try:
            # 学習データセットをエクスポート
            dataset_file = self.data_collector.export_training_dataset()
            if not dataset_file:
                print("⚠️ データセットのエクスポートに失敗しました")
                return
            
            # データセットを読み込み
            with open(dataset_file, 'r', encoding='utf-8') as f:
                import json
                dataset = json.load(f)
            
            # モデルを作成
            if self.model is None:
                self.model = BasicHandLSTM()
                self.trainer = HandLSTMTrainer(self.model)
                print(f"✅ 新しいモデルを作成しました")
            
            # 学習データを準備
            train_data = self.trainer.prepare_training_data(dataset)
            
            # 学習を実行
            print(f"🎓 学習開始...")
            
            # 進捗追跡開始
            self._start_progress_tracking(50)  # 50エポック
            
            history = self.trainer.train(
                train_data=train_data,
                epochs=50,
                batch_size=16,
                early_stopping_patience=5
            )
            
            # モデルを保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(self.data_collector.data_dir, "models", self.user_id, f"hand_lstm_{timestamp}.pth")
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            
            self.model.save_model(model_file)
            
            # 学習履歴を保存
            history_file = os.path.join(self.data_collector.data_dir, "models", self.user_id, f"training_history_{timestamp}.pkl")
            self.trainer.save_training_history(history_file)
            
            print(f"✅ 学習完了！")
            print(f"   モデル: {model_file}")
            print(f"   履歴: {history_file}")
            
            # 進捗表示
            self._show_progress(100, 100)
            
        except Exception as e:
            print(f"⚠️ 学習エラー: {e}")
            print(f"エラータイプ: {type(e).__name__}")
            print(f"エラー詳細: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _show_progress(self, current: int, total: int):
        """進捗を表示"""
        if total > 0:
            progress = (current / total) * 100
            elapsed_time = ""
            if self.progress['start_time']:
                elapsed = time.time() - self.progress['start_time']
                elapsed_time = f" (経過時間: {elapsed:.1f}秒)"
            
            print(f"📊 進捗: {progress:.1f}% ({current}/{total}){elapsed_time}")
            
            # プログレスバーの表示
            bar_length = 30
            filled_length = int(bar_length * current // total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"   [{bar}] {progress:.1f}%")
    
    def _start_progress_tracking(self, total: int):
        """進捗追跡を開始"""
        self.progress['total'] = total
        self.progress['current'] = 0
        self.progress['start_time'] = time.time()
        print(f"🚀 進捗追跡開始: 目標 {total} ステップ")
    
    def _update_progress(self, current: int):
        """進捗を更新"""
        self.progress['current'] = current
        if self.progress['total'] > 0:
            self._show_progress(current, self.progress['total'])
    
    def _display_prediction(self, frame, predicted_key: str, confidence: float):
        """予測結果を表示"""
        h, w = frame.shape[:2]
        
        # 予測テキスト
        text = f"Pred: {predicted_key}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # テキストサイズを取得
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 背景を描画
        x = w - text_width - 20
        y = text_height + 20
        cv2.rectangle(frame, (x - 10, y - text_height - 10), 
                     (x + text_width + 10, y + 10), (0, 0, 0), -1)
        
        # テキストを描画
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)
    
    def _display_info(self, frame, fps: float):
        """情報を表示"""
        h, w = frame.shape[:2]
        
        # 基本情報
        info_lines = [
            f"Learning Mode - {self.user_id}",
            f"Target: {self.current_target_text}",
            f"Input: {self.current_input_text}",
            f"FPS: {fps:.1f}",
            f"Samples: {self.data_collector.stats['total_samples']}",
            f"Trajectories: {self.data_collector.stats['total_trajectories']}"
        ]
        
        # 操作説明
        controls = [
            "SPACE: Toggle Collection",
            "L: Run Training",
            "T: Toggle Trajectory",
            "P: Toggle Predictions",
            "ESC: Exit"
        ]
        
        # 情報を描画
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 操作説明を描画（右側）
        for i, line in enumerate(controls):
            cv2.putText(frame, line, (w - 200, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # データ収集状態
        status = "COLLECTING" if self.data_collector.is_collecting else "PAUSED"
        status_color = (0, 255, 0) if self.data_collector.is_collecting else (0, 0, 255)
        cv2.putText(frame, status, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    def _cleanup(self):
        """クリーンアップ処理"""
        print(f"\n🧹 学習モードを終了します")
        
        # データ収集セッションを停止
        if self.data_collector.is_collecting:
            self.data_collector.stop_collection_session()
        
        # カメラを解放
        self.camera.release()
        cv2.destroyAllWindows()
        
        # 最終統計
        dataset_info = self.data_collector.get_training_dataset_info()
        print(f"📊 最終統計:")
        print(f"   総サンプル数: {dataset_info['total_samples']}")
        print(f"   総軌跡数: {dataset_info['total_trajectories']}")
        print(f"   データディレクトリ: {dataset_info['samples_directory']}")


def main():
    """メイン関数"""
    print("🎓 キーボード入力支援システム - 学習モード")
    print("=" * 50)
    
    # ユーザーIDの入力
    user_id = input("ユーザーIDを入力してください (デフォルト: user_001): ").strip()
    if not user_id:
        user_id = "user_001"
    
    # 目標テキストの入力
    target_text = input("目標テキストを入力してください (デフォルト: hello world): ").strip()
    if not target_text:
        target_text = "hello world"
    
    # 学習モードを開始
    learning_mode = LearningMode(user_id=user_id)
    
    try:
        learning_mode.run_learning_mode(target_text)
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
    except Exception as e:
        print(f"\n⚠️ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        learning_mode._cleanup()


if __name__ == "__main__":
    main()
