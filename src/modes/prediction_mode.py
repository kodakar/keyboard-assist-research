# src/modes/prediction_mode.py
"""
リアルタイム予測モード
学習済みモデルを使って入力意図を予測
"""

import os
import cv2
import numpy as np
import torch
import time
from collections import deque
from typing import List, Tuple, Optional, Dict
import json

# 既存のモジュールをインポート
from src.core.camera import Camera
from src.core.hand_tracker import HandTracker
from src.input.keyboard_map import KeyboardMap
from src.processing.coordinate_transformer import CoordinateTransformer
from src.processing.models.hand_lstm import BasicHandLSTM


class PredictionMode:
    """リアルタイム予測モードクラス"""
    
    def __init__(self, model_path: str, keyboard_map_path: str = 'keyboard_map.json'):
        """
        予測モードの初期化
        
        Args:
            model_path: 学習済みモデルのパス
            keyboard_map_path: キーボードマップのパス
        """
        self.model_path = model_path
        self.keyboard_map_path = keyboard_map_path
        
        # コンポーネントの初期化
        self.camera = None
        self.hand_tracker = None
        self.keyboard_map = None
        self.coordinate_transformer = None
        self.model = None
        
        # 予測用のバッファ
        self.frame_buffer = deque(maxlen=60)  # 60フレーム分の特徴量
        self.trajectory_buffer = deque(maxlen=30)  # 30フレーム分の軌跡
        
        # 予測結果
        self.current_prediction = None
        self.prediction_history = []
        
        # 評価モード用
        self.evaluation_mode = False
        self.test_text = "hello world"
        self.current_char_index = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # デバッグ情報
        self.show_debug = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # 37キーの定義
        self.KEY_CHARS = (
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' '
        )
        
        print(f"🎯 予測モード初期化完了")
        print(f"   モデルパス: {model_path}")
        print(f"   キーボードマップ: {keyboard_map_path}")
    
    def initialize_components(self) -> bool:
        """コンポーネントの初期化"""
        try:
            print("🔧 コンポーネントを初期化中...")
            
            # カメラの初期化
            self.camera = Camera()
            if not self.camera.is_opened():
                print("❌ カメラの初期化に失敗しました")
                return False
            
            # 画面サイズを設定
            frame = self.camera.read_frame()
            if frame is None:
                print("❌ カメラからのフレーム取得に失敗しました")
                return False
            
            height, width = frame.shape[:2]
            print(f"✅ カメラ初期化完了: {width}x{height}")
            
            # 手追跡の初期化
            self.hand_tracker = HandTracker()
            print("✅ 手追跡初期化完了")
            
            # キーボードマップの初期化
            self.keyboard_map = KeyboardMap()
            if not self.keyboard_map.key_positions:
                print("⚠️ キーボードマッピングが存在しません")
                print("   マッピングを開始します...")
                if not self.keyboard_map.start_calibration():
                    print("❌ キーボードマッピングに失敗しました")
                    return False
            
            # 座標変換器の初期化
            self.coordinate_transformer = CoordinateTransformer(self.keyboard_map_path)
            self.coordinate_transformer.set_screen_size(width, height)
            self.coordinate_transformer.set_keyboard_corners(
                self.keyboard_map.get_keyboard_corners()
            )
            print("✅ 座標変換器初期化完了")
            
            # モデルの読み込み
            if not self.load_model():
                print("❌ モデルの読み込みに失敗しました")
                return False
            
            print("✅ コンポーネント初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ コンポーネント初期化エラー: {e}")
            return False
    
    def load_model(self) -> bool:
        """学習済みモデルの読み込み"""
        try:
            print("🤖 モデルを読み込み中...")
            
            # チェックポイントの読み込み
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # モデル設定の取得
            model_config = checkpoint.get('model_config', {})
            input_size = model_config.get('input_size', 15)
            hidden_size = model_config.get('hidden_size', 128)
            num_classes = model_config.get('num_classes', 37)
            
            # モデルの初期化
            self.model = BasicHandLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_classes=num_classes
            )
            
            # 重みの読み込み
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # デバイスの設定
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            print(f"✅ モデル読み込み完了")
            print(f"   入力サイズ: {input_size}")
            print(f"   隠れ層サイズ: {hidden_size}")
            print(f"   クラス数: {num_classes}")
            print(f"   使用デバイス: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            return False
    
    def extract_features(self, hand_landmarks) -> Optional[np.ndarray]:
        """手のランドマークから特徴量を抽出"""
        try:
            # キーボード空間での指の座標を取得
            index_finger = hand_landmarks.landmark[8]  # 人差し指先端
            
            # ピクセル座標をキーボード空間に変換
            kb_coords = self.coordinate_transformer.pixel_to_keyboard_space(
                index_finger.x, index_finger.y
            )
            
            if kb_coords is None:
                return None
            
            kb_x, kb_y = kb_coords
            
            # 最近傍3キーへの相対座標を取得
            nearest_keys = self.coordinate_transformer.get_nearest_keys_with_relative_coords(
                kb_x, kb_y, top_k=3
            )
            
            # 特徴量の構築（15次元）
            features = np.zeros(15)
            
            # キーボード空間での指の座標（2次元）
            features[0] = kb_x
            features[1] = kb_y
            
            # 最近傍3キーへの相対座標（6次元）
            for i, key_info in enumerate(nearest_keys[:3]):
                if i < 3:
                    features[2 + i*2] = key_info.relative_x
                    features[2 + i*2 + 1] = key_info.relative_y
            
            # 最近傍3キーへの距離（3次元）
            for i, key_info in enumerate(nearest_keys[:3]):
                if i < 3:
                    features[8 + i] = key_info.distance
            
            # 速度・加速度（4次元）- 前フレームとの差分から計算
            if len(self.frame_buffer) > 0:
                prev_features = self.frame_buffer[-1]
                if prev_features is not None:
                    # 速度（X, Y方向）
                    features[11] = (kb_x - prev_features[0]) * 30  # 30fps
                    features[12] = (kb_y - prev_features[1]) * 30
                    
                    # 加速度（X, Y方向）
                    if len(self.frame_buffer) > 1:
                        prev_prev_features = self.frame_buffer[-2]
                        if prev_prev_features is not None:
                            features[13] = (prev_features[0] - 2*kb_x + prev_prev_features[0]) * 30**2
                            features[14] = (prev_features[1] - 2*kb_y + prev_prev_features[1]) * 30**2
            
            # 特徴量の正規化
            features = self.normalize_features(features)
            
            return features
            
        except Exception as e:
            print(f"⚠️ 特徴量抽出エラー: {e}")
            return None
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """特徴量の正規化"""
        # 座標系の正規化（0-1の範囲に収める）
        features[:2] = np.clip(features[:2], 0.0, 1.0)
        
        # 相対座標の正規化（-5から5の範囲に収める）
        features[2:8] = np.clip(features[2:8], -5.0, 5.0)
        
        # 距離の正規化（0から10の範囲に収める）
        features[8:11] = np.clip(features[8:11], 0.0, 10.0)
        
        # 速度・加速度の正規化（-2から2の範囲に収める）
        features[11:] = np.clip(features[11:], -2.0, 2.0)
        
        return features
    
    def predict_intent(self) -> Optional[List[Tuple[str, float]]]:
        """入力意図を予測"""
        try:
            if len(self.frame_buffer) < 60:
                return None
            
            # 特徴量をテンソルに変換
            features = np.array(list(self.frame_buffer))
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # 推論時間の計測
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            inference_time = time.time() - start_time
            
            # Top-3の予測結果を取得
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            
            predictions = []
            for i in range(3):
                key = self.KEY_CHARS[top3_indices[0][i].item()]
                confidence = top3_probs[0][i].item() * 100
                predictions.append((key, confidence))
            
            # 推論時間を記録
            self.inference_time = inference_time
            
            return predictions
            
        except Exception as e:
            print(f"⚠️ 予測エラー: {e}")
            return None
    
    def run_prediction_mode(self):
        """予測モードの実行"""
        print("🚀 予測モードを開始します")
        print("   操作説明:")
        print("   - D: デバッグ情報の表示/非表示")
        print("   - E: 評価モードの切り替え")
        print("   - R: 評価のリセット")
        print("   - ESC: 終了")
        
        try:
            while True:
                # フレーム取得
                frame = self.camera.read_frame()
                if frame is None:
                    print("❌ カメラからのフレーム取得に失敗しました")
                    break
                
                # 手の検出
                results = self.hand_tracker.detect_hands(frame)
                self.last_results = results  # 結果を保存
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # 特徴量の抽出
                    features = self.extract_features(hand_landmarks)
                    
                    if features is not None:
                        # フレームバッファに追加
                        self.frame_buffer.append(features)
                        
                        # 軌跡バッファに追加
                        index_finger = hand_landmarks.landmark[8]
                        self.trajectory_buffer.append((index_finger.x, index_finger.y))
                        
                        # 60フレーム溜まったら予測実行
                        if len(self.frame_buffer) == 60:
                            predictions = self.predict_intent()
                            if predictions:
                                self.current_prediction = predictions
                                self.prediction_history.append(predictions)
                
                # 画面表示の更新
                frame = self.update_display(frame)
                
                # 画面に表示
                cv2.imshow('Intent Prediction Mode', frame)
                
                # キー入力の処理
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('d') or key == ord('D'):  # Dキーでデバッグ情報トグル
                    self.show_debug = not self.show_debug
                    print(f"デバッグ情報: {'表示' if self.show_debug else '非表示'}")
                elif key == ord('e') or key == ord('E'):  # Eキーで評価モード切り替え
                    self.evaluation_mode = not self.evaluation_mode
                    print(f"評価モード: {'有効' if self.evaluation_mode else '無効'}")
                elif key == ord('r') or key == ord('R'):  # Rキーで評価リセット
                    self.reset_evaluation()
                    print("評価をリセットしました")
                
                # FPSの計算
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = time.time()
        
        except KeyboardInterrupt:
            print("\n⚠️ 予測モードがユーザーによって中断されました")
        except Exception as e:
            print(f"\n❌ 予測モードエラー: {e}")
        finally:
            self.cleanup()
    
    def update_display(self, frame: np.ndarray) -> np.ndarray:
        """画面表示の更新"""
        h, w = frame.shape[:2]
        
        # 手のランドマークを描画
        if hasattr(self, 'hand_tracker') and hasattr(self, 'last_results'):
            frame = self.hand_tracker.draw_landmarks(frame, self.last_results)
        
        # キーボードマップを可視化
        if self.keyboard_map:
            frame = self.keyboard_map.visualize(frame)
        
        # 予測結果の表示
        if self.current_prediction:
            self.draw_predictions(frame, self.current_prediction)
        
        # 手の軌跡を描画
        self.draw_trajectory(frame)
        
        # 評価モードの表示
        if self.evaluation_mode:
            self.draw_evaluation_info(frame)
        
        # デバッグ情報の表示
        if self.show_debug:
            self.draw_debug_info(frame)
        
        # 操作説明の表示
        instruction_text = "D: デバッグ | E: 評価モード | R: リセット | ESC: 終了"
        cv2.putText(frame, instruction_text, (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def draw_predictions(self, frame: np.ndarray, predictions: List[Tuple[str, float]]):
        """予測結果の描画"""
        h, w = frame.shape[:2]
        
        # 予測結果の背景
        cv2.rectangle(frame, (w - 300, 20), (w - 20, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 300, 20), (w - 20, 140), (255, 255, 255), 2)
        
        # タイトル
        cv2.putText(frame, "Prediction Results", (w - 280, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top-3の予測結果
        for i, (key, confidence) in enumerate(predictions):
            y_pos = 70 + i * 25
            color = (0, 255, 0) if i == 0 else (255, 255, 0) if i == 1 else (0, 255, 255)
            
            # キーと確信度
            text = f"{key}: {confidence:.1f}%"
            cv2.putText(frame, text, (w - 280, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 確信度バー
            bar_width = int((confidence / 100.0) * 200)
            cv2.rectangle(frame, (w - 280, y_pos + 5), (w - 280 + bar_width, y_pos + 15), color, -1)
            cv2.rectangle(frame, (w - 280, y_pos + 5), (w - 80, y_pos + 15), (255, 255, 255), 1)
        
        # 最も可能性の高いキーをハイライト
        if predictions:
            best_key = predictions[0][0]
            if self.keyboard_map and best_key in self.keyboard_map.key_positions:
                key_info = self.keyboard_map.key_positions[best_key]
                center_x = int(key_info['center_x'])
                center_y = int(key_info['center_y'])
                
                # キーをハイライト
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 3)
                cv2.putText(frame, best_key, (center_x - 10, center_y + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    def draw_trajectory(self, frame: np.ndarray):
        """手の軌跡の描画"""
        if len(self.trajectory_buffer) < 2:
            return
        
        # 過去30フレームの軌跡を描画
        for i in range(len(self.trajectory_buffer) - 1):
            x1, y1 = self.trajectory_buffer[i]
            x2, y2 = self.trajectory_buffer[i + 1]
            
            # 座標をピクセル座標に変換
            px1 = int(x1 * frame.shape[1])
            py1 = int(y1 * frame.shape[0])
            px2 = int(x2 * frame.shape[1])
            py2 = int(y2 * frame.shape[0])
            
            # 軌跡の色（時間に応じて変化）
            alpha = i / len(self.trajectory_buffer)
            color = (int(255 * alpha), int(255 * (1 - alpha)), 255)
            
            cv2.line(frame, (px1, py1), (px2, py2), color, 2)
    
    def draw_evaluation_info(self, frame: np.ndarray):
        """評価情報の描画"""
        h, w = frame.shape[:2]
        
        # 評価情報の背景
        cv2.rectangle(frame, (20, 20), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 20), (400, 120), (255, 255, 255), 2)
        
        # テストテキスト
        cv2.putText(frame, f"Test: {self.test_text}", (30, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 現在の文字をハイライト
        if self.current_char_index < len(self.test_text):
            current_char = self.test_text[self.current_char_index]
            cv2.putText(frame, f"Current: '{current_char}'", (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 精度
        if self.total_predictions > 0:
            accuracy = (self.correct_predictions / self.total_predictions) * 100
            cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (30, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 予測が正解かチェック
        if self.current_prediction and self.evaluation_mode:
            predicted_key = self.current_prediction[0][0]
            if self.current_char_index < len(self.test_text):
                target_char = self.test_text[self.current_char_index]
                if predicted_key.lower() == target_char.lower():
                    self.correct_predictions += 1
                self.total_predictions += 1
    
    def draw_debug_info(self, frame: np.ndarray):
        """デバッグ情報の描画"""
        h, w = frame.shape[:2]
        
        # デバッグ情報の背景
        cv2.rectangle(frame, (20, h - 200), (400, h - 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, h - 200), (400, h - 20), (255, 255, 255), 2)
        
        # タイトル
        cv2.putText(frame, "Debug Info", (30, h - 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # フレームレート
        cv2.putText(frame, f"FPS: {self.fps}", (30, h - 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # バッファの充填状態
        buffer_fill = (len(self.frame_buffer) / 60) * 100
        cv2.putText(frame, f"Buffer: {buffer_fill:.1f}%", (30, h - 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 推論時間
        if hasattr(self, 'inference_time'):
            cv2.putText(frame, f"Inference: {self.inference_time*1000:.1f}ms", (30, h - 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 特徴量の値（最初の5次元のみ表示）
        if len(self.frame_buffer) > 0:
            latest_features = self.frame_buffer[-1]
            if latest_features is not None:
                cv2.putText(frame, f"Features[0-4]: {latest_features[:5][:3]}", (30, h - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def reset_evaluation(self):
        """評価のリセット"""
        self.current_char_index = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.prediction_history = []
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        print("🧹 リソースのクリーンアップ完了")


def run_prediction_mode(model_path: str, keyboard_map_path: str = 'keyboard_map.json'):
    """予測モードを実行"""
    # モデルファイルの存在確認
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが存在しません: {model_path}")
        print("学習を先に実行してください")
        return False
    
    # 予測モードの作成と実行
    prediction_mode = PredictionMode(model_path, keyboard_map_path)
    
    if not prediction_mode.initialize_components():
        print("❌ 初期化に失敗しました")
        return False
    
    # 予測モードの実行
    prediction_mode.run_prediction_mode()
    
    return True


if __name__ == "__main__":
    # テスト用
    model_path = "models/intent_model_latest/best_model.pth"
    
    if os.path.exists(model_path):
        run_prediction_mode(model_path)
    else:
        print(f"モデルファイルが存在しません: {model_path}")
        print("学習を先に実行してください")
