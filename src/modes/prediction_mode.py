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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.camera import Camera
from src.core.hand_tracker import HandTracker
from src.input.keyboard_map import KeyboardMap
from src.processing.coordinate_transformer import WorkAreaTransformer
from src.processing.models.hand_lstm import BasicHandLSTM
from src.processing.feature_extractor import FeatureExtractor


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
        self.transformer = None
        self.model = None
        
        # 予測用のバッファ（学習システムと統一）
        self.trajectory_buffer = deque(maxlen=60)  # 60フレーム分の軌跡データ
        
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
        
        # ラベルマップ（学習時に保存したものを使用）
        self.KEY_CHARS = None
        self.label_map_loaded = False
        
        # 特徴量抽出器（学習システムと統一）
        self.feature_extractor = FeatureExtractor(sequence_length=60, fps=30.0)
        
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
            self.keyboard_map = KeyboardMap(self.keyboard_map_path)
            
            # キーボードマッピングの確認
            if not self.keyboard_map.key_positions:
                print("⚠️ キーボードマッピングが未設定です")
                print("   キーボードマッピングを開始します...")
                if not self.keyboard_map.start_calibration():
                    print("❌ キーボードマッピングに失敗しました")
                    return False
            else:
                print("\n📁 保存済みのキーボード設定ファイル (keyboard_map.json) が見つかりました。")
                print("\n1: 保存した設定を再利用する")
                print("2: 新しくキャリブレーションをやり直す")
                
                while True:
                    try:
                        choice = input("\nどちらにしますか？ (1/2): ").strip()
                        
                        if choice == "1":
                            print("✅ 保存した設定を再利用します。")
                            break
                        elif choice == "2":
                            print("🔄 新しいキャリブレーションを開始します...")
                            if not self.keyboard_map.start_calibration():
                                print("❌ キーボードマッピングに失敗しました")
                                return False
                            break
                        else:
                            print("❌ 無効な選択です。1 または 2 を入力してください。")
                            
                    except KeyboardInterrupt:
                        print("\n❌ プログラムを終了します。")
                        return False
                    except EOFError:
                        print("\n❌ プログラムを終了します。")
                        return False
            
            # 座標変換器の初期化
            self.transformer = WorkAreaTransformer(self.keyboard_map_path)
            # キーボードマップから作業領域の4隅を取得
            keyboard_corners = self.keyboard_map.get_work_area_corners()
            if keyboard_corners is not None:
                self.transformer.set_work_area_corners(keyboard_corners)
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
            
            # ラベルマップの読み込み（チェックポイント or JSON）
            self.KEY_CHARS = checkpoint.get('label_map')
            if self.KEY_CHARS is None:
                # フォールバック: 同ディレクトリのlabel_map.json
                label_map_path = os.path.join(os.path.dirname(self.model_path), 'label_map.json')
                if os.path.exists(label_map_path):
                    with open(label_map_path, 'r', encoding='utf-8') as f:
                        self.KEY_CHARS = json.load(f).get('labels')
            if self.KEY_CHARS is None:
                # 最後のフォールバック（従来定義）
                self.KEY_CHARS = (
                    'a','b','c','d','e','f','g','h','i','j','k','l','m',
                    'n','o','p','q','r','s','t','u','v','w','x','y','z',
                    '0','1','2','3','4','5','6','7','8','9',' '
                )
            else:
                self.label_map_loaded = True

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
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """フレームを処理して軌跡データを生成"""
        try:
            # 手追跡
            results = self.hand_tracker.detect_hands(frame)
            if not results.multi_hand_landmarks:
                return None
            
            # 最初の手のランドマークを取得
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 人差し指の座標を取得
            index_finger = hand_landmarks.landmark[8]  # 人差し指先端
            
            # ピクセル座標を作業領域空間に変換
            wa_coords = self.transformer.pixel_to_work_area(
                index_finger.x, index_finger.y
            )
            
            if wa_coords is None:
                return None
            
            wa_x, wa_y = wa_coords
            
            # 最近傍3キーへの相対座標を取得
            nearest_keys = self.transformer.get_nearest_keys_with_relative_coords(
                wa_x, wa_y, top_k=3
            )
            
            # 軌跡データフレームを構築（学習システムと統一）
            frame_data = {
                'work_area_coords': {
                    'index_finger': {'x': wa_x, 'y': wa_y}
                },
                'nearest_keys_relative': [
                    {
                        'key': k.key,
                        'relative_x': k.relative_x,
                        'relative_y': k.relative_y,
                        'distance': np.sqrt(k.relative_x**2 + k.relative_y**2)
                    } for k in nearest_keys[:3]
                ]
            }
            
            return frame_data
            
        except Exception as e:
            print(f"⚠️ フレーム処理エラー: {e}")
            return None
    
    def predict_intent(self) -> Optional[List[Tuple[str, float]]]:
        """入力意図を予測"""
        try:
            if len(self.trajectory_buffer) < 60:
                return None
            
            # 軌跡データを特徴量に変換（学習システムと統一）
            trajectory_data = list(self.trajectory_buffer)
            features_np = self.feature_extractor.extract_from_trajectory(trajectory_data)
            
            # テンソルに変換
            features_tensor = torch.FloatTensor(features_np).unsqueeze(0).to(self.device)
            
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
        print("   操作方法:")
        print("   - 手をカメラに映してキーボード入力の意図を予測")
        print("   - ESC: 終了")
        print("   - 'd': デバッグ情報の表示/非表示")
        print("   - 'e': 評価モードの切り替え")
        print("   - 'r': 評価のリセット")
        
        try:
            while True:
                # フレーム取得
                frame = self.camera.read_frame()
                if frame is None:
                    continue
                
                # フレーム処理
                frame_data = self.process_frame(frame)
                if frame_data is not None:
                    self.trajectory_buffer.append(frame_data)
                
                # 予測実行
                if len(self.trajectory_buffer) >= 60:
                    predictions = self.predict_intent()
                    if predictions:
                        self.current_prediction = predictions
                        self.prediction_history.append(predictions)
                
                # 画面描画
                self.draw_frame(frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                elif key == ord('e'):
                    self.evaluation_mode = not self.evaluation_mode
                    if self.evaluation_mode:
                        print("📊 評価モードを有効にしました")
                    else:
                        print("📊 評価モードを無効にしました")
                elif key == ord('r'):
                    self.reset_evaluation()
                    print("🔄 評価をリセットしました")
                
                # FPS計算
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = time.time()
        
        except KeyboardInterrupt:
            print("\n⚠️ ユーザーによって中断されました")
        
        finally:
            self.cleanup()
    
    def draw_frame(self, frame: np.ndarray):
        """フレームの描画"""
        # キーボードマップの描画
        if self.keyboard_map:
            self.keyboard_map.visualize(frame)
        
        # 手追跡の描画
        if self.hand_tracker:
            # 手の検出結果を取得して描画
            results = self.hand_tracker.detect_hands(frame)
            if results.multi_hand_landmarks:
                self.hand_tracker.draw_landmarks(frame, results)
        
        # 予測結果の描画
        if self.current_prediction:
            self.draw_prediction(frame)
        
        # 評価情報の描画
        if self.evaluation_mode:
            self.draw_evaluation_info(frame)
        
        # デバッグ情報の描画
        if self.show_debug:
            self.draw_debug_info(frame)
        
        # 基本情報の描画
        self.draw_basic_info(frame)
        
        # 画面表示
        cv2.imshow('Keyboard Intent Prediction', frame)
    
    def draw_prediction(self, frame: np.ndarray):
        """予測結果の描画"""
        if not self.current_prediction:
            return
        
        # 予測結果の背景
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # タイトル
        cv2.putText(frame, "Prediction Results", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Top-3予測結果
        for i, (key, confidence) in enumerate(self.current_prediction[:3]):
            color = (0, 255, 0) if i == 0 else (255, 255, 0) if i == 1 else (0, 255, 255)
            cv2.putText(frame, f"{i+1}. {key}: {confidence:.1f}%", (20, 55 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_evaluation_info(self, frame: np.ndarray):
        """評価情報の描画"""
        # 評価情報の背景
        cv2.rectangle(frame, (10, 130), (300, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 130), (300, 180), (255, 255, 255), 2)
        
        # 現在の文字
        if self.current_char_index < len(self.test_text):
            current_char = self.test_text[self.current_char_index]
            cv2.putText(frame, f"Current: '{current_char}'", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 精度
        if self.total_predictions > 0:
            accuracy = (self.correct_predictions / self.total_predictions) * 100
            cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (20, 170), 
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
        buffer_fill = (len(self.trajectory_buffer) / 60) * 100
        cv2.putText(frame, f"Buffer: {buffer_fill:.1f}%", (30, h - 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 推論時間
        if hasattr(self, 'inference_time'):
            cv2.putText(frame, f"Inference: {self.inference_time*1000:.1f}ms", (30, h - 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def draw_basic_info(self, frame: np.ndarray):
        """基本情報の描画"""
        h, w = frame.shape[:2]
        
        # 基本情報の背景
        cv2.rectangle(frame, (w - 200, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 200, 10), (w - 10, 80), (255, 255, 255), 2)
        
        # タイトル
        cv2.putText(frame, "Controls", (w - 190, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 操作方法
        cv2.putText(frame, "ESC: Quit", (w - 190, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "d: Debug", (w - 190, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
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
    # 最新のモデルを自動検出
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ modelsディレクトリが存在しません")
        print("学習を先に実行してください")
        exit(1)
    
    # モデルディレクトリを検索
    model_dirs = [d for d in os.listdir(models_dir) 
                  if d.startswith("intent_model_") and os.path.isdir(os.path.join(models_dir, d))]
    
    if not model_dirs:
        print("❌ 学習済みモデルが見つかりません")
        print("学習を先に実行してください")
        exit(1)
    
    # 最新のモデルディレクトリを選択
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(models_dir, latest_model_dir, "best_model.pth")
    
    print(f"🔍 使用するモデル: {model_path}")
    
    if os.path.exists(model_path):
        run_prediction_mode(model_path)
    else:
        print(f"❌ モデルファイルが存在しません: {model_path}")
        print("学習を先に実行してください")
