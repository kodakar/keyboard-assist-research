# src/modes/evaluation_mode.py
"""
リアルタイム評価実験モード
被験者によるシステムの実用性能を測定するための評価モード
"""

import os
import cv2
import numpy as np
import torch
import time
import json
from collections import deque
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# 既存のモジュールをインポート
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.camera import Camera
from src.core.hand_tracker import HandTracker
from src.input.keyboard_map import KeyboardMap
from src.input.keyboard_tracker import KeyboardTracker
from src.processing.coordinate_transformer import WorkAreaTransformer
from src.processing.models.hand_lstm import BasicHandLSTM
from src.processing.feature_extractor import FeatureExtractor


class EvaluationMode:
    """
    リアルタイム評価実験用モード
    被験者に実際にシステムを使ってもらい、精度を測定
    """
    
    def __init__(self, model_path: str, keyboard_map_path: str = 'keyboard_map.json'):
        """
        初期化
        - prediction_mode.py と同じコンポーネントを使用
        - 評価ログを記録する機能を追加
        """
        self.model_path = model_path
        self.keyboard_map_path = keyboard_map_path
        
        # コンポーネントの初期化
        self.camera = None
        self.hand_tracker = None
        self.keyboard_tracker = None
        self.keyboard_map = None
        self.transformer = None
        self.model = None
        
        # 予測用のバッファ（学習システムと統一）
        self.trajectory_buffer = deque(maxlen=60)  # 60フレーム分の軌跡データ
        
        # 予測結果
        self.current_prediction = None
        
        # ラベルマップ（学習時に保存したものを使用）
        self.KEY_CHARS = None
        self.label_map_loaded = False
        
        # 特徴量抽出器（学習システムと統一）
        self.feature_extractor = FeatureExtractor(sequence_length=60, fps=30.0)
        
        # 評価ログ
        self.evaluation_log = []
        self.current_task = None
        self.current_inputs = []
        self.start_time = None
        self.last_input_time = None
        
        print(f"🎯 評価モード初期化完了")
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
            
            # キーボードトラッカーの初期化
            self.keyboard_tracker = KeyboardTracker()
            self.keyboard_tracker.start()
            print("✅ キーボードトラッカー初期化完了")
            
            # キーボードマップの初期化
            self.keyboard_map = KeyboardMap(self.keyboard_map_path)
            
            # キーボードマッピングの確認
            if not self.keyboard_map.key_positions:
                print("⚠️ キーボードマッピングが未設定です")
                print("   キーボードマッピングを開始します...")
                if not self.keyboard_map.start_calibration(existing_camera=self.camera):
                    print("❌ キーボードマッピングに失敗しました")
                    return False
            else:
                print("✅ 保存済みのキーボード設定を使用します")
            
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
            input_size = model_config.get('input_size', 18)
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
            
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Top-3の予測結果を取得
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            
            predictions = []
            for i in range(3):
                key = self.KEY_CHARS[top3_indices[0][i].item()]
                confidence = top3_probs[0][i].item() * 100
                predictions.append((key, confidence))
            
            return predictions
            
        except Exception as e:
            print(f"⚠️ 予測エラー: {e}")
            return None
    
    def run_evaluation_session(self, participant_id: str, target_texts: List[str]):
        """
        評価セッションの実行
        
        Args:
            participant_id: 被験者ID（例: "P001", "P002"）
            target_texts: 入力してもらうテキストのリスト
                例: ["hello world", "the quick brown fox jumps over the lazy dog"]
        """
        print(f"🚀 評価セッション開始")
        print(f"   被験者ID: {participant_id}")
        print(f"   タスク数: {len(target_texts)}")
        print("\n操作方法:")
        print("   - 手をカメラに映してキーボード入力の意図を予測")
        print("   - 画面に表示された文字をキーボードで入力")
        print("   - ESC: セッション終了")
        
        try:
            # コンポーネントの初期化
            if not self.initialize_components():
                print("❌ 初期化に失敗しました")
                return False
            
            # 各タスクを実行
            for task_idx, target_text in enumerate(target_texts):
                print(f"\n📝 タスク {task_idx + 1}/{len(target_texts)}")
                print(f"   目標テキスト: \"{target_text}\"")
                print("   準備ができたらスペースキーを押してください...")
                
                # スペースキー待機
                while True:
                    frame = self.camera.read_frame()
                    if frame is None:
                        continue
                    
                    self.draw_waiting_screen(frame, task_idx + 1, len(target_texts), target_text)
                    cv2.imshow('Evaluation Mode', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        return False
                    elif key == 32:  # スペース
                        break
                
                # タスク開始
                success = self.run_single_task(task_idx, target_text)
                if not success:
                    print("⚠️ タスクが中断されました")
                    return False
            
            # 評価完了
            print("\n✅ 全てのタスクが完了しました")
            self.show_summary()
            self.save_results(participant_id)
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ 評価セッションが中断されました")
            return False
        finally:
            self.cleanup()
    
    def run_single_task(self, task_idx: int, target_text: str) -> bool:
        """単一タスクの実行"""
        self.current_task = {
            'task_idx': task_idx,
            'target_text': target_text,
            'inputs': []
        }
        self.current_inputs = []
        
        # 文字インデックス
        char_idx = 0
        self.start_time = time.time()
        self.last_input_time = self.start_time
        
        try:
            while char_idx < len(target_text):
                target_char = target_text[char_idx]
                
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
                
                # キー入力チェック
                actual_input = self.keyboard_tracker.get_key_event()
                if actual_input is not None:
                    # 入力時間の計算
                    current_time = time.time()
                    input_time = current_time - self.last_input_time
                    self.last_input_time = current_time
                    
                    # 正解/不正解の判定
                    is_correct = (actual_input.lower() == target_char.lower())
                    
                    # 予測結果の記録
                    predicted_top3 = []
                    predicted_probs = []
                    if self.current_prediction:
                        for key, prob in self.current_prediction[:3]:
                            predicted_top3.append(key)
                            predicted_probs.append(prob)
                    
                    # 入力ログを記録
                    input_log = {
                        'target_char': target_char,
                        'predicted_top3': predicted_top3,
                        'predicted_probs': predicted_probs,
                        'actual_input': actual_input,
                        'is_correct': is_correct,
                        'input_time': input_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.current_inputs.append(input_log)
                    print(f"   {target_char} -> {actual_input} ({'✓' if is_correct else '✗'}) [{input_time:.2f}s]")
                    
                    char_idx += 1
                
                # 画面描画
                self.draw_task_screen(frame, target_text, char_idx, len(target_text))
                cv2.imshow('Evaluation Mode', frame)
                
                # ESCキーで終了
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    return False
            
            # タスク完了
            self.current_task['inputs'] = self.current_inputs
            self.evaluation_log.append(self.current_task)
            
            return True
            
        except Exception as e:
            print(f"❌ タスク実行エラー: {e}")
            return False
    
    def draw_waiting_screen(self, frame: np.ndarray, task_num: int, total_tasks: int, target_text: str):
        """待機画面の描画"""
        h, w = frame.shape[:2]
        
        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # タイトル
        cv2.putText(frame, "EVALUATION MODE", (w//2 - 150, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # タスク情報
        cv2.putText(frame, f"Task {task_num}/{total_tasks}", (w//2 - 80, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 目標テキスト
        cv2.putText(frame, "Target Text:", (w//2 - 100, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'"{target_text}"', (w//2 - len(target_text)*8, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 指示
        cv2.putText(frame, "Press SPACE to start", (w//2 - 120, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press ESC to exit", (w//2 - 100, 330), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_task_screen(self, frame: np.ndarray, target_text: str, current_char: int, total_chars: int):
        """タスク実行画面の描画"""
        # キーボードマップの描画
        if self.keyboard_map:
            self.keyboard_map.visualize(frame)
        
        # 手追跡の描画
        if self.hand_tracker:
            results = self.hand_tracker.detect_hands(frame)
            if results.multi_hand_landmarks:
                self.hand_tracker.draw_landmarks(frame, results)
        
        h, w = frame.shape[:2]
        
        # 情報パネル
        panel_width = 400
        panel_height = 200
        x1 = w - panel_width - 20
        y1 = 20
        x2 = w - 20
        y2 = y1 + panel_height
        
        # 背景
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # 目標テキスト
        cv2.putText(frame, f"Target: \"{target_text}\"", (x1 + 10, y1 + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 現在の文字
        if current_char < len(target_text):
            target_char = target_text[current_char]
            cv2.putText(frame, f"Next: '{target_char}' ({current_char + 1}/{total_chars})", 
                       (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Task Complete!", (x1 + 10, y1 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 進捗バー
        progress = current_char / total_chars
        bar_width = panel_width - 20
        bar_height = 10
        bar_x = x1 + 10
        bar_y = y1 + 70
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        
        # 進捗テキスト
        cv2.putText(frame, f"Progress: {progress*100:.0f}%", (x1 + 10, y1 + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 予測結果
        if self.current_prediction:
            cv2.putText(frame, "Predictions:", (x1 + 10, y1 + 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, (key, prob) in enumerate(self.current_prediction[:3]):
                color = (0, 255, 0) if i == 0 else (255, 255, 0) if i == 1 else (0, 255, 255)
                cv2.putText(frame, f"{i+1}. {key} ({prob:.1f}%)", (x1 + 10, y1 + 145 + i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            cv2.putText(frame, "Loading predictions...", (x1 + 10, y1 + 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def calculate_metrics(self) -> Dict:
        """
        評価指標を計算
        
        Returns:
            {
                'top1_accuracy': float,      # Top-1正解率
                'top3_accuracy': float,      # Top-3正解率
                'avg_input_time': float,     # 平均入力時間（秒/文字）
                'wpm': float,                # Words Per Minute
                'error_rate': float,         # エラー率
                'total_inputs': int,         # 総入力数
                'correct_inputs': int        # 正解数
            }
        """
        if not self.evaluation_log:
            return {
                'top1_accuracy': 0.0,
                'top3_accuracy': 0.0,
                'avg_input_time': 0.0,
                'wpm': 0.0,
                'error_rate': 100.0,
                'total_inputs': 0,
                'correct_inputs': 0
            }
        
        total_inputs = 0
        correct_inputs = 0
        top3_correct = 0
        total_input_time = 0.0
        
        for task in self.evaluation_log:
            for input_log in task['inputs']:
                total_inputs += 1
                total_input_time += input_log['input_time']
                
                if input_log['is_correct']:
                    correct_inputs += 1
                
                # Top-3正解率の計算
                if input_log['predicted_top3'] and len(input_log['predicted_top3']) > 0:
                    target_char = input_log['target_char'].lower()
                    if target_char in [pred.lower() for pred in input_log['predicted_top3']]:
                        top3_correct += 1
        
        # 指標の計算
        top1_accuracy = (correct_inputs / total_inputs * 100) if total_inputs > 0 else 0.0
        top3_accuracy = (top3_correct / total_inputs * 100) if total_inputs > 0 else 0.0
        avg_input_time = (total_input_time / total_inputs) if total_inputs > 0 else 0.0
        
        # WPMの計算（英語では平均5文字 = 1単語として計算）
        wpm = (60 / avg_input_time / 5) if avg_input_time > 0 else 0.0
        
        error_rate = 100.0 - top1_accuracy
        
        return {
            'top1_accuracy': round(top1_accuracy, 2),
            'top3_accuracy': round(top3_accuracy, 2),
            'avg_input_time': round(avg_input_time, 2),
            'wpm': round(wpm, 2),
            'error_rate': round(error_rate, 2),
            'total_inputs': total_inputs,
            'correct_inputs': correct_inputs
        }
    
    def save_results(self, participant_id: str):
        """
        結果をJSON形式で保存
        
        保存先: evaluation_results/{participant_id}/evaluation_YYYYMMDD_HHMMSS.json
        """
        try:
            # ディレクトリの作成
            results_dir = f"evaluation_results/{participant_id}"
            os.makedirs(results_dir, exist_ok=True)
            
            # ファイル名の生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # 評価指標の計算
            metrics = self.calculate_metrics()
            
            # 結果データの構築
            results = {
                'participant_id': participant_id,
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'evaluation_log': self.evaluation_log,
                'metrics': metrics
            }
            
            # JSONファイルとして保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 評価結果を保存しました: {filepath}")
            
        except Exception as e:
            print(f"❌ 結果保存エラー: {e}")
    
    def show_summary(self):
        """
        評価結果のサマリーをコンソールに表示
        """
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("📊 評価結果サマリー")
        print("="*60)
        print(f"Top-1精度:     {metrics['top1_accuracy']:.1f}%")
        print(f"Top-3精度:     {metrics['top3_accuracy']:.1f}%")
        print(f"平均入力時間:   {metrics['avg_input_time']:.2f}秒/文字")
        print(f"WPM:          {metrics['wpm']:.1f}")
        print(f"エラー率:      {metrics['error_rate']:.1f}%")
        print(f"総入力数:      {metrics['total_inputs']}")
        print(f"正解数:        {metrics['correct_inputs']}")
        print("="*60)
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        if self.camera:
            self.camera.release()
        
        if self.keyboard_tracker:
            self.keyboard_tracker.stop()
        
        cv2.destroyAllWindows()
        print("🧹 リソースのクリーンアップ完了")


def run_evaluation_mode(model_path: str, participant_id: str, target_texts: List[str], 
                       keyboard_map_path: str = 'keyboard_map.json'):
    """評価モードを実行"""
    # モデルファイルの存在確認
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが存在しません: {model_path}")
        print("学習を先に実行してください")
        return False
    
    # 評価モードの作成と実行
    evaluation_mode = EvaluationMode(model_path, keyboard_map_path)
    
    # 評価セッションの実行
    success = evaluation_mode.run_evaluation_session(participant_id, target_texts)
    
    return success


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='リアルタイム評価モード')
    parser.add_argument('--model', type=str, required=True,
                        help='使用するモデルパス（.pthファイル）')
    parser.add_argument('--participant', type=str, required=True,
                        help='被験者ID（例: P001）')
    parser.add_argument('--texts', type=str, nargs='+', required=True,
                        help='入力してもらうテキストのリスト')
    parser.add_argument('--map', type=str, default='keyboard_map.json',
                        help='キーボードマップJSONのパス (default: keyboard_map.json)')
    
    args = parser.parse_args()
    
    # 評価モードの実行
    success = run_evaluation_mode(
        model_path=args.model,
        participant_id=args.participant,
        target_texts=args.texts,
        keyboard_map_path=args.map
    )
    
    if success:
        print("✅ 評価が正常に完了しました")
    else:
        print("❌ 評価が中断されました")
