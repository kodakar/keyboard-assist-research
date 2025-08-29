# src/processing/enhanced_data_collector.py
"""
拡張データ収集システム
手の軌跡データの時系列保存と学習用データセット構築
"""

import json
import os
import numpy as np
from datetime import datetime
from collections import deque
import cv2
from typing import List, Dict, Optional, Tuple
from src.processing.coordinate_transformer import WorkAreaTransformer

class EnhancedDataCollector:
    def __init__(self, 
                 trajectory_buffer_size: int = 30,
                 data_dir: str = "data",
                 user_id: str = "user_001"):
        """
        拡張データ収集システムの初期化
        
        Args:
            trajectory_buffer_size: 軌跡バッファのサイズ（フレーム数）
            data_dir: データ保存ディレクトリ
            user_id: ユーザーID
        """
        self.trajectory_buffer_size = trajectory_buffer_size
        self.data_dir = os.path.abspath(data_dir)
        self.user_id = user_id
        
        # 座標変換クラスの初期化
        self.transformer = WorkAreaTransformer()
        
        # 手の軌跡をバッファ（時系列データ）
        self.trajectory_buffer = deque(maxlen=trajectory_buffer_size)
    
        # 前フレームの座標（速度計算用）
        self.previous_coords = None
        
        # 現在の入力コンテキスト
        self.current_context = ""
        self.target_text = ""
        
        # データ収集の状態
        self.is_collecting = False
        self.collection_start_time = None
        
        # データディレクトリの作成
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "samples"), exist_ok=True)
        
        # モデル保存用ディレクトリの作成
        models_dir = os.path.join(self.data_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(models_dir, "user_001"), exist_ok=True)
        os.makedirs(os.path.join(models_dir, "shared"), exist_ok=True)
        
        # 統計情報
        self.stats = {
            "total_samples": 0,
            "total_trajectories": 0,
            "session_start": datetime.now().isoformat()
        }
    
    def set_work_area_corners(self, corners: np.ndarray):
        """
        作業領域の4隅の座標を設定
        
        Args:
            corners: 4隅の座標 (左上, 右上, 右下, 左下) の配列
        """
        self.transformer.set_work_area_corners(corners)
    
    def start_collection_session(self, target_text: str = ""):
        """
        データ収集セッションを開始
        
        Args:
            target_text: 目標とする入力テキスト
        """
        self.target_text = target_text
        self.current_context = ""
        self.is_collecting = True
        self.collection_start_time = datetime.now()
        self.trajectory_buffer.clear()
        
        print(f"🚀 データ収集セッション開始")
        print(f"   目標テキスト: {target_text}")
        print(f"   ユーザーID: {self.user_id}")
    
    def stop_collection_session(self):
        """データ収集セッションを停止"""
        self.is_collecting = False
        session_duration = datetime.now() - self.collection_start_time
        
        print(f"⏹️ データ収集セッション終了")
        print(f"   セッション時間: {session_duration}")
        print(f"   収集サンプル数: {self.stats['total_samples']}")
        print(f"   収集軌跡数: {self.stats['total_trajectories']}")
        
        # セッション統計を保存
        self._save_session_stats()
    
    def add_hand_position(self, hand_landmarks, frame_timestamp: Optional[float] = None):
        """
        手の位置を軌跡バッファに追加（相対座標系）
        
        Args:
            hand_landmarks: MediaPipeの手のランドマーク
            frame_timestamp: フレームのタイムスタンプ
        """
        if not self.is_collecting:
            return
        
        try:
            # 重要なランドマークの座標を取得（0-1正規化）
            index_finger = hand_landmarks.landmark[8]  # 人差し指の先端
            middle_finger = hand_landmarks.landmark[12]  # 中指の先端
            wrist = hand_landmarks.landmark[0]  # 手首
            
            # 作業領域座標に変換
            wa_coords = {}
            for name, landmark in [("index_finger", index_finger), 
                                  ("middle_finger", middle_finger), 
                                  ("wrist", wrist)]:
                # ピクセル座標を作業領域座標に変換
                wa_coord = self.transformer.pixel_to_work_area(
                    landmark.x, landmark.y
                )
                if wa_coord:
                    wa_coords[name] = {"x": float(wa_coord[0]), "y": float(wa_coord[1])}
                else:
                    # 変換失敗時は元の座標を使用
                    wa_coords[name] = {"x": float(landmark.x), "y": float(landmark.y)}
            
            # 人差し指の位置から最近傍キーと相対座標を取得
            nearest_keys_info = []
            if "index_finger" in wa_coords:
                index_x = wa_coords["index_finger"]["x"]
                index_y = wa_coords["index_finger"]["y"]
                
                nearest_keys = self.transformer.get_nearest_keys_with_relative_coords(
                    index_x, index_y, top_k=3
                )
                
                for key_info in nearest_keys:
                    # 速度計算
                    approach_velocity = self._calculate_approach_velocity(
                        index_x, index_y, key_info.key, frame_timestamp
                    )
                    
                    nearest_keys_info.append({
                        "key": key_info.key,
                        "relative_x": float(key_info.relative_x),
                        "relative_y": float(key_info.relative_y),
                        "distance": float(np.sqrt(key_info.relative_x**2 + key_info.relative_y**2)),
                        "approach_velocity": float(approach_velocity) if approach_velocity is not None else 0.0
                    })
            
            # 軌跡データを作成
            trajectory_point = {
                "timestamp": float(frame_timestamp or datetime.now().timestamp()),
                "frame_index": len(self.trajectory_buffer),
                "work_area_coords": wa_coords,  # 名前変更
                "nearest_keys_relative": nearest_keys_info,
                "data_version": "2.0"  # バージョン追加
            }
            
            # 前フレームの座標を更新（速度計算用）
            self.previous_coords = {
                "index_finger": (float(index_finger.x), float(index_finger.y)),
                "timestamp": float(frame_timestamp or datetime.now().timestamp())
            }
            
            # 軌跡データをバッファに追加
            self.trajectory_buffer.append(trajectory_point)
            
        except Exception as e:
            print(f"⚠️ 手の位置追加エラー: {e}")
            # エラー時は元の形式で保存
            landmarks_array = []
            for landmark in hand_landmarks.landmark:
                landmarks_array.extend([landmark.x, landmark.y, landmark.z])
            
            trajectory_point = {
                'timestamp': frame_timestamp or datetime.now().timestamp(),
                'landmarks': landmarks_array,
                'frame_index': len(self.trajectory_buffer)
            }
            self.trajectory_buffer.append(trajectory_point)
    
    def _calculate_approach_velocity(self, current_x: float, current_y: float, 
                                   target_key: str, current_timestamp: float) -> float:
        """
        指定キーへの接近速度を計算（キーサイズ/秒）
        
        Args:
            current_x: 現在のX座標（キーボード空間）
            current_y: 現在のY座標（キーボード空間）
            target_key: 目標キー
            current_timestamp: 現在のタイムスタンプ
            
        Returns:
            接近速度（キーサイズ/秒）
        """
        if self.previous_coords is None:
            return 0.0
        
        try:
            # 前フレームの座標を取得
            prev_x, prev_y = self.previous_coords["index_finger"]
            prev_timestamp = self.previous_coords["timestamp"]
            
            # 時間差を計算
            time_diff = current_timestamp - prev_timestamp
            if time_diff <= 0:
                return 0.0
            
            # 前フレームの座標を作業領域に変換
            prev_kb_coord = self.transformer.pixel_to_work_area(prev_x, prev_y)
            if prev_kb_coord is None:
                return 0.0
            
            prev_kb_x, prev_kb_y = prev_kb_coord
            
            # 距離の変化を計算
            distance_diff = np.sqrt((current_x - prev_kb_x)**2 + (current_y - prev_kb_y)**2)
            
            # 速度を計算（キーサイズ/秒）
            velocity = distance_diff / time_diff
            
            return velocity
            
        except Exception as e:
            print(f"⚠️ 速度計算エラー: {e}")
            return 0.0
    
    def add_key_sample(self, intended_key: str, actual_key: str,
                      hand_landmarks, target_text: str = ""):
        """
        キー入力サンプルを記録
        
        Args:
            intended_key: 意図したキー
            actual_key: 実際に押されたキー
            hand_landmarks: キー押下時の手のランドマーク
            target_text: 目標テキスト
        """
        if not self.is_collecting:
            return
        
        # 現在のコンテキストを更新
        if intended_key == actual_key:
            self.current_context += intended_key
        else:
            # 誤入力の場合は、意図したキーでコンテキストを更新
            self.current_context += intended_key
        
        # 軌跡データを取得（バッファが満たされている場合）
        trajectory_data = None
        if len(self.trajectory_buffer) >= self.trajectory_buffer_size:
            trajectory_data = list(self.trajectory_buffer)
        
        # サンプルデータを作成（相対座標系）
        sample = {
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id,
            'target_text': target_text or self.target_text,
            'intended_key': intended_key,
            'actual_key': actual_key,
            'current_context': self.current_context,
            'hand_landmarks': self._landmarks_to_array(hand_landmarks),
            'landmarks_format': 'array_63d',  # 21点 × 3座標（後方互換性）
            'trajectory_data': trajectory_data,
            'trajectory_length': len(self.trajectory_buffer),
            'session_duration': (datetime.now() - self.collection_start_time).total_seconds(),
            'coordinate_system': 'work_area_v2'  # 座標系の指定
        }
        
        # サンプルを保存
        self._save_sample(sample)
        
        # 軌跡データを保存（別ファイル）
        if trajectory_data:
            self._save_trajectory(trajectory_data, intended_key)
        
        # 統計を更新
        self.stats['total_samples'] += 1
        if trajectory_data:
            self.stats['total_trajectories'] += 1
        
        print(f"📝 サンプル記録: {intended_key} -> {actual_key} "
              f"(軌跡: {len(self.trajectory_buffer)}フレーム)")
    
    def set_screen_size(self, width: int, height: int):
        """
        画面サイズを設定（座標変換に使用）
        
        Args:
            width: 画面幅（ピクセル）
            height: 画面高さ（ピクセル）
        """
        self.transformer.set_screen_size(width, height)
        print(f"✅ 画面サイズを設定しました: {width}x{height}")
    
    def set_work_area_corners(self, corners: np.ndarray):
        """
        作業領域の4隅の座標を設定
        
        Args:
            corners: 4隅の座標 (左上, 右上, 右下, 左下) の配列
                    各座標は (x, y) の形式で、0-1の正規化座標
        """
        self.transformer.set_work_area_corners(corners)
        print(f"✅ 作業領域の4隅を設定しました")
    
    def _landmarks_to_array(self, hand_landmarks) -> List[float]:
        """手のランドマークを配列に変換"""
        landmarks_array = []
        for landmark in hand_landmarks.landmark:
            landmarks_array.extend([landmark.x, landmark.y, landmark.z])
        return landmarks_array
    
    def _save_sample(self, sample: Dict):
        """サンプルデータをJSONファイルとして保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_{timestamp}_{sample['intended_key']}_{self.stats['total_samples']:04d}.json"
        filepath = os.path.join(self.data_dir, "samples", filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ サンプル保存エラー: {e}")
    
    def _save_trajectory(self, trajectory_data: List[Dict], key: str):
        """軌跡データをJSONファイルとして保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{timestamp}_{key}_{self.stats['total_trajectories']:04d}.json"
        filepath = os.path.join(self.data_dir, "trajectories", filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 軌跡保存エラー: {e}")
    
    def _save_session_stats(self):
        """セッション統計を保存"""
        stats_file = os.path.join(self.data_dir, f"session_stats_{self.user_id}.json")
        
        session_stats = {
            'user_id': self.user_id,
            'session_start': self.collection_start_time.isoformat(),
            'session_end': datetime.now().isoformat(),
            'total_samples': self.stats['total_samples'],
            'total_trajectories': self.stats['total_trajectories'],
            'target_text': self.target_text,
            'final_context': self.current_context
        }
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(session_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 統計保存エラー: {e}")
    
    def get_training_dataset_info(self) -> Dict:
        """学習用データセットの情報を取得"""
        samples_dir = os.path.join(self.data_dir, "samples")
        trajectories_dir = os.path.join(self.data_dir, "trajectories")
        
        sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.json')]
        trajectory_files = [f for f in os.listdir(trajectories_dir) if f.endswith('.json')]
        
        return {
            'total_samples': len(sample_files),
            'total_trajectories': len(trajectory_files),
            'samples_directory': samples_dir,
            'trajectories_directory': trajectories_dir,
            'user_id': self.user_id,
            'last_updated': datetime.now().isoformat()
        }
    
    def export_training_dataset(self, output_file: str = None) -> str:
        """学習用データセットを統合してエクスポート"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.data_dir, f"training_dataset_{self.user_id}_{timestamp}.json")
        
        samples_dir = os.path.join(self.data_dir, "samples")
        sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.json')]
        
        dataset = []
        for sample_file in sample_files:
            try:
                with open(os.path.join(samples_dir, sample_file), 'r', encoding='utf-8') as f:
                    sample = json.load(f)
                    dataset.append(sample)
            except Exception as e:
                print(f"⚠️ サンプル読み込みエラー {sample_file}: {e}")
        
        # データセットを保存
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"✅ 学習データセットをエクスポート: {output_file}")
            print(f"   サンプル数: {len(dataset)}")
            return output_file
        except Exception as e:
            print(f"⚠️ データセットエクスポートエラー: {e}")
            return ""
    
    def visualize_trajectory(self, frame, trajectory_data: List[Dict] = None):
        """
        軌跡データを画像上に可視化
        
        Args:
            frame: 表示用フレーム
            trajectory_data: 軌跡データ（Noneの場合は現在のバッファを使用）
        """
        if trajectory_data is None:
            trajectory_data = list(self.trajectory_buffer)
        
        if not trajectory_data:
            return frame
        
        h, w = frame.shape[:2]
        
        # 軌跡を線で描画
        points = []
        for i, point in enumerate(trajectory_data):
            if 'landmarks' in point and len(point['landmarks']) >= 63:
                try:
                    # 手の中心位置（ランドマーク9番、中指の付け根）を使用
                    # MediaPipe Hands: 21個のランドマーク × 3座標(x,y,z) = 63次元
                    # ランドマーク9番: 中指の付け根 → landmarks[27:30] (x, y, z)
                    # 正しいインデックス: 9番目のランドマーク = landmarks[27:30]
                    x = int(point['landmarks'][27] * w)  # x座標 (27 = 9*3)
                    y = int(point['landmarks'][28] * h)  # y座標 (28 = 9*3+1)
                    points.append((x, y))
                except (IndexError, ValueError) as e:
                    print(f"⚠️ 軌跡データの座標変換エラー: {e}")
                    continue
        
        # 軌跡を描画
        if len(points) > 1:
            for i in range(1, len(points)):
                # 色をグラデーションで変化（古い点は青、新しい点は赤）
                color_ratio = i / len(points)
                color = (
                    int(255 * color_ratio),  # B
                    int(255 * (1 - color_ratio)),  # G
                    int(255)  # R
                )
                cv2.line(frame, points[i-1], points[i], color, 2)
            
            # 最新の点を強調表示
            if points:
                cv2.circle(frame, points[-1], 8, (0, 255, 255), -1)
        
        return frame
    
    def cleanup_memory(self):
        """メモリクリーンアップを実行"""
        # 軌跡バッファが大きくなりすぎた場合のクリーンアップ
        if len(self.trajectory_buffer) > self.trajectory_buffer_size * 2:
            print(f"🧹 メモリクリーンアップ実行: バッファサイズ {len(self.trajectory_buffer)} -> {self.trajectory_buffer_size}")
            self.trajectory_buffer.clear()
        
        # 統計情報のリセット（長時間セッション対策）
        if self.collection_start_time:
            session_duration = (datetime.now() - self.collection_start_time).total_seconds()
            if session_duration > 3600:  # 1時間以上
                print(f"⏰ 長時間セッション検出: {session_duration:.1f}秒")
                print(f"   統計情報をリセットします")
                self.stats['total_samples'] = 0
                self.stats['total_trajectories'] = 0
                self.collection_start_time = datetime.now()
    
    def get_memory_usage(self) -> Dict:
        """メモリ使用状況を取得"""
        return {
            'trajectory_buffer_size': len(self.trajectory_buffer),
            'max_buffer_size': self.trajectory_buffer_size,
            'memory_usage_percent': (len(self.trajectory_buffer) / self.trajectory_buffer_size) * 100,
            'session_duration': (datetime.now() - self.collection_start_time).total_seconds() if self.collection_start_time else 0
        }
