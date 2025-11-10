"""
共通特徴量抽出モジュール
学習・推論で同一の前処理（30次元×時系列）を使用するためのユーティリティ
"""

from typing import Dict, List
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.feature_config import get_feature_dim, get_fps, get_sequence_length, get_normalization_range, get_new_feature_range, get_window_size


class FeatureExtractor:
    """キーボード意図推定用の共通特徴量抽出器"""

    def __init__(self, sequence_length: int = None, fps: float = None):
        # 設定ファイルから値を取得（引数で上書き可能）
        self.sequence_length = sequence_length or get_sequence_length()
        self.feature_dim = get_feature_dim()  # 設定ファイルから取得
        self.fps = fps or get_fps()  # 設定ファイルから取得

    def extract_from_trajectory(self, trajectory_data: List[Dict]) -> np.ndarray:
        """
        サンプルの軌跡フレーム配列から特徴量配列を生成

        Args:
            trajectory_data: 各フレームの辞書データのリスト

        Returns:
            features: (sequence_length, feature_dim) のnumpy配列
        """
        features = np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        # 指先座標を先に抽出（速度/加速度計算のため）
        idx_x = []
        idx_y = []
        for i in range(min(self.sequence_length, len(trajectory_data))):
            # 後方互換性のため両方のキー名をチェック
            coords = trajectory_data[i].get('work_area_coords') or trajectory_data[i].get('keyboard_space_coords', {})
            index_finger = coords.get('index_finger', {})
            idx_x.append(float(index_finger.get('x', 0.0)))
            idx_y.append(float(index_finger.get('y', 0.0)))

        # 速度・加速度（過去のみ参照でデータリーク防止）
        vel_x = np.zeros(self.sequence_length, dtype=np.float32)
        vel_y = np.zeros(self.sequence_length, dtype=np.float32)
        acc_x = np.zeros(self.sequence_length, dtype=np.float32)
        acc_y = np.zeros(self.sequence_length, dtype=np.float32)

        for i in range(len(idx_x)):
            if i >= 1:
                vel_x[i] = (idx_x[i] - idx_x[i-1]) * self.fps
                vel_y[i] = (idx_y[i] - idx_y[i-1]) * self.fps
            if i >= 2:
                acc_x[i] = (idx_x[i] - 2*idx_x[i-1] + idx_x[i-2]) * (self.fps ** 2)
                acc_y[i] = (idx_y[i] - 2*idx_y[i-1] + idx_y[i-2]) * (self.fps ** 2)

        # 累積距離を事前計算（最適化）
        cumulative_lengths = np.zeros(min(self.sequence_length, len(trajectory_data)), dtype=np.float32)
        for i in range(1, min(self.sequence_length, len(trajectory_data))):
            if i < len(idx_x) and i < len(idx_y):
                dx = idx_x[i] - idx_x[i-1]
                dy = idx_y[i] - idx_y[i-1]
                segment_length = np.sqrt(dx**2 + dy**2)
                cumulative_lengths[i] = cumulative_lengths[i-1] + segment_length

        # 各フレームの特徴量を構築
        for i in range(min(self.sequence_length, len(trajectory_data))):
            frame = trajectory_data[i]

            # 2: 指の座標（エラーハンドリング付き）
            try:
                finger_x = idx_x[i] if i < len(idx_x) else 0.0
                finger_y = idx_y[i] if i < len(idx_y) else 0.0
            except (IndexError, TypeError):
                finger_x, finger_y = 0.0, 0.0

            # 6: 最近傍3キーへの相対座標（エラーハンドリング付き）
            nearest_keys = frame.get('nearest_keys_relative', [])
            rel = np.zeros(6, dtype=np.float32)
            try:
                for j, key_info in enumerate(nearest_keys[:3]):
                    if isinstance(key_info, dict):
                        rel[j*2] = float(key_info.get('relative_x', 0.0))
                        rel[j*2+1] = float(key_info.get('relative_y', 0.0))
            except (TypeError, ValueError, KeyError):
                rel = np.zeros(6, dtype=np.float32)

            # 3: 最近傍3キーへの距離（エラーハンドリング付き）
            dists = np.zeros(3, dtype=np.float32)
            try:
                for j, key_info in enumerate(nearest_keys[:3]):
                    if isinstance(key_info, dict):
                        dists[j] = float(key_info.get('distance', 0.0))
            except (TypeError, ValueError, KeyError):
                dists = np.zeros(3, dtype=np.float32)

            # 4: 速度(x,y), 加速度(x,y)（エラーハンドリング付き）
            try:
                vxa = vel_x[i] if i < len(vel_x) else 0.0
                vya = vel_y[i] if i < len(vel_y) else 0.0
                axa = acc_x[i] if i < len(acc_x) else 0.0
                aya = acc_y[i] if i < len(acc_y) else 0.0
            except (IndexError, TypeError):
                vxa, vya, axa, aya = 0.0, 0.0, 0.0, 0.0

            # 新規追加: 3つの新しい特徴量
            # 振幅（x方向）: 過去window_sizeフレーム分のx座標の標準偏差
            amplitude_x = 0.0
            window_size = get_window_size()
            start_idx = max(0, i - (window_size - 1))  # 過去window_sizeフレーム分の開始インデックス
            if i > 0:  # 過去のフレームがある場合
                past_x = idx_x[start_idx:i]  # 過去window_sizeフレーム（現在フレームは含まない）
                amplitude_x = np.std(past_x)
            
            # 振幅（y方向）: 過去window_sizeフレーム分のy座標の標準偏差
            amplitude_y = 0.0
            if i > 0:  # 過去のフレームがある場合
                past_y = idx_y[start_idx:i]  # 過去window_sizeフレーム（現在フレームは含まない）
                amplitude_y = np.std(past_y)
            
            # 方向転換の頻度: 過去window_sizeフレーム分のx方向速度の符号変化回数を正規化
            direction_change_freq = 0.0
            if i > 0:  # 過去のフレームがある場合
                past_vel_x = vel_x[start_idx:i]  # 過去window_sizeフレーム（現在フレームは含まない）
                direction_change_freq = self._calculate_direction_change_frequency(past_vel_x)

            # 特徴量の実装（18次元、24次元、30次元）
            if self.feature_dim == 18:
                # 18次元特徴量の実装
                features[i] = np.concatenate([
                    np.array([finger_x, finger_y], dtype=np.float32),  # 2次元: 指の座標
                    rel,  # 6次元: 最近傍3キーへの相対座標
                    dists,  # 3次元: 最近傍3キーへの距離
                    np.array([vxa, vya, axa, aya], dtype=np.float32),  # 4次元: 速度・加速度
                    np.array([amplitude_x, amplitude_y, direction_change_freq], dtype=np.float32),  # 3次元: 既存特徴量
                ])
            elif self.feature_dim == 24:
                # 24次元特徴量の実装（重要度の低い特徴量を削除）
                new_features = self._calculate_new_features(
                    i, trajectory_data, idx_x, idx_y, vel_x, vel_y, acc_x, acc_y, 
                    finger_x, finger_y, nearest_keys, cumulative_lengths,
                    use_actual_length=False  # 固定長モード
                )
                
                # 削除する特徴量: amplitude_x, amplitude_y, elapsed_time, acc_x, acc_y, acceleration_magnitude
                # 残す特徴量: target_angle, velocity_angle, angle_to_target, speed, jerk, trajectory_length, approach_velocity, trajectory_curvature, speed_std, velocity_consistency
                selected_new_features = np.array([
                    new_features[1],  # target_angle
                    new_features[2],  # velocity_angle
                    new_features[3],  # angle_to_target
                    new_features[4],  # speed
                    new_features[6],  # jerk
                    new_features[7],  # trajectory_length
                    new_features[8],  # approach_velocity
                    new_features[9],  # trajectory_curvature
                    new_features[10], # speed_std
                    new_features[11], # velocity_consistency
                ], dtype=np.float32)
                
                features[i] = np.concatenate([
                    np.array([finger_x, finger_y], dtype=np.float32),  # 2次元: 指の座標
                    rel,  # 6次元: 最近傍3キーへの相対座標
                    dists,  # 3次元: 最近傍3キーへの距離
                    np.array([vxa, vya], dtype=np.float32),  # 2次元: 速度（加速度を削除）
                    np.array([direction_change_freq], dtype=np.float32),  # 1次元: 方向転換頻度（振幅を削除）
                    selected_new_features  # 10次元: 選択された新規特徴量
                ])
            else:
                # 30次元特徴量の実装
                new_features = self._calculate_new_features(
                    i, trajectory_data, idx_x, idx_y, vel_x, vel_y, acc_x, acc_y, 
                    finger_x, finger_y, nearest_keys, cumulative_lengths,
                    use_actual_length=False  # 固定長モード
                )
                
                features[i] = np.concatenate([
                    np.array([finger_x, finger_y], dtype=np.float32),  # 2次元: 指の座標
                    rel,  # 6次元: 最近傍3キーへの相対座標
                    dists,  # 3次元: 最近傍3キーへの距離
                    np.array([vxa, vya, axa, aya], dtype=np.float32),  # 4次元: 速度・加速度
                    np.array([amplitude_x, amplitude_y, direction_change_freq], dtype=np.float32),  # 3次元: 既存特徴量
                    new_features  # 12次元: 新規特徴量
                ])

        # 正規化/クリップ（設定ファイルから取得）
        finger_coords_range = get_normalization_range('finger_coords')
        relative_coords_range = get_normalization_range('relative_coords')
        distances_range = get_normalization_range('distances')
        velocity_acceleration_range = get_normalization_range('velocity_acceleration')
        amplitude_direction_range = get_normalization_range('amplitude_direction')
        new_features_range = get_normalization_range('new_features')
        
        features[:, :2] = np.clip(features[:, :2], *finger_coords_range)  # 指の座標
        features[:, 2:8] = np.clip(features[:, 2:8], *relative_coords_range)  # 相対座標
        features[:, 8:11] = np.clip(features[:, 8:11], *distances_range)  # 距離
        
        if self.feature_dim == 18:
            features[:, 11:15] = np.clip(features[:, 11:15], *velocity_acceleration_range)  # 速度・加速度
            features[:, 15:18] = np.clip(features[:, 15:18], *amplitude_direction_range)  # 既存特徴量
        elif self.feature_dim == 24:
            features[:, 11:13] = np.clip(features[:, 11:13], *velocity_acceleration_range)  # 速度（加速度を削除）
            features[:, 13:14] = np.clip(features[:, 13:14], *amplitude_direction_range)  # 方向転換頻度（振幅を削除）
            # 24次元の新規特徴量の個別クリップ（選択された特徴量のみ）
            selected_indices = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]  # 削除した特徴量を除く
            for k, j in enumerate(selected_indices):
                clip_range = get_new_feature_range(j)  # 元のインデックス
                features[:, 14 + k] = np.clip(features[:, 14 + k], *clip_range)
        else:  # 30次元
            features[:, 11:15] = np.clip(features[:, 11:15], *velocity_acceleration_range)  # 速度・加速度
            features[:, 15:18] = np.clip(features[:, 15:18], *amplitude_direction_range)  # 既存特徴量
            # 新規特徴量の個別クリップ（設定ファイルから範囲を取得）
            for j in range(18, 30):  # 18-29番目の特徴量
                clip_range = get_new_feature_range(j - 18)  # インデックス0-11に対応
                features[:, j] = np.clip(features[:, j], *clip_range)

        # 出力形状の保証
        assert features.shape == (self.sequence_length, self.feature_dim), \
            f"特徴量の形状が不正: {features.shape}, 期待: ({self.sequence_length}, {self.feature_dim})"

        return features

    def extract_from_trajectory_variable_length(self, trajectory_data: List[Dict]) -> np.ndarray:
        """
        可変長軌跡データから特徴量を抽出（可変長対応）
        
        Args:
            trajectory_data: 各フレームの辞書データのリスト（長さは可変）
        
        Returns:
            features: (actual_length, feature_dim) のnumpy配列
        """
        actual_length = len(trajectory_data)
        if actual_length == 0:
            # 空の場合は最小長（5フレーム）のダミーデータを返す
            # ダミーデータを作成
            trajectory_data = [{}] * 5
            actual_length = 5
        
        features = np.zeros((actual_length, self.feature_dim), dtype=np.float32)
        
        # 指先座標を先に抽出（速度/加速度計算のため）
        idx_x = []
        idx_y = []
        for i in range(actual_length):
            # 後方互換性のため両方のキー名をチェック
            coords = trajectory_data[i].get('work_area_coords') or trajectory_data[i].get('keyboard_space_coords', {})
            index_finger = coords.get('index_finger', {})
            idx_x.append(float(index_finger.get('x', 0.0)))
            idx_y.append(float(index_finger.get('y', 0.0)))
        
        # 速度・加速度（過去のみ参照でデータリーク防止）
        vel_x = np.zeros(actual_length, dtype=np.float32)
        vel_y = np.zeros(actual_length, dtype=np.float32)
        acc_x = np.zeros(actual_length, dtype=np.float32)
        acc_y = np.zeros(actual_length, dtype=np.float32)
        
        for i in range(len(idx_x)):
            if i >= 1:
                vel_x[i] = (idx_x[i] - idx_x[i-1]) * self.fps
                vel_y[i] = (idx_y[i] - idx_y[i-1]) * self.fps
            if i >= 2:
                acc_x[i] = (idx_x[i] - 2*idx_x[i-1] + idx_x[i-2]) * (self.fps ** 2)
                acc_y[i] = (idx_y[i] - 2*idx_y[i-1] + idx_y[i-2]) * (self.fps ** 2)
        
        # 累積距離を事前計算（最適化）
        cumulative_lengths = np.zeros(actual_length, dtype=np.float32)
        for i in range(1, actual_length):
            if i < len(idx_x) and i < len(idx_y):
                dx = idx_x[i] - idx_x[i-1]
                dy = idx_y[i] - idx_y[i-1]
                segment_length = np.sqrt(dx**2 + dy**2)
                cumulative_lengths[i] = cumulative_lengths[i-1] + segment_length
        
        # 各フレームの特徴量を構築
        for i in range(actual_length):
            frame = trajectory_data[i]
            
            # 2: 指の座標（エラーハンドリング付き）
            try:
                finger_x = idx_x[i] if i < len(idx_x) else 0.0
                finger_y = idx_y[i] if i < len(idx_y) else 0.0
            except (IndexError, TypeError):
                finger_x, finger_y = 0.0, 0.0
            
            # 6: 最近傍3キーへの相対座標（エラーハンドリング付き）
            nearest_keys = frame.get('nearest_keys_relative', [])
            rel = np.zeros(6, dtype=np.float32)
            try:
                for j, key_info in enumerate(nearest_keys[:3]):
                    if isinstance(key_info, dict):
                        rel[j*2] = float(key_info.get('relative_x', 0.0))
                        rel[j*2+1] = float(key_info.get('relative_y', 0.0))
            except (TypeError, ValueError, KeyError):
                rel = np.zeros(6, dtype=np.float32)
            
            # 3: 最近傍3キーへの距離（エラーハンドリング付き）
            dists = np.zeros(3, dtype=np.float32)
            try:
                for j, key_info in enumerate(nearest_keys[:3]):
                    if isinstance(key_info, dict):
                        dists[j] = float(key_info.get('distance', 0.0))
            except (TypeError, ValueError, KeyError):
                dists = np.zeros(3, dtype=np.float32)
            
            # 4: 速度(x,y), 加速度(x,y)（エラーハンドリング付き）
            try:
                vxa = vel_x[i] if i < len(vel_x) else 0.0
                vya = vel_y[i] if i < len(vel_y) else 0.0
                axa = acc_x[i] if i < len(acc_x) else 0.0
                aya = acc_y[i] if i < len(acc_y) else 0.0
            except (IndexError, TypeError):
                vxa, vya, axa, aya = 0.0, 0.0, 0.0, 0.0
            
            # 新規追加: 3つの新しい特徴量
            # 振幅（x方向）: 過去window_sizeフレーム分のx座標の標準偏差
            amplitude_x = 0.0
            window_size = get_window_size()
            start_idx = max(0, i - (window_size - 1))
            if i > 0:
                past_x = idx_x[start_idx:i]
                if len(past_x) > 0:
                    amplitude_x = np.std(past_x) if len(past_x) > 1 else 0.0
            
            # 振幅（y方向）: 過去window_sizeフレーム分のy座標の標準偏差
            amplitude_y = 0.0
            if i > 0:
                past_y = idx_y[start_idx:i]
                if len(past_y) > 0:
                    amplitude_y = np.std(past_y) if len(past_y) > 1 else 0.0
            
            # 方向転換の頻度: 過去window_sizeフレーム分のx方向速度の符号変化回数を正規化
            direction_change_freq = 0.0
            if i > 0:
                past_vel_x = vel_x[start_idx:i]
                direction_change_freq = self._calculate_direction_change_frequency(past_vel_x)
            
            # 特徴量の実装（18次元、24次元、30次元）
            if self.feature_dim == 18:
                # 18次元特徴量の実装
                features[i] = np.concatenate([
                    np.array([finger_x, finger_y], dtype=np.float32),
                    rel,
                    dists,
                    np.array([vxa, vya, axa, aya], dtype=np.float32),
                    np.array([amplitude_x, amplitude_y, direction_change_freq], dtype=np.float32),
                ])
            elif self.feature_dim == 24:
                # 24次元特徴量の実装
                new_features = self._calculate_new_features(
                    i, trajectory_data, idx_x, idx_y, vel_x, vel_y, acc_x, acc_y,
                    finger_x, finger_y, nearest_keys, cumulative_lengths,
                    use_actual_length=True  # 可変長モード
                )
                
                selected_new_features = np.array([
                    new_features[1], new_features[2], new_features[3], new_features[4],
                    new_features[6], new_features[7], new_features[8], new_features[9],
                    new_features[10], new_features[11],
                ], dtype=np.float32)
                
                features[i] = np.concatenate([
                    np.array([finger_x, finger_y], dtype=np.float32),
                    rel,
                    dists,
                    np.array([vxa, vya], dtype=np.float32),
                    np.array([direction_change_freq], dtype=np.float32),
                    selected_new_features
                ])
            else:
                # 30次元特徴量の実装
                new_features = self._calculate_new_features(
                    i, trajectory_data, idx_x, idx_y, vel_x, vel_y, acc_x, acc_y,
                    finger_x, finger_y, nearest_keys, cumulative_lengths,
                    use_actual_length=True  # 可変長モード
                )
                
                features[i] = np.concatenate([
                    np.array([finger_x, finger_y], dtype=np.float32),
                    rel,
                    dists,
                    np.array([vxa, vya, axa, aya], dtype=np.float32),
                    np.array([amplitude_x, amplitude_y, direction_change_freq], dtype=np.float32),
                    new_features
                ])
        
        # 正規化/クリップ（設定ファイルから取得）
        finger_coords_range = get_normalization_range('finger_coords')
        relative_coords_range = get_normalization_range('relative_coords')
        distances_range = get_normalization_range('distances')
        velocity_acceleration_range = get_normalization_range('velocity_acceleration')
        amplitude_direction_range = get_normalization_range('amplitude_direction')
        new_features_range = get_normalization_range('new_features')
        
        features[:, :2] = np.clip(features[:, :2], *finger_coords_range)
        features[:, 2:8] = np.clip(features[:, 2:8], *relative_coords_range)
        features[:, 8:11] = np.clip(features[:, 8:11], *distances_range)
        
        if self.feature_dim == 18:
            features[:, 11:15] = np.clip(features[:, 11:15], *velocity_acceleration_range)
            features[:, 15:18] = np.clip(features[:, 15:18], *amplitude_direction_range)
        elif self.feature_dim == 24:
            features[:, 11:13] = np.clip(features[:, 11:13], *velocity_acceleration_range)
            features[:, 13:14] = np.clip(features[:, 13:14], *amplitude_direction_range)
            selected_indices = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
            for k, j in enumerate(selected_indices):
                clip_range = get_new_feature_range(j)
                features[:, 14 + k] = np.clip(features[:, 14 + k], *clip_range)
        else:  # 30次元
            features[:, 11:15] = np.clip(features[:, 11:15], *velocity_acceleration_range)
            features[:, 15:18] = np.clip(features[:, 15:18], *amplitude_direction_range)
            for j in range(18, 30):
                clip_range = get_new_feature_range(j - 18)
                features[:, j] = np.clip(features[:, j], *clip_range)
        
        return features

    def _calculate_new_features(self, i: int, trajectory_data: List[Dict], idx_x: List[float], idx_y: List[float], 
                               vel_x: np.ndarray, vel_y: np.ndarray, acc_x: np.ndarray, acc_y: np.ndarray,
                               finger_x: float, finger_y: float, nearest_keys: List[Dict], cumulative_lengths: np.ndarray,
                               use_actual_length: bool = False) -> np.ndarray:
        """
        新規追加特徴量（12次元）を計算
        
        Args:
            use_actual_length: Trueの場合、trajectory_dataの実際の長さを使用（可変長モード）
                               Falseの場合、self.sequence_lengthを使用（固定長モード）
        
        Returns:
            new_features: 12次元の新規特徴量配列
        """
        # 1. elapsed_time: 軌跡開始からの経過時間（正規化）
        if use_actual_length:
            # 可変長対応: trajectory_dataの実際の長さを使用
            actual_length = len(trajectory_data)
            elapsed_time = i / actual_length if actual_length > 0 else 0.0
        else:
            # 固定長モード: sequence_lengthを使用
            elapsed_time = i / self.sequence_length if self.sequence_length > 0 else 0.0
        
        # 2. target_angle: 最近傍キーへの角度
        target_angle = 0.0
        if nearest_keys and len(nearest_keys) > 0:
            key_info = nearest_keys[0]
            if isinstance(key_info, dict):
                rel_x = key_info.get('relative_x', 0.0)
                rel_y = key_info.get('relative_y', 0.0)
                target_angle = np.arctan2(rel_y, rel_x)
        
        # 3. velocity_angle: 速度ベクトルの角度
        velocity_angle = 0.0
        if i > 0 and i < len(vel_x) and i < len(vel_y):
            if vel_x[i] != 0 or vel_y[i] != 0:
                velocity_angle = np.arctan2(vel_y[i], vel_x[i])
        
        # 4. angle_to_target: 目標角度と速度角度の差
        angle_to_target = abs(target_angle - velocity_angle)
        angle_to_target = min(angle_to_target, np.pi)  # 0-πにクリップ
        
        # 5. speed: 速度の大きさ
        speed = 0.0
        if i > 0 and i < len(vel_x) and i < len(vel_y):
            speed = np.sqrt(vel_x[i]**2 + vel_y[i]**2)
        
        # 6. acceleration_magnitude: 加速度の大きさ
        acceleration_magnitude = 0.0
        if i > 1 and i < len(acc_x) and i < len(acc_y):
            acceleration_magnitude = np.sqrt(acc_x[i]**2 + acc_y[i]**2)
        
        # 7. jerk: ジャーク（加速度の変化率）
        jerk = 0.0
        if i > 2:
            jerk_x = (acc_x[i] - acc_x[i-1]) * self.fps if i < len(acc_x) else 0.0
            jerk_y = (acc_y[i] - acc_y[i-1]) * self.fps if i < len(acc_y) else 0.0
            jerk = np.sqrt(jerk_x**2 + jerk_y**2)
        
        # 8. trajectory_length: 軌跡の累積移動距離（最適化版）
        trajectory_length = cumulative_lengths[i] if i < len(cumulative_lengths) else 0.0
        
        # 9. approach_velocity: キーへの接近速度
        approach_velocity = 0.0
        if nearest_keys and len(nearest_keys) > 0 and i > 0:
            key_info = nearest_keys[0]
            if isinstance(key_info, dict):
                current_dist = key_info.get('distance', 0.0)
                
                # 1フレーム前の距離を取得
                if i-1 < len(trajectory_data):
                    prev_frame = trajectory_data[i-1]
                    prev_nearest = prev_frame.get('nearest_keys_relative', [])
                    if prev_nearest and len(prev_nearest) > 0:
                        prev_key = prev_nearest[0]
                        if isinstance(prev_key, dict):
                            prev_dist = prev_key.get('distance', 0.0)
                            # 接近速度 = (前の距離 - 現在の距離) * fps
                            # 正の値 = 近づいている、負の値 = 遠ざかっている
                            approach_velocity = (prev_dist - current_dist) * self.fps
        
        # 10. trajectory_curvature: 軌跡の曲率
        trajectory_curvature = 0.0
        if i >= 2:
            trajectory_curvature = self._calculate_curvature(
                idx_x[i-2:i+1], idx_y[i-2:i+1]
            )
        
        # 11. speed_std: 速度の標準偏差（過去window_sizeフレーム）
        speed_std = 0.0
        window_size = get_window_size()
        start_idx = max(0, i - (window_size - 1))
        if i > 0:
            past_speeds = []
            for j in range(start_idx, i):
                if j > 0 and j < len(vel_x) and j < len(vel_y):
                    past_speeds.append(np.sqrt(vel_x[j]**2 + vel_y[j]**2))
            if len(past_speeds) > 1:
                speed_std = np.std(past_speeds)
        
        # 12. velocity_consistency: 速度の変動係数
        velocity_consistency = 0.0
        if i > 0:
            past_speeds = []
            for j in range(start_idx, i):
                if j > 0 and j < len(vel_x) and j < len(vel_y):
                    past_speeds.append(np.sqrt(vel_x[j]**2 + vel_y[j]**2))
            if len(past_speeds) > 1:
                mean_speed = np.mean(past_speeds)
                if mean_speed > 1e-8:
                    velocity_consistency = np.std(past_speeds) / mean_speed
        
        return np.array([
            elapsed_time, target_angle, velocity_angle, angle_to_target,
            speed, acceleration_magnitude, jerk, trajectory_length,
            approach_velocity, trajectory_curvature, speed_std, velocity_consistency
        ], dtype=np.float32)

    def _calculate_curvature(self, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """
        3点から軌跡の曲率を計算
        
        Args:
            x_coords: x座標の配列（3点）
            y_coords: y座標の配列（3点）
            
        Returns:
            curvature: 曲率（正規化済み）
        """
        if len(x_coords) != 3 or len(y_coords) != 3:
            return 0.0
        
        # 3点の座標
        x1, x2, x3 = x_coords
        y1, y2, y3 = y_coords
        
        # ベクトル計算
        v1x, v1y = x2 - x1, y2 - y1
        v2x, v2y = x3 - x2, y3 - y2
        
        # ベクトルの大きさ
        v1_mag = np.sqrt(v1x**2 + v1y**2)
        v2_mag = np.sqrt(v2x**2 + v2y**2)
        
        if v1_mag < 1e-8 or v2_mag < 1e-8:
            return 0.0
        
        # 外積による曲率計算
        cross_product = v1x * v2y - v1y * v2x
        curvature = abs(cross_product) / (v1_mag * v2_mag)
        
        # 正規化（0-1の範囲）
        return min(curvature, 1.0)

    def _calculate_direction_change_frequency(self, velocity_sequence: np.ndarray) -> float:
        """
        速度シーケンスから方向転換の頻度を計算
        
        Args:
            velocity_sequence: 速度の配列
            
        Returns:
            direction_change_freq: 方向転換の頻度（0.0-1.0）
        """
        if len(velocity_sequence) < 2:
            return 0.0
        
        # 符号の変化を検出
        signs = np.sign(velocity_sequence)
        sign_changes = 0
        
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1] and signs[i-1] != 0:  # 0を除く
                sign_changes += 1
        
        # フレーム数で正規化（最大値は1.0）
        max_possible_changes = len(velocity_sequence) - 1
        if max_possible_changes > 0:
            return sign_changes / max_possible_changes
        else:
            return 0.0


