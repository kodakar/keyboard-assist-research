"""
共通特徴量抽出モジュール
学習・推論で同一の前処理（18次元×時系列）を使用するためのユーティリティ
"""

from typing import Dict, List
import numpy as np


class FeatureExtractor:
    """キーボード意図推定用の共通特徴量抽出器"""

    def __init__(self, sequence_length: int = 60, fps: float = 30.0):
        self.sequence_length = sequence_length
        self.feature_dim = 18  # 2 + 6 + 3 + 4 + 3 (新規追加)
        self.fps = fps

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
            # 振幅（x方向）: 過去10フレーム分のx座標の標準偏差
            amplitude_x = 0.0
            start_idx = max(0, i-9)  # 過去10フレーム分の開始インデックス
            if i > 0:  # 過去のフレームがある場合
                past_x = idx_x[start_idx:i]  # i-9からi-1まで（現在フレームは含まない）
                amplitude_x = np.std(past_x)
            
            # 振幅（y方向）: 過去10フレーム分のy座標の標準偏差
            amplitude_y = 0.0
            if i > 0:  # 過去のフレームがある場合
                past_y = idx_y[start_idx:i]  # i-9からi-1まで（現在フレームは含まない）
                amplitude_y = np.std(past_y)
            
            # 方向転換の頻度: 過去10フレーム分のx方向速度の符号変化回数を正規化
            direction_change_freq = 0.0
            if i > 0:  # 過去のフレームがある場合
                past_vel_x = vel_x[start_idx:i]  # i-9からi-1まで（現在フレームは含まない）
                direction_change_freq = self._calculate_direction_change_frequency(past_vel_x)

            features[i] = np.concatenate([
                np.array([finger_x, finger_y], dtype=np.float32),
                rel,
                dists,
                np.array([vxa, vya, axa, aya], dtype=np.float32),
                np.array([amplitude_x, amplitude_y, direction_change_freq], dtype=np.float32)
            ])

        # 正規化/クリップ
        features[:, :2] = np.clip(features[:, :2], 0.0, 1.0)  # 指の座標
        features[:, 2:8] = np.clip(features[:, 2:8], -5.0, 5.0)  # 相対座標
        features[:, 8:11] = np.clip(features[:, 8:11], 0.0, 10.0)  # 距離
        features[:, 11:15] = np.clip(features[:, 11:15], -5.0, 5.0)  # 速度・加速度
        features[:, 15:18] = np.clip(features[:, 15:18], 0.0, 1.0)  # 新規特徴量（振幅・方向転換頻度）

        # 出力形状の保証
        assert features.shape == (self.sequence_length, self.feature_dim), \
            f"特徴量の形状が不正: {features.shape}, 期待: ({self.sequence_length}, {self.feature_dim})"

        return features

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


