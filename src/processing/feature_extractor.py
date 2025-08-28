"""
共通特徴量抽出モジュール
学習・推論で同一の前処理（15次元×時系列）を使用するためのユーティリティ
"""

from typing import Dict, List
import numpy as np


class FeatureExtractor:
    """キーボード意図推定用の共通特徴量抽出器"""

    def __init__(self, sequence_length: int = 60, fps: float = 30.0):
        self.sequence_length = sequence_length
        self.feature_dim = 15  # 2 + 6 + 3 + 4
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
            kb = trajectory_data[i].get('keyboard_space_coords', {})
            index_finger = kb.get('index_finger', {})
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

            # 2: 指の座標
            finger_x = idx_x[i] if i < len(idx_x) else 0.0
            finger_y = idx_y[i] if i < len(idx_y) else 0.0

            # 6: 最近傍3キーへの相対座標
            nearest_keys = frame.get('nearest_keys_relative', [])
            rel = np.zeros(6, dtype=np.float32)
            for j, key_info in enumerate(nearest_keys[:3]):
                rel[j*2] = float(key_info.get('relative_x', 0.0))
                rel[j*2+1] = float(key_info.get('relative_y', 0.0))

            # 3: 最近傍3キーへの距離
            dists = np.zeros(3, dtype=np.float32)
            for j, key_info in enumerate(nearest_keys[:3]):
                dists[j] = float(key_info.get('distance', 0.0))

            # 4: 速度(x,y), 加速度(x,y)
            vxa = vel_x[i]
            vya = vel_y[i]
            axa = acc_x[i]
            aya = acc_y[i]

            features[i] = np.concatenate([
                np.array([finger_x, finger_y], dtype=np.float32),
                rel,
                dists,
                np.array([vxa, vya, axa, aya], dtype=np.float32)
            ])

        # 正規化/クリップ
        features[:, :2] = np.clip(features[:, :2], 0.0, 1.0)
        features[:, 2:8] = np.clip(features[:, 2:8], -5.0, 5.0)
        features[:, 8:11] = np.clip(features[:, 8:11], 0.0, 10.0)
        features[:, 11:] = np.clip(features[:, 11:], -5.0, 5.0)

        return features


