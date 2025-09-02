# src/processing/data_loader.py
"""
PyTorch用データセットクラス
キーボード入力意図推定用のデータローダー
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from collections import Counter
import random
from sklearn.model_selection import train_test_split
import warnings
from .feature_extractor import FeatureExtractor


class KeyboardIntentDataset(Dataset):
    """
    キーボード入力意図推定用のデータセットクラス
    PyTorchのDatasetクラスを継承
    """
    
    # 37キーの定義（英字26個 + 数字10個 + スペース）
    KEY_CHARS = (
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' '
    )
    
    def __init__(self, data_dir: str, sequence_length: int = 60, 
                 train: bool = True, train_ratio: float = 0.8,
                 augment: bool = False, noise_std: float = 0.01,
                 random_seed: int = 42):
        """
        データセットの初期化
        
        Args:
            data_dir: データディレクトリのパス
            sequence_length: 時系列データの長さ（フレーム数）
            train: 訓練データかどうか
            train_ratio: 訓練データの割合
            augment: データ拡張を行うかどうか
            noise_std: ガウシアンノイズの標準偏差
            random_seed: 乱数シード
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.train = train
        self.train_ratio = train_ratio
        self.augment = augment
        self.noise_std = noise_std
        
        # 乱数シードを設定
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # データファイルの読み込み
        self.samples = self._load_data_files()
        
        # 訓練/検証の分割
        self.samples = self._split_train_val()
        
        # 特徴量の次元数（固定）
        self.feature_dim = 18
        
        # 特徴量抽出器のインスタンス化
        self.feature_extractor = FeatureExtractor(sequence_length=self.sequence_length)
        
        # クラス数
        self.num_classes = len(self.KEY_CHARS)
        
        print(f"✅ データセット初期化完了")
        print(f"   データディレクトリ: {data_dir}")
        print(f"   サンプル数: {len(self.samples)}")
        print(f"   特徴量次元: {self.feature_dim}")
        print(f"   時系列長: {sequence_length}")
        print(f"   クラス数: {self.num_classes}")
        print(f"   モード: {'訓練' if train else '検証'}")
        print(f"   データ拡張: {'有効' if augment else '無効'}")
    
    def _load_data_files(self) -> List[Dict]:
        """データディレクトリから全JSONファイルを読み込み"""
        samples = []
        
        try:
            print(f"🔍 データディレクトリを探索中: {self.data_dir}")
            # データディレクトリ内の全JSONファイルを探索
            for root, dirs, files in os.walk(self.data_dir):
                print(f"📁 探索ディレクトリ: {root}")
                print(f"📁 サブディレクトリ: {dirs}")
                print(f"📁 ファイル: {files}")
                for file in files:
                    print(f"🔍 ファイルチェック: {file} (JSON: {file.endswith('.json')}, sample_含む: {'sample_' in file})")
                    if file.endswith('.json') and 'sample_' in file:
                        file_path = os.path.join(root, file)
                        print(f"📄 サンプルファイル発見: {file_path}")
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                sample_data = json.load(f)
                                print(f"📄 ファイル読み込み成功: {file_path}")
                                
                                # 必要なデータが含まれているかチェック
                                if self._validate_sample(sample_data):
                                    print(f"✅ サンプル検証成功: {file_path}")
                                    samples.append(sample_data)
                                else:
                                    print(f"❌ サンプル検証失敗: {file_path}")
                                
                        except Exception as e:
                            warnings.warn(f"ファイル読み込みエラー {file_path}: {e}")
                            continue
            
            print(f"📁 {len(samples)}個のサンプルファイルを読み込みました")
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return []
        
        return samples
    
    def _validate_sample(self, sample_data: Dict) -> bool:
        """サンプルデータの妥当性をチェック"""
        required_fields = ['target_char', 'trajectory_data', 'coordinate_system']
        
        print(f"🔍 サンプル検証開始: {list(sample_data.keys())}")
        
        # 必須フィールドの存在チェック
        for field in required_fields:
            if field not in sample_data:
                print(f"❌ 必須フィールド不足: {field}")
                return False
        
        # 座標系のチェック（新しい座標系も受け入れる）
        coord_sys = sample_data.get('coordinate_system')
        print(f"🔍 座標系: {coord_sys}")
        if coord_sys not in ['relative_keyboard_space', 'work_area_v2']:
            print(f"❌ 座標系不一致: {coord_sys} (期待値: relative_keyboard_space または work_area_v2)")
            return False
        
        # 軌跡データの存在チェック
        trajectory_data = sample_data.get('trajectory_data', [])
        print(f"🔍 軌跡データ長: {len(trajectory_data)}")
        if not isinstance(trajectory_data, list) or len(trajectory_data) == 0:
            print(f"❌ 軌跡データ不正: {type(trajectory_data)}, 長さ: {len(trajectory_data) if isinstance(trajectory_data, list) else 'N/A'}")
            return False
        
        # 目標文字の妥当性チェック
        target_char = sample_data.get('target_char', '').lower()
        print(f"🔍 目標文字: {target_char}")
        if target_char not in self.KEY_CHARS:
            print(f"❌ 目標文字不正: {target_char}")
            return False
        
        print(f"✅ サンプル検証成功")
        return True
    
    def _split_train_val(self) -> List[Dict]:
        """訓練/検証データの分割"""
        if len(self.samples) == 0:
            return []
        
        # ユーザーIDでグループ化
        user_groups = {}
        for sample in self.samples:
            user_id = sample.get('user_id', 'unknown')
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(sample)
        
        # 各ユーザーごとに訓練/検証分割
        train_samples = []
        val_samples = []
        
        for user_id, user_samples in user_groups.items():
            if len(user_samples) < 2:
                # サンプルが少ない場合は全件を訓練データに
                train_samples.extend(user_samples)
                print(f"   📝 ユーザー {user_id}: サンプル数 {len(user_samples)} < 2 のため全件を訓練データに")
            else:
                # 訓練/検証分割（クラス数が少ない場合はstratifyを無効化）
                try:
                    user_train, user_val = train_test_split(
                        user_samples, 
                        train_size=self.train_ratio, 
                        random_state=42,
                        stratify=[s.get('target_char', '').lower() for s in user_samples]
                    )
                except ValueError as e:
                    # stratifyでエラーが発生した場合（クラス数が少ない場合）
                    print(f"   ⚠️ stratify分割でエラー: {e}")
                    print(f"   📝 通常分割を使用します")
                    user_train, user_val = train_test_split(
                        user_samples, 
                        train_size=self.train_ratio, 
                        random_state=42
                    )
                train_samples.extend(user_train)
                val_samples.extend(user_val)
                print(f"   📝 ユーザー {user_id}: 訓練 {len(user_train)}, 検証 {len(user_val)}")
        
        # サンプル数が少ない場合の特別処理
        if len(train_samples) == 0:
            print(f"   ⚠️ 警告: 訓練データが0件です")
            return []
        
        if len(val_samples) == 0:
            print(f"   ⚠️ 警告: 検証データが0件です。訓練データを検証データとしても使用します")
            # 検証データがない場合は、訓練データの一部を検証データとして使用
            if len(train_samples) >= 2:
                val_samples = train_samples[:1]  # 最初の1件を検証データに
                train_samples = train_samples[1:]  # 残りを訓練データに
                print(f"   📝 検証データを作成: 訓練 {len(train_samples)}, 検証 {len(val_samples)}")
        
        # 指定されたモードに応じてサンプルを返す
        if self.train:
            return train_samples
        else:
            return val_samples
    
    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, int]:
        """
        指定されたインデックスのサンプルを取得
        
        Args:
            idx: サンプルのインデックス
            
        Returns:
            features: 特徴量テンソル (sequence_length, feature_dim)
            label: ラベル（0-36の整数）
        """
        sample = self.samples[idx]
        
        # 軌跡データから特徴量を抽出
        trajectory = sample.get('trajectory_data', [])
        if not self.feature_extractor:
            self.feature_extractor = FeatureExtractor(sequence_length=self.sequence_length)
        
        # 軌跡データの長さを統一
        trajectory = self._normalize_trajectory_length(trajectory)
        
        features_np = self.feature_extractor.extract_from_trajectory(trajectory)
        features = torch.FloatTensor(features_np)
        
        # ラベルを取得
        target_char = sample.get('target_char', '').lower()
        label = self.key_to_index(target_char)
        
        # データ拡張（訓練時のみ）
        if self.train and self.augment:
            features = self._augment_features(features)
        
        return features, label
    
    def _extract_features(self, sample: Dict) -> torch.FloatTensor:
        """後方互換のため残す（内部でFeatureExtractorを呼ぶ）"""
        if not self.feature_extractor:
            self.feature_extractor = FeatureExtractor(sequence_length=self.sequence_length)
        features_np = self.feature_extractor.extract_from_trajectory(sample.get('trajectory_data', []))
        return torch.FloatTensor(features_np)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """非推奨（FeatureExtractorに移管済み）。互換のため残置。"""
        return features
    
    def _normalize_trajectory_length(self, trajectory: List[Dict]) -> List[Dict]:
        """軌跡データの長さをsequence_lengthに統一"""
        if len(trajectory) == self.sequence_length:
            return trajectory
        elif len(trajectory) > self.sequence_length:
            # 長すぎる場合は中央部分を抽出
            start_idx = (len(trajectory) - self.sequence_length) // 2
            return trajectory[start_idx:start_idx + self.sequence_length]
        else:
            # 短すぎる場合は最後のフレームを繰り返し
            normalized = trajectory.copy()
            last_frame = trajectory[-1] if trajectory else {}
            while len(normalized) < self.sequence_length:
                normalized.append(last_frame)
            return normalized
    
    def _augment_features(self, features: torch.FloatTensor) -> torch.FloatTensor:
        """特徴量のデータ拡張"""
        features = features.clone()
        
        # ガウシアンノイズの追加
        noise = torch.randn_like(features) * self.noise_std
        features = features + noise
        
        # 座標の微小な平行移動
        if random.random() < 0.2:  # 20%の確率で実行
            shift = torch.randn(2) * 0.01  # 微小な移動
            features[:, :2] = features[:, :2] + shift.unsqueeze(0)
        
        # 人工的な震えの追加（50%の確率で実行）
        if random.random() < 0.5:
            features = self._add_artificial_tremor(features)
        
        # 形状の保証
        assert features.shape == (self.sequence_length, self.feature_dim), \
            f"データ拡張後の形状が不正: {features.shape}, 期待: ({self.sequence_length}, {self.feature_dim})"
        
        return features
    
    def _add_artificial_tremor(self, features: torch.FloatTensor) -> torch.FloatTensor:
        """人工的な震えを特徴量に追加"""
        features = features.clone()
        
        # 基本となる正弦波のパラメータをランダムに決定
        frequency = random.uniform(4.0, 12.0)  # 4Hzから12Hz
        amplitude_x = random.uniform(0.005, 0.02)  # 振幅0.005から0.02
        amplitude_y = amplitude_x * 0.8  # y方向はx方向の80%の強さ
        
        # 時間軸の生成（フレーム数に基づく）
        time_steps = torch.arange(self.sequence_length, dtype=torch.float32)
        
        # 正弦波の生成
        tremor_x = amplitude_x * torch.sin(2 * torch.pi * frequency * time_steps / self.sequence_length)
        tremor_y = amplitude_y * torch.sin(2 * torch.pi * frequency * time_steps / self.sequence_length)
        
        # 不規則性を加える（正規分布ノイズ、振幅の10%程度）
        noise_x = torch.randn(self.sequence_length) * amplitude_x * 0.1
        noise_y = torch.randn(self.sequence_length) * amplitude_y * 0.1
        
        # 震えデータを座標に加算
        features[:, 0] = features[:, 0] + tremor_x + noise_x  # x座標（0列目）
        features[:, 1] = features[:, 1] + tremor_y + noise_y  # y座標（1列目）
        
        # 座標が0.0から1.0の範囲に収まるようにクリッピング
        features[:, :2] = torch.clamp(features[:, :2], 0.0, 1.0)
        
        return features
    
    def key_to_index(self, key: str) -> int:
        """キー文字を0-36のインデックスに変換"""
        key = key.lower()
        if key in self.KEY_CHARS:
            return self.KEY_CHARS.index(key)
        else:
            # 不明なキーの場合は0を返す
            warnings.warn(f"不明なキー: {key}, インデックス0を使用")
            return 0
    
    def index_to_key(self, index: int) -> str:
        """インデックスをキー文字に変換"""
        if 0 <= index < len(self.KEY_CHARS):
            return self.KEY_CHARS[index]
        else:
            # 範囲外のインデックスの場合は'a'を返す
            warnings.warn(f"範囲外のインデックス: {index}, キー'a'を使用")
            return 'a'
    
    def get_label_distribution(self) -> Dict[str, int]:
        """各キーのサンプル数を返す"""
        label_counts = Counter()
        
        for sample in self.samples:
            target_char = sample.get('target_char', '').lower()
            if target_char in self.KEY_CHARS:
                label_counts[target_char] += 1
        
        return dict(label_counts)
    
    def get_class_weights(self) -> torch.FloatTensor:
        """クラス重みを計算（クラスバランス調整用）"""
        label_dist = self.get_label_distribution()
        
        if not label_dist:
            return torch.ones(self.num_classes)
        
        # 各クラスのサンプル数を取得
        class_counts = []
        for key in self.KEY_CHARS:
            count = label_dist.get(key, 1)  # 最低1は保証
            class_counts.append(count)
        
        # クラス重みを計算（サンプル数の逆数）
        total_samples = sum(class_counts)
        class_weights = [total_samples / (self.num_classes * count) for count in class_counts]
        
        return torch.FloatTensor(class_weights)
    
    def get_sample_info(self) -> Dict:
        """データセットの情報を取得"""
        label_dist = self.get_label_distribution()
        
        return {
            'total_samples': len(self.samples),
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'label_distribution': label_dist,
            'mode': 'train' if self.train else 'validation',
            'augmentation': self.augment
        }


def create_data_loaders(data_dir: str, batch_size: int = 32, 
                       sequence_length: int = 60, train_ratio: float = 0.8,
                       augment: bool = True, num_workers: int = 0,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    データローダーを作成
    
    Args:
        data_dir: データディレクトリのパス
        batch_size: バッチサイズ
        sequence_length: 時系列データの長さ
        train_ratio: 訓練データの割合
        augment: データ拡張を行うかどうか
        num_workers: データ読み込みのワーカー数
        random_seed: 乱数シード
        
    Returns:
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
    """
    
    # 訓練データセット
    train_dataset = KeyboardIntentDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        train=True,
        train_ratio=train_ratio,
        augment=augment,
        random_seed=random_seed
    )
    
    # 検証データセット
    val_dataset = KeyboardIntentDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        train=False,
        train_ratio=train_ratio,
        augment=False,  # 検証時は拡張なし
        random_seed=random_seed
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✅ データローダー作成完了")
    print(f"   訓練データ: {len(train_dataset)} サンプル")
    print(f"   検証データ: {len(val_dataset)} サンプル")
    print(f"   バッチサイズ: {batch_size}")
    
    return train_loader, val_loader


# 使用例
if __name__ == "__main__":
    # データセットのテスト
    data_dir = "data/training/user_001"
    
    if os.path.exists(data_dir):
        # データセットの作成
        dataset = KeyboardIntentDataset(data_dir, augment=True)
        
        # サンプル情報の表示
        info = dataset.get_sample_info()
        print("データセット情報:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # クラス重みの表示
        class_weights = dataset.get_class_weights()
        print(f"クラス重み: {class_weights[:10]}...")  # 最初の10個
        
        # サンプルの取得テスト
        if len(dataset) > 0:
            features, label = dataset[0]
            print(f"サンプル0:")
            print(f"  特徴量形状: {features.shape}")
            print(f"  ラベル: {label} ({dataset.index_to_key(label)})")
        
        # データローダーの作成テスト
        train_loader, val_loader = create_data_loaders(data_dir, batch_size=16)
        
    else:
        print(f"データディレクトリが存在しません: {data_dir}")
        print("データ収集を先に実行してください")
