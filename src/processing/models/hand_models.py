# src/processing/models/hand_models.py
"""
各種手の軌跡認識モデル
可変長対応のモデルを提供
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from config.feature_config import get_feature_dim, get_num_classes


# ====================================
# 1. 1D-CNN（推奨）
# ====================================
class HandCNN(nn.Module):
    """
    1D-CNNモデル（可変長対応が簡単）
    
    畳み込みでローカルパターン（速度、加速度）を捉え、
    AdaptivePoolingで可変長を固定長に変換する。
    """
    
    def __init__(self, input_size=None, num_classes=None, dropout=0.2):
        super().__init__()
        self.input_size = input_size or get_feature_dim()
        self.num_classes = num_classes or get_num_classes()
        
        # 畳み込み層
        self.conv1 = nn.Conv1d(self.input_size, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # 可変長を固定長に変換（重要！）
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_size)
            lengths: (batch,) - 各系列の実際の長さ（使用しない）
        
        Returns:
            output: (batch, num_classes)
        """
        # (batch, seq_len, input_size) → (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # 畳み込み
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # 可変長を1に集約（lengths不要！）
        x = self.adaptive_pool(x).squeeze(-1)
        
        # 全結合
        output = self.fc(x)
        
        return output


# ====================================
# 2. GRU（LSTMより高速）
# ====================================
class HandGRU(nn.Module):
    """
    GRUモデル（LSTMの改良版）
    
    LSTMよりパラメータが少なく高速。
    同等の精度が期待できる。
    """
    
    def __init__(self, input_size=None, hidden_size=128, num_layers=2, 
                 num_classes=None, dropout=0.2):
        super().__init__()
        self.input_size = input_size or get_feature_dim()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes or get_num_classes()
        
        # GRU層
        self.gru = nn.GRU(
            self.input_size,
            self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_size)
            lengths: (batch,) - 各系列の実際の長さ
        
        Returns:
            output: (batch, num_classes)
        """
        if lengths is not None:
            # 可変長対応（PackedSequence）
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            # 固定長（後方互換）
            _, hidden = self.gru(x)
        
        # 最後の隠れ状態（最上層）
        output = self.fc(hidden[-1])
        
        return output


# ====================================
# 3. LSTM（可変長対応版）
# ====================================
class HandLSTM(nn.Module):
    """
    LSTMモデル（可変長対応版）
    
    現在のBasicHandLSTMを可変長対応に改良したもの。
    """
    
    def __init__(self, input_size=None, hidden_size=128, num_layers=2,
                 num_classes=None, dropout=0.2):
        super().__init__()
        self.input_size = input_size or get_feature_dim()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes or get_num_classes()
        
        # LSTM層
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_size)
            lengths: (batch,) - 各系列の実際の長さ
        
        Returns:
            output: (batch, num_classes)
        """
        if lengths is not None:
            # 可変長対応（PackedSequence）
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, cell) = self.lstm(packed)
        else:
            # 固定長（後方互換）
            _, (hidden, cell) = self.lstm(x)
        
        # 最後の隠れ状態（最上層）
        output = self.fc(hidden[-1])
        
        return output


# ====================================
# 4. TCN（Temporal Convolutional Network）
# ====================================
class TemporalBlock(nn.Module):
    """TCNの基本ブロック"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 残差接続用
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # パディング情報を保持
        self.padding = padding
    
    def forward(self, x):
        # 畳み込み
        out = self.conv1(x)
        
        # 因果的畳み込み（未来を見ない）
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 残差接続
        if self.residual is not None:
            res = self.residual(x)
        else:
            res = x
        
        # 残差接続のサイズ調整
        if res.shape[2] != out.shape[2]:
            res = res[:, :, :out.shape[2]]
        
        return self.relu(out + res)


class HandTCN(nn.Module):
    """
    TCNモデル（Temporal Convolutional Network）
    
    因果的畳み込みとdilationで長期依存を捉える。
    RNNより並列計算が可能で高速。
    """
    
    def __init__(self, input_size=None, num_channels=[64, 128, 256], 
                 kernel_size=3, num_classes=None, dropout=0.2):
        super().__init__()
        self.input_size = input_size or get_feature_dim()
        self.num_classes = num_classes or get_num_classes()
        
        # TCN層
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            in_channels = self.input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation = 2 ** i  # 指数的にdilationを増やす
            
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_size)
            lengths: (batch,) - 各系列の実際の長さ（使用しない）
        
        Returns:
            output: (batch, num_classes)
        """
        # (batch, seq_len, input_size) → (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # TCN処理
        x = self.network(x)
        
        # 可変長を1に集約
        x = self.adaptive_pool(x).squeeze(-1)
        
        # 全結合
        output = self.fc(x)
        
        return output


# ====================================
# モデル作成関数
# ====================================
def create_model(model_type='cnn', **kwargs):
    """
    モデルを作成する
    
    Args:
        model_type: 'cnn', 'gru', 'lstm', 'tcn'
        **kwargs: モデル固有のパラメータ
    
    Returns:
        モデルインスタンス
    
    Examples:
        >>> model = create_model('cnn', input_size=18, num_classes=37)
        >>> model = create_model('gru', hidden_size=128, num_layers=2)
        >>> model = create_model('lstm', dropout=0.3)
        >>> model = create_model('tcn', num_channels=[64, 128, 256])
    """
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return HandCNN(**kwargs)
    elif model_type == 'gru':
        return HandGRU(**kwargs)
    elif model_type == 'lstm':
        return HandLSTM(**kwargs)
    elif model_type == 'tcn':
        return HandTCN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: cnn, gru, lstm, tcn")


def get_model_info(model_type='cnn'):
    """
    モデルの情報を取得する
    
    Args:
        model_type: 'cnn', 'gru', 'lstm', 'tcn'
    
    Returns:
        モデル情報の辞書
    """
    info = {
        'cnn': {
            'name': '1D-CNN',
            'description': '畳み込みニューラルネットワーク',
            'pros': ['高速', '可変長対応が簡単', 'ローカルパターンに強い'],
            'cons': ['長期依存は限定的'],
            'best_for': '速度・加速度などのローカルパターン認識'
        },
        'gru': {
            'name': 'GRU',
            'description': 'Gated Recurrent Unit',
            'pros': ['LSTMより高速', 'LSTMより省メモリ', '同等の精度'],
            'cons': ['学習が逐次的'],
            'best_for': 'LSTMの代替として'
        },
        'lstm': {
            'name': 'LSTM',
            'description': 'Long Short-Term Memory',
            'pros': ['長期依存を学習', '実装が成熟'],
            'cons': ['学習が遅い', 'メモリ消費大'],
            'best_for': '系列全体の文脈が必要な場合'
        },
        'tcn': {
            'name': 'TCN',
            'description': 'Temporal Convolutional Network',
            'pros': ['並列計算可能', '長期依存を捉える', '最新手法'],
            'cons': ['実装がやや複雑'],
            'best_for': 'RNNの代替として（高速版）'
        }
    }
    
    return info.get(model_type.lower(), None)


# ====================================
# テスト
# ====================================
if __name__ == "__main__":
    print("🧪 モデルのテスト")
    print("=" * 50)
    
    batch_size = 8
    input_size = 18
    num_classes = 37
    
    # 可変長データの作成
    lengths = torch.tensor([15, 30, 45, 60, 20, 35, 50, 25])
    max_length = lengths.max().item()
    
    # パディング済みデータ
    x = torch.randn(batch_size, max_length, input_size)
    
    print(f"入力データ: {x.shape}")
    print(f"長さ: {lengths}")
    print()
    
    # 各モデルをテスト
    models = ['cnn', 'gru', 'lstm', 'tcn']
    
    for model_type in models:
        print(f"📦 {model_type.upper()} モデル")
        
        # モデル作成
        model = create_model(model_type, input_size=input_size, num_classes=num_classes)
        model.eval()
        
        # 推論
        with torch.no_grad():
            output = model(x, lengths)
        
        # パラメータ数
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"   出力: {output.shape}")
        print(f"   パラメータ数: {num_params:,}")
        
        # モデル情報
        info = get_model_info(model_type)
        if info:
            print(f"   特徴: {', '.join(info['pros'])}")
        
        print()
    
    print("✅ テスト完了")

