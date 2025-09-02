# src/processing/models/hand_lstm.py
"""
手の動きを学習する基本的なLSTMモデル
PyTorchを使用した実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pickle

class BasicHandLSTM(nn.Module):
    def __init__(self, 
                 input_size: int = 18,      # 18次元（作業領域特徴量）
                 hidden_size: int = 64,     # LSTM隠れ層サイズ
                 num_layers: int = 2,       # LSTM層数
                 num_classes: int = 37,     # 37キー（a-z, 0-9, space）
                 dropout: float = 0.2):     # ドロップアウト率
        """
        基本的な手の動き学習用LSTMモデル
        
        Args:
            input_size: 入力特徴量の次元数（作業領域特徴量、18次元）
            hidden_size: LSTM隠れ層のサイズ
            num_layers: LSTMの層数
            num_classes: 分類クラス数（キーの数）
            dropout: ドロップアウト率
        """
        super(BasicHandLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # モデル情報
        self.model_info = {
            'model_type': 'BasicHandLSTM',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'dropout': dropout,
            'created_at': datetime.now().isoformat()
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル (batch_size, sequence_length, input_size)
        
        Returns:
            出力テンソル (batch_size, num_classes)
        """
        # LSTMの出力を取得
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 最後の時刻の出力を使用
        last_output = lstm_out[:, -1, :]
        
        # 全結合層で分類
        output = self.fc(last_output)
        
        return output
    
    @torch.no_grad()
    def predict_key(self, hand_sequence: np.ndarray) -> Tuple[str, float]:
        """
        手の動きシーケンスからキーを予測
        
        Args:
            hand_sequence: 手の動きシーケンス (sequence_length, input_size)
        
        Returns:
            (予測キー, 確信度)
        """
        self.eval()
        
        # テンソルに変換
        x = torch.FloatTensor(hand_sequence).unsqueeze(0)  # (1, seq_len, input_size)
        
        # 予測
        output = self.forward(x)
        probabilities = torch.softmax(output, dim=1)
        
        # 最も確率の高いクラスを取得
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        # クラスIDをキーに変換
        predicted_key = self._class_id_to_key(predicted_class)
        
        return predicted_key, confidence
    
    def _class_id_to_key(self, class_id: int) -> str:
        """クラスIDをキー文字に変換"""
        if class_id < 26:
            return chr(ord('a') + class_id)  # a-z
        elif class_id < 36:
            return chr(ord('0') + class_id - 26)  # 0-9
        else:
            return 'space'  # スペース
    
    def _key_to_class_id(self, key: str) -> int:
        """キー文字をクラスIDに変換"""
        if key.isalpha() and key.islower():
            return ord(key) - ord('a')
        elif key.isdigit():
            return 26 + int(key)
        elif key == 'space':
            return 36
        else:
            return 0  # デフォルト
    
    def save_model(self, filepath: str):
        """モデルを保存"""
        try:
            # モデルの状態辞書を作成
            model_state = {
                'model_state_dict': self.state_dict(),
                'model_info': self.model_info,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_classes': self.num_classes
            }
            
            # PyTorch形式で保存
            torch.save(model_state, filepath)
            print(f"✅ モデルを保存しました: {filepath}")
            
        except Exception as e:
            print(f"⚠️ モデル保存エラー: {e}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BasicHandLSTM':
        """保存されたモデルを読み込み"""
        try:
            # モデルの状態辞書を読み込み
            model_state = torch.load(filepath, map_location='cpu')
            
            # モデルインスタンスを作成
            model = cls(
                input_size=model_state['input_size'],
                hidden_size=model_state['hidden_size'],
                num_layers=model_state['num_layers'],
                num_classes=model_state['num_classes']
            )
            
            # 重みを読み込み
            model.load_state_dict(model_state['model_state_dict'])
            model.model_info = model_state['model_info']
            
            print(f"✅ モデルを読み込みました: {filepath}")
            return model
            
        except Exception as e:
            print(f"⚠️ モデル読み込みエラー: {e}")
            raise


class HandLSTMTrainer:
    def __init__(self, model: BasicHandLSTM, learning_rate: float = 0.001):
        """
        LSTMモデルの学習器
        
        Args:
            model: 学習対象のLSTMモデル
            learning_rate: 学習率
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学習履歴
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def prepare_training_data(self, dataset: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        学習データを準備
        
        Args:
            dataset: 学習データセット
        
        Returns:
            (入力テンソル, ラベルテンソル)
        """
        X, y = [], []
        
        for sample in dataset:
            if 'trajectory_data' in sample and sample['trajectory_data']:
                # 軌跡データを取得
                trajectory = sample['trajectory_data']
                
                # ランドマークデータを抽出
                landmarks_sequence = []
                for point in trajectory:
                    if 'landmarks' in point:
                        landmarks_sequence.append(point['landmarks'])
                
                if len(landmarks_sequence) >= 10:  # 最小シーケンス長
                    # シーケンス長を統一（パディングまたは切り詰め）
                    if len(landmarks_sequence) > 30:
                        landmarks_sequence = landmarks_sequence[:30]
                    elif len(landmarks_sequence) < 30:
                        # パディング（最後のフレームを繰り返し）
                        last_frame = landmarks_sequence[-1]
                        while len(landmarks_sequence) < 30:
                            landmarks_sequence.append(last_frame)
                    
                    X.append(landmarks_sequence)
                    
                    # ラベルを準備
                    intended_key = sample['intended_key']
                    class_id = self.model._key_to_class_id(intended_key)
                    y.append(class_id)
        
        if not X:
            raise ValueError("有効な学習データが見つかりません")
        
        # テンソルに変換
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        print(f"📊 学習データ準備完了")
        print(f"   サンプル数: {len(X)}")
        print(f"   入力形状: {X_tensor.shape}")
        print(f"   ラベル形状: {y_tensor.shape}")
        
        return X_tensor, y_tensor
    
    def train(self, 
              train_data: Tuple[torch.Tensor, torch.Tensor],
              val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
              epochs: int = 100,
              batch_size: int = 32,
              early_stopping_patience: int = 10):
        """
        モデルの学習を実行
        
        Args:
            train_data: 学習データ (X, y)
            val_data: 検証データ (X, y)
            epochs: エポック数
            batch_size: バッチサイズ
            early_stopping_patience: 早期停止のパテンス
        """
        X_train, y_train = train_data
        
        # データローダーを作成
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # 検証データローダー
        val_loader = None
        if val_data:
            X_val, y_val = val_data
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        print(f"🚀 学習開始")
        print(f"   エポック数: {epochs}")
        print(f"   バッチサイズ: {batch_size}")
        print(f"   学習サンプル数: {len(X_train)}")
        if val_data:
            print(f"   検証サンプル数: {len(X_val)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 学習
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # 順伝播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 逆伝播
                loss.backward()
                self.optimizer.step()
                
                # 統計
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # 学習結果
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # 検証
            val_loss = 0.0
            val_accuracy = 0.0
            
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * val_correct / val_total
                
                # 早期停止チェック
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"⏹️ 早期停止: {epoch + 1}エポック目")
                    break
            
            # 履歴を記録
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            
            if val_loader:
                self.training_history['val_loss'].append(avg_val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
            
            # 進捗表示
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.2f}%", end="")
                
                if val_loader:
                    print(f" | Val Loss: {avg_val_loss:.4f} | "
                          f"Val Acc: {val_accuracy:.2f}%")
                else:
                    print()
        
        print(f"✅ 学習完了")
        return self.training_history
    
    def save_training_history(self, filepath: str):
        """学習履歴を保存"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.training_history, f)
            print(f"✅ 学習履歴を保存しました: {filepath}")
        except Exception as e:
            print(f"⚠️ 学習履歴保存エラー: {e}")
    
    def load_training_history(self, filepath: str):
        """学習履歴を読み込み"""
        try:
            with open(filepath, 'rb') as f:
                self.training_history = pickle.load(f)
            print(f"✅ 学習履歴を読み込みました: {filepath}")
        except Exception as e:
            print(f"⚠️ 学習履歴読み込みエラー: {e}")


def create_sample_model() -> BasicHandLSTM:
    """サンプルモデルを作成"""
    model = BasicHandLSTM(
        input_size=15,      # 15次元（作業領域特徴量）
        hidden_size=64,     # LSTM隠れ層サイズ
        num_layers=2,       # LSTM層数
        num_classes=37,     # 37キー
        dropout=0.2         # ドロップアウト率
    )
    
    print(f"✅ サンプルモデルを作成しました")
    print(f"   パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    return model
