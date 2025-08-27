#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実践的な学習スクリプト
収集したデータを使ってLSTMモデルを学習
"""

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 既存のモジュールをインポート
import sys
sys.path.append('src')
from src.processing.data_loader import create_data_loaders, KeyboardIntentDataset
from src.processing.models.hand_lstm import BasicHandLSTM


class IntentModelTrainer:
    """意図推定モデルの学習クラス"""
    
    def __init__(self, data_dir: str, epochs: int = 100, batch_size: int = 32,
                 learning_rate: float = 0.001, model_save_path: str = None):
        """
        学習クラスの初期化
        
        Args:
            data_dir: データディレクトリのパス
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            model_save_path: モデル保存先パス
        """
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # モデル保存先の設定
        if model_save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_save_path = f"models/intent_model_{timestamp}"
        else:
            self.model_save_path = model_save_path
        
        # ディレクトリの作成
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # デバイスの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用デバイス: {self.device}")
        
        # コンポーネントの初期化
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # 学習履歴
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_top3_accuracies = []
        self.val_top3_accuracies = []
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=f"runs/intent_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Early Stopping
        self.best_val_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0
        
        print(f"🎯 意図推定モデル学習クラス初期化完了")
        print(f"   データディレクトリ: {data_dir}")
        print(f"   エポック数: {epochs}")
        print(f"   バッチサイズ: {batch_size}")
        print(f"   学習率: {learning_rate}")
        print(f"   モデル保存先: {self.model_save_path}")
    
    def setup_data(self):
        """データセットとデータローダーの設定"""
        print("📊 データセットの設定中...")
        
        try:
            # データローダーの作成
            self.train_loader, self.val_loader = create_data_loaders(
                data_dir=self.data_dir,
                batch_size=self.batch_size,
                augment=True,
                num_workers=0
            )
            
            if len(self.train_loader) == 0 or len(self.val_loader) == 0:
                raise ValueError("データローダーが空です")
            
            # データセットの情報を取得
            train_dataset = self.train_loader.dataset
            val_dataset = self.val_loader.dataset
            
            print(f"✅ データセット設定完了")
            print(f"   訓練データ: {len(train_dataset)} サンプル")
            print(f"   検証データ: {len(val_dataset)} サンプル")
            print(f"   特徴量次元: {train_dataset.feature_dim}")
            print(f"   時系列長: {train_dataset.sequence_length}")
            print(f"   クラス数: {train_dataset.num_classes}")
            
            return True
            
        except Exception as e:
            print(f"❌ データセット設定エラー: {e}")
            return False
    
    def setup_model(self):
        """モデル、損失関数、オプティマイザの設定"""
        print("🤖 モデルの設定中...")
        
        try:
            # データセットから情報を取得
            train_dataset = self.train_loader.dataset
            
            # モデルの初期化
            self.model = BasicHandLSTM(
                input_size=train_dataset.feature_dim,
                hidden_size=128,
                num_classes=train_dataset.num_classes
            ).to(self.device)
            
            # モデルパラメータ数の計算
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"📊 モデル情報:")
            print(f"   総パラメータ数: {total_params:,}")
            print(f"   学習可能パラメータ数: {trainable_params:,}")
            
            # 損失関数
            self.criterion = nn.CrossEntropyLoss()
            
            # オプティマイザ
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # 学習率スケジューラー
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            print(f"✅ モデル設定完了")
            return True
            
        except Exception as e:
            print(f"❌ モデル設定エラー: {e}")
            return False
    
    def train_epoch(self, epoch: int):
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        correct_top3_predictions = 0
        total_samples = 0
        
        # プログレスバーの表示
        num_batches = len(self.train_loader)
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # データをデバイスに移動
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 勾配をゼロにリセット
            self.optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # 逆伝播
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            self.optimizer.step()
            
            # 統計情報の更新
            total_loss += loss.item()
            total_samples += labels.size(0)
            
            # 精度の計算
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            
            # Top-3精度の計算
            _, top3_indices = torch.topk(outputs.data, 3, dim=1)
            correct_top3 = torch.sum(top3_indices == labels.unsqueeze(1), dim=1)
            correct_top3_predictions += correct_top3.sum().item()
            
            # プログレスバーの更新
            if batch_idx % 10 == 0:
                progress = (batch_idx + 1) / num_batches
                bar_length = 20
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\rEpoch {epoch}/{self.epochs} [{bar}] {progress*100:.1f}%", end='')
        
        print()  # 改行
        
        # 平均値を計算
        avg_loss = total_loss / num_batches
        accuracy = (correct_predictions / total_samples) * 100
        top3_accuracy = (correct_top3_predictions / total_samples) * 100
        
        return avg_loss, accuracy, top3_accuracy
    
    def validate_epoch(self, epoch: int):
        """1エポックの検証"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        correct_top3_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                # データをデバイスに移動
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # 順伝播
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # 統計情報の更新
                total_loss += loss.item()
                total_samples += labels.size(0)
                
                # 精度の計算
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                
                # Top-3精度の計算
                _, top3_indices = torch.topk(outputs.data, 3, dim=1)
                correct_top3 = torch.sum(top3_indices == labels.unsqueeze(1), dim=1)
                correct_top3_predictions += correct_top3.sum().item()
                
                # 混同行列用のデータを保存
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 平均値を計算
        avg_loss = total_loss / len(self.val_loader)
        accuracy = (correct_predictions / total_samples) * 100
        top3_accuracy = (correct_top3_predictions / total_samples) * 100
        
        return avg_loss, accuracy, top3_accuracy, all_predictions, all_labels
    
    def train(self):
        """学習の実行"""
        print("\n🚀 学習を開始します")
        print("=" * 60)
        
        # データとモデルの設定
        if not self.setup_data() or not self.setup_model():
            print("❌ 初期化に失敗しました")
            return False
        
        # 学習開始時刻
        start_time = time.time()
        
        print(f"\n📊 学習設定:")
        print(f"   データサンプル数: {len(self.train_loader.dataset) + len(self.val_loader.dataset)}")
        print(f"   訓練/検証分割: {len(self.train_loader.dataset)}/{len(self.val_loader.dataset)}")
        print(f"   モデルパラメータ数: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        try:
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()
                
                # 訓練
                train_loss, train_acc, train_top3_acc = self.train_epoch(epoch)
                
                # 検証
                val_loss, val_acc, val_top3_acc, predictions, labels = self.validate_epoch(epoch)
                
                # 学習履歴の保存
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
                self.train_top3_accuracies.append(train_top3_acc)
                self.val_top3_accuracies.append(val_top3_acc)
                
                # TensorBoardへの記録
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
                self.writer.add_scalar('Top3_Accuracy/Train', train_top3_acc, epoch)
                self.writer.add_scalar('Top3_Accuracy/Validation', val_top3_acc, epoch)
                
                # エポック時間の計算
                epoch_time = time.time() - epoch_start_time
                
                # 結果の表示
                print(f"Epoch {epoch:3d}/{self.epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.1f}%, Top-3: {val_top3_acc:.1f}% "
                      f"({epoch_time:.1f}s)")
                
                # 学習率スケジューラーの更新
                self.scheduler.step(val_loss)
                
                # Early Stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # ベストモデルの保存
                    self.save_model("best_model.pth", epoch, val_loss, val_acc)
                    print(f"💾 ベストモデルを保存しました (Val Loss: {val_loss:.4f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"⏹️  Early Stopping: {self.patience}エポック改善なし")
                        break
                
                # 定期的なモデル保存
                if epoch % 10 == 0:
                    self.save_model(f"checkpoint_epoch_{epoch}.pth", epoch, val_loss, val_acc)
            
            # 学習完了
            total_time = time.time() - start_time
            print(f"\n✅ 学習完了！")
            print(f"   総学習時間: {total_time/60:.1f}分")
            print(f"   最終検証精度: {val_acc:.1f}%")
            print(f"   最終検証Top-3精度: {val_top3_acc:.1f}%")
            
            # 最終結果の保存
            self.save_final_results(predictions, labels)
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n⚠️ 学習がユーザーによって中断されました")
            return False
        except Exception as e:
            print(f"\n❌ 学習エラー: {e}")
            return False
        finally:
            self.cleanup()
    
    def save_model(self, filename: str, epoch: int, val_loss: float, val_acc: float):
        """モデルの保存"""
        filepath = os.path.join(self.model_save_path, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_top3_accuracies': self.train_top3_accuracies,
            'val_top3_accuracies': self.val_top3_accuracies,
            'model_config': {
                'input_size': self.train_loader.dataset.feature_dim,
                'hidden_size': 128,
                'num_classes': self.train_loader.dataset.num_classes,
                'sequence_length': self.train_loader.dataset.sequence_length
            },
            'training_config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'data_dir': self.data_dir
            }
        }, filepath)
    
    def save_final_results(self, predictions: list, labels: list):
        """最終結果の保存"""
        print("💾 最終結果を保存中...")
        
        # 混同行列の計算
        cm = confusion_matrix(labels, predictions)
        
        # 分類レポートの生成
        dataset = self.val_loader.dataset
        target_names = [dataset.index_to_key(i) for i in range(dataset.num_classes)]
        report = classification_report(labels, predictions, target_names=target_names, output_dict=True)
        
        # 結果の保存
        results = {
            'final_metrics': {
                'final_val_loss': self.val_losses[-1] if self.val_losses else None,
                'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None,
                'final_val_top3_accuracy': self.val_top3_accuracies[-1] if self.val_top3_accuracies else None,
                'best_val_loss': self.best_val_loss,
                'total_epochs': len(self.train_losses)
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'learning_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'train_top3_accuracies': self.train_top3_accuracies,
                'val_top3_accuracies': self.val_top3_accuracies
            }
        }
        
        # JSONファイルとして保存
        results_file = os.path.join(self.model_save_path, 'training_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 混同行列の可視化
        self.plot_confusion_matrix(cm, target_names)
        
        # 学習曲線の可視化
        self.plot_learning_curves()
        
        print(f"✅ 最終結果を保存しました: {results_file}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, target_names: list):
        """混同行列の可視化"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存
        cm_file = os.path.join(self.model_save_path, 'confusion_matrix.png')
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 混同行列を保存しました: {cm_file}")
    
    def plot_learning_curves(self):
        """学習曲線の可視化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 損失曲線
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 精度曲線
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Top-3精度曲線
        ax3.plot(epochs, self.train_top3_accuracies, 'b-', label='Training Top-3 Accuracy')
        ax3.plot(epochs, self.val_top3_accuracies, 'r-', label='Validation Top-3 Accuracy')
        ax3.set_title('Training and Validation Top-3 Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Top-3 Accuracy (%)')
        ax3.legend()
        ax3.grid(True)
        
        # 学習率の推移
        ax4.plot(epochs, [self.learning_rate] * len(epochs), 'g-', label='Learning Rate')
        ax4.set_title('Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存
        curves_file = os.path.join(self.model_save_path, 'learning_curves.png')
        plt.savefig(curves_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 学習曲線を保存しました: {curves_file}")
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        if self.writer:
            self.writer.close()
        print("🧹 リソースのクリーンアップ完了")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='意図推定モデルの学習スクリプト')
    parser.add_argument('--data-dir', default='data/training/user_001', 
                       help='データディレクトリ (default: data/training/user_001)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='エポック数 (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='バッチサイズ (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001, 
                       help='学習率 (default: 0.001)')
    parser.add_argument('--model-save-path', default=None, 
                       help='モデル保存先パス (default: auto-generated)')
    
    args = parser.parse_args()
    
    # データディレクトリの存在確認
    if not os.path.exists(args.data_dir):
        print(f"❌ データディレクトリが存在しません: {args.data_dir}")
        print("データ収集を先に実行してください")
        return 1
    
    print("🎯 Training Intent Prediction Model")
    print("=" * 50)
    
    # 学習クラスの作成と実行
    trainer = IntentModelTrainer(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=args.model_save_path
    )
    
    # 学習の実行
    success = trainer.train()
    
    if success:
        print("\n🎉 学習が正常に完了しました！")
        print(f"📁 モデルと結果は以下に保存されています:")
        print(f"   {trainer.model_save_path}")
        print(f"🌐 TensorBoardログ: {trainer.writer.log_dir}")
        return 0
    else:
        print("\n❌ 学習に失敗しました")
        return 1


if __name__ == "__main__":
    exit(main())
