"""
特徴量重要度分析スクリプト
学習済みLSTMモデルの特徴量重要度をPermutation Importanceで分析
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score
import json
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.processing.data_loader import create_data_loaders
from src.processing.models.hand_lstm import BasicHandLSTM


class FeatureImportanceAnalyzer:
    """特徴量重要度分析クラス"""
    
    def __init__(self, model_path: str, data_dir: str, output_dir: str = "analysis_results/"):
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # 出力ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # モデルとデータの初期化
        self.model = None
        self.test_loader = None
        self.feature_names = []
        self.feature_dim = 0
        self.use_variable_length = False  # 可変長対応フラグ
        
        print(f"[INFO] 特徴量重要度分析を初期化")
        print(f"   モデルパス: {model_path}")
        print(f"   データディレクトリ: {data_dir}")
        print(f"   出力ディレクトリ: {output_dir}")
    
    def load_model(self):
        """モデルを読み込み、特徴量数を自動検出"""
        print(f"[INFO] モデルを読み込み中...")
        
        # モデルファイルの存在確認
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
        
        # モデルを読み込み
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # モデル設定を読み込み（可変長対応かどうかを確認）
        model_config = checkpoint.get('model_config', {})
        self.use_variable_length = model_config.get('use_variable_length', False)
        model_type = model_config.get('model_type', 'lstm')
        
        # モデルの構造を確認
        if 'model_state_dict' in checkpoint:
            # 学習済みモデルの場合
            model_state = checkpoint['model_state_dict']
            # 入力サイズを取得（model_configから、または重みから）
            if 'input_size' in model_config:
                self.feature_dim = model_config['input_size']
            else:
                # LSTMの入力サイズを取得（後方互換性）
                lstm_weight_key = None
                for key in model_state.keys():
                    if 'lstm.weight_ih_l0' in key or 'gru.weight_ih_l0' in key:
                        lstm_weight_key = key
                        break
                
                if lstm_weight_key:
                    lstm_input_weight = model_state[lstm_weight_key]
                    self.feature_dim = lstm_input_weight.shape[1]
                else:
                    # CNN/TCNの場合
                    for key in model_state.keys():
                        if 'conv1.weight' in key or 'network.0.conv1.weight' in key:
                            conv_weight = model_state[key]
                            self.feature_dim = conv_weight.shape[1]
                            break
                    else:
                        raise ValueError("モデルの入力サイズを検出できません")
        else:
            # 直接モデル状態の場合（後方互換性）
            lstm_weight_key = None
            for key in checkpoint.keys():
                if 'lstm.weight_ih_l0' in key:
                    lstm_weight_key = key
                    break
            
            if lstm_weight_key:
                lstm_input_weight = checkpoint[lstm_weight_key]
                self.feature_dim = lstm_input_weight.shape[1]
            else:
                raise ValueError("LSTMの入力重みが見つかりません")
        
        # モデルの構造を自動検出
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
        
        # モデルタイプに応じてモデルを作成
        if model_type in ['cnn', 'gru', 'lstm', 'tcn']:
            # 新しいモデル構造（可変長対応）
            from src.processing.models.hand_models import create_model
            input_size = model_config.get('input_size', self.feature_dim)
            hidden_size = model_config.get('hidden_size', 128)
            num_classes = model_config.get('num_classes', 37)
            
            model_params = {
                'model_type': model_type,
                'input_size': input_size,
                'num_classes': num_classes
            }
            
            if model_type in ['gru', 'lstm']:
                model_params['hidden_size'] = hidden_size
            
            self.model = create_model(**model_params)
        else:
            # 古いモデル（BasicHandLSTM）- 後方互換性
            # LSTMの隠れ層サイズを自動検出
            lstm_hidden_key = None
            for key in model_state.keys():
                if 'lstm.weight_hh_l0' in key:
                    lstm_hidden_key = key
                    break
            
            if lstm_hidden_key:
                lstm_hidden_weight = model_state[lstm_hidden_key]
                hidden_size = lstm_hidden_weight.shape[1]
            else:
                hidden_size = 64  # デフォルト値
            
            print(f"   検出された隠れ層サイズ: {hidden_size}")
            
            self.model = BasicHandLSTM(
                input_size=self.feature_dim,
                hidden_size=hidden_size,
                num_layers=2,
                num_classes=37,
                dropout=0.2
            )
        
        # モデルに重みを読み込み
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        print(f"[OK] モデル読み込み完了")
        print(f"   モデルタイプ: {model_type}")
        print(f"   可変長対応: {'有効' if self.use_variable_length else '無効'}")
        print(f"   特徴量次元数: {self.feature_dim}")
        
        # 特徴量名を自動設定
        self._set_feature_names()
    
    def _set_feature_names(self):
        """特徴量名を自動設定"""
        if self.feature_dim == 18:
            self.feature_names = [
                'finger_x', 'finger_y',  # 0-1
                'rel_key1_x', 'rel_key1_y', 'rel_key2_x', 'rel_key2_y', 'rel_key3_x', 'rel_key3_y',  # 2-7
                'dist_key1', 'dist_key2', 'dist_key3',  # 8-10
                'vel_x', 'vel_y', 'acc_x', 'acc_y',  # 11-14
                'amplitude_x', 'amplitude_y', 'direction_change'  # 15-17
            ]
        elif self.feature_dim == 30:
            self.feature_names = [
                # 0-17: 基本特徴量
                'finger_x', 'finger_y',  # 0-1
                'rel_key1_x', 'rel_key1_y', 'rel_key2_x', 'rel_key2_y', 'rel_key3_x', 'rel_key3_y',  # 2-7
                'dist_key1', 'dist_key2', 'dist_key3',  # 8-10
                'vel_x', 'vel_y', 'acc_x', 'acc_y',  # 11-14
                'amplitude_x', 'amplitude_y', 'direction_change',  # 15-17
                # 18-29: 新規特徴量
                'elapsed_time', 'target_angle', 'velocity_angle', 'angle_to_target',  # 18-21
                'speed', 'acceleration_magnitude', 'jerk', 'trajectory_length',  # 22-25
                'approach_velocity', 'trajectory_curvature', 'speed_std', 'velocity_consistency'  # 26-29
            ]
        else:
            # その他の次元数の場合は番号で命名
            self.feature_names = [f'feature_{i}' for i in range(self.feature_dim)]
        
        print(f"[INFO] 特徴量名を設定: {len(self.feature_names)}個")
    
    def load_test_data(self):
        """テストデータを読み込み"""
        print(f"[INFO] テストデータを読み込み中...")
        print(f"   可変長対応: {'有効' if self.use_variable_length else '無効'}")
        
        try:
            train_loader, val_loader, test_loader = create_data_loaders(
                self.data_dir, 
                batch_size=32,
                augment=False,  # 分析時は拡張なし
                use_variable_length=self.use_variable_length  # 可変長対応
            )
            self.test_loader = test_loader
            
            print(f"[OK] テストデータ読み込み完了")
            print(f"   テストサンプル数: {len(test_loader.dataset)}")
            
        except Exception as e:
            raise RuntimeError(f"テストデータの読み込みに失敗: {e}")
    
    def calculate_baseline_accuracy(self) -> float:
        """ベースライン精度を計算"""
        print(f"[INFO] ベースライン精度を計算中...")
        
        all_predictions = []
        all_labels = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        with torch.no_grad():
            for batch in self.test_loader:
                # 可変長対応：batchは(features, labels)または(features, labels, lengths)
                if self.use_variable_length:
                    features, labels, lengths = batch
                    lengths = lengths.to(device)
                else:
                    features, labels = batch
                    lengths = None
                
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = self.model(features, lengths)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        baseline_accuracy = accuracy_score(all_labels, all_predictions)
        print(f"[OK] ベースライン精度: {baseline_accuracy:.4f}")
        
        return baseline_accuracy
    
    def calculate_permutation_importance(self, n_permutations: int = 5) -> Dict[str, float]:
        """Permutation Importanceを計算"""
        print(f"[INFO] Permutation Importanceを計算中...")
        print(f"   置換回数: {n_permutations}")
        
        # ベースライン精度を計算
        baseline_accuracy = self.calculate_baseline_accuracy()
        
        # 全テストデータを一度に読み込み
        all_features = []
        all_labels = []
        
        for features, labels in self.test_loader:
            all_features.append(features)
            all_labels.append(labels)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"   総サンプル数: {len(all_features)}")
        
        # 各特徴量の重要度を計算
        importance_scores = {}
        
        for feature_idx in range(self.feature_dim):
            feature_name = self.feature_names[feature_idx]
            print(f"   特徴量 {feature_idx+1}/{self.feature_dim}: {feature_name}")
            
            # 複数回の置換で平均を取る
            accuracy_drops = []
            
            for perm_idx in range(n_permutations):
                # 特徴量をシャッフル
                shuffled_features = all_features.clone()
                shuffle_indices = torch.randperm(len(shuffled_features))
                shuffled_features[:, :, feature_idx] = shuffled_features[shuffle_indices, :, feature_idx]
                
                # 精度を計算
                with torch.no_grad():
                    outputs = self.model(shuffled_features)
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = accuracy_score(all_labels.cpu().numpy(), predictions.cpu().numpy())
                
                # 精度低下を計算
                accuracy_drop = baseline_accuracy - accuracy
                accuracy_drops.append(accuracy_drop)
            
            # 平均精度低下を重要度とする
            importance_scores[feature_name] = np.mean(accuracy_drops)
        
        print(f"[OK] Permutation Importance計算完了")
        
        return importance_scores
    
    def visualize_importance(self, importance_scores: Dict[str, float]):
        """重要度を可視化"""
        print(f"[INFO] 重要度を可視化中...")
        
        # 重要度でソート
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        feature_names = [item[0] for item in sorted_features]
        importance_values = [item[1] for item in sorted_features]
        
        # 棒グラフを作成
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(feature_names)), importance_values, 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        # 各バーに数値を表示
        for i, (bar, value) in enumerate(zip(bars, importance_values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('特徴量', fontsize=12)
        plt.ylabel('重要度 (精度低下)', fontsize=12)
        plt.title(f'特徴量重要度 (Permutation Importance)\n特徴量次元: {self.feature_dim}', fontsize=14)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存
        output_file = os.path.join(self.output_dir, 'importance.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] 重要度グラフを保存: {output_file}")
    
    def save_results(self, importance_scores: Dict[str, float]):
        """結果をCSVで保存"""
        print(f"[INFO] 結果をCSVで保存中...")
        
        # 重要度でソート
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # DataFrameを作成
        df = pd.DataFrame(sorted_features, columns=['feature_name', 'importance'])
        df['rank'] = range(1, len(df) + 1)
        
        # CSVで保存
        output_file = os.path.join(self.output_dir, 'importance.csv')
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"[OK] 結果をCSVで保存: {output_file}")
        
        return df
    
    def print_top_features(self, df: pd.DataFrame, top_n: int = 10):
        """上位特徴量をコンソール表示"""
        print(f"\n[INFO] 上位{top_n}個の重要特徴量:")
        print("=" * 50)
        
        for i, row in df.head(top_n).iterrows():
            print(f"{row['rank']:2d}. {row['feature_name']:20s} : {row['importance']:.4f}")
        
        print("=" * 50)
    
    def save_analysis_info(self, importance_scores: Dict[str, float]):
        """分析情報をJSONで保存"""
        analysis_info = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_dir': self.data_dir,
            'feature_dim': self.feature_dim,
            'feature_names': self.feature_names,
            'total_features': len(importance_scores),
            'importance_scores': importance_scores
        }
        
        output_file = os.path.join(self.output_dir, 'analysis_info.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_info, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] 分析情報を保存: {output_file}")
    
    def run_analysis(self, n_permutations: int = 5, top_n: int = 10):
        """分析を実行"""
        print(f"\n[INFO] 特徴量重要度分析を開始")
        print("=" * 60)
        
        try:
            # 1. モデル読み込み
            self.load_model()
            
            # 2. テストデータ読み込み
            self.load_test_data()
            
            # 3. Permutation Importance計算
            importance_scores = self.calculate_permutation_importance(n_permutations)
            
            # 4. 可視化
            self.visualize_importance(importance_scores)
            
            # 5. 結果保存
            df = self.save_results(importance_scores)
            
            # 6. 上位特徴量表示
            self.print_top_features(df, top_n)
            
            # 7. 分析情報保存
            self.save_analysis_info(importance_scores)
            
            print(f"\n[OK] 特徴量重要度分析完了!")
            print(f"   結果保存先: {self.output_dir}")
            
        except Exception as e:
            print(f"\n[ERROR] 分析中にエラーが発生: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='特徴量重要度分析')
    parser.add_argument('--model', type=str, required=True,
                       help='モデルパス（.pthファイル）')
    parser.add_argument('--data', type=str, required=True,
                       help='データディレクトリ')
    parser.add_argument('--output', type=str, default='analysis_results/',
                       help='結果保存先ディレクトリ（デフォルト: analysis_results/）')
    parser.add_argument('--permutations', type=int, default=5,
                       help='置換回数（デフォルト: 5）')
    parser.add_argument('--top', type=int, default=10,
                       help='表示する上位特徴量数（デフォルト: 10）')
    
    args = parser.parse_args()
    
    # 分析実行
    analyzer = FeatureImportanceAnalyzer(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output
    )
    
    analyzer.run_analysis(
        n_permutations=args.permutations,
        top_n=args.top
    )


if __name__ == "__main__":
    main()
