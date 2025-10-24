#!/usr/bin/env python3
"""
オフライン評価スクリプト
学習済みモデルをテストセット（学習時未使用）で評価し、真の性能を測定する
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.processing.data_loader import KeyboardIntentDataset
from src.processing.models.hand_lstm import BasicHandLSTM
from torch.utils.data import DataLoader
from config.feature_config import get_feature_dim


def evaluate_on_testset(model_path: str, data_dir: str, 
                       output_dir: str = None):
    """
    テストセットでのオフライン評価
    
    Args:
        model_path: 学習済みモデルのパス（.pthファイル）
        data_dir: データディレクトリ（data/training）
        output_dir: 結果保存先ディレクトリ
    
    処理:
    1. テストセット用データローダーを作成
    2. モデルを読み込み
    3. テストセットで評価
       - Top-1精度
       - Top-3精度
       - クラス別精度
       - 混同行列
    4. 結果をJSON・PNGで保存
    5. サマリーをコンソール表示
    """
    
    # モデル名からディレクトリ名を生成
    if output_dir is None:
        model_dir = os.path.basename(os.path.dirname(model_path))
        output_dir = f'evaluation_results/offline/{model_dir}'
    
    print("🚀 オフライン評価を開始します")
    print(f"   モデル: {model_path}")
    print(f"   データ: {data_dir}")
    print(f"   出力先: {output_dir}")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. テストデータセットの作成
        print("\n📊 テストデータセットを作成中...")
        test_dataset = KeyboardIntentDataset(
            data_dir=data_dir,
            sequence_length=60,
            split_mode='test',  # testモード
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            augment=False  # テスト時は拡張なし
        )
        
        if len(test_dataset) == 0:
            print("❌ テストデータが存在しません")
            return False
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ テストデータセット作成完了: {len(test_dataset)} サンプル")
        
        # 2. モデルの読み込み
        print("\n🤖 モデルを読み込み中...")
        model = load_model(model_path)
        if model is None:
            return False
        
        # デバイスの設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print(f"✅ モデル読み込み完了: {device}")
        
        # 3. 評価ループ
        print("\n🧪 テストセットで評価中...")
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print(f"   バッチ {batch_idx + 1}/{len(test_loader)}")
                
                # デバイスに移動
                features = features.to(device)
                labels = labels.to(device)
                
                # 順伝播
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Top-1予測
                _, predicted = torch.max(outputs, 1)
                
                # 記録
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        print("✅ 評価完了")
        
        # 4. 精度計算
        print("\n📈 精度を計算中...")
        metrics = calculate_metrics(all_labels, all_predictions, all_probs, test_dataset)
        
        # 5. 可視化
        print("\n🎨 可視化を作成中...")
        create_visualizations(all_labels, all_predictions, test_dataset, output_dir)
        
        # 6. 結果保存
        print("\n💾 結果を保存中...")
        save_results(metrics, all_labels, all_predictions, test_dataset, 
                    model_path, data_dir, output_dir)
        
        # 7. サマリー表示
        show_summary(metrics, len(test_dataset), output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ 評価エラー: {e}")
        return False


def load_model(model_path: str):
    """学習済みモデルの読み込み"""
    try:
        # チェックポイント読み込み
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('model_config', {})
        
        # モデル初期化
        model = BasicHandLSTM(
            input_size=model_config.get('input_size', get_feature_dim()),
            hidden_size=model_config.get('hidden_size', 128),
            num_classes=model_config.get('num_classes', 37)
        )
        
        # 重み読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return None


def calculate_metrics(all_labels: List[int], all_predictions: List[int], 
                     all_probs: List[np.ndarray], test_dataset) -> Dict:
    """精度指標の計算"""
    
    # Top-1精度
    top1_accuracy = accuracy_score(all_labels, all_predictions) * 100
    
    # Top-3精度
    top3_correct = 0
    for i, label in enumerate(all_labels):
        top3_indices = np.argsort(all_probs[i])[-3:]
        if label in top3_indices:
            top3_correct += 1
    top3_accuracy = (top3_correct / len(all_labels)) * 100
    
    # 混同行列
    cm = confusion_matrix(all_labels, all_predictions)
    
    # クラス別精度
    report = classification_report(
        all_labels, all_predictions,
        target_names=[test_dataset.index_to_key(i) for i in range(37)],
        output_dict=True,
        zero_division=0
    )
    
    return {
        'top1_accuracy': round(top1_accuracy, 2),
        'top3_accuracy': round(top3_accuracy, 2),
        'confusion_matrix': cm,
        'classification_report': report
    }


def create_visualizations(all_labels: List[int], all_predictions: List[int], 
                         test_dataset, output_dir: str):
    """可視化の作成"""
    
    # 混同行列のプロット
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[test_dataset.index_to_key(i) for i in range(37)],
               yticklabels=[test_dataset.index_to_key(i) for i in range(37)])
    plt.title('Confusion Matrix (Test Set)', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # クラス別精度のプロット
    report = classification_report(
        all_labels, all_predictions,
        target_names=[test_dataset.index_to_key(i) for i in range(37)],
        output_dict=True,
        zero_division=0
    )
    
    class_accuracies = []
    class_names = []
    for i in range(37):
        key = test_dataset.index_to_key(i)
        class_names.append(key)
        acc = report[key]['recall'] * 100 if key in report else 0
        class_accuracies.append(acc)
    
    plt.figure(figsize=(18, 8))
    bars = plt.bar(range(37), class_accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 各バーに数値を表示
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy (Test Set)', fontsize=16)
    plt.xticks(range(37), class_names, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 可視化ファイルを作成しました")


def save_results(metrics: Dict, all_labels: List[int], all_predictions: List[int], 
                test_dataset, model_path: str, data_dir: str, output_dir: str):
    """結果の保存"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 結果データの構築
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'data_dir': data_dir,
        'test_samples': len(test_dataset),
        'metrics': {
            'top1_accuracy': metrics['top1_accuracy'],
            'top3_accuracy': metrics['top3_accuracy']
        },
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classification_report': metrics['classification_report']
    }
    
    # JSON保存
    json_path = f'{output_dir}/offline_evaluation_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 結果を保存: {json_path}")


def show_summary(metrics: Dict, test_samples: int, output_dir: str):
    """サマリーの表示"""
    
    print("\n" + "="*60)
    print("📊 オフライン評価結果（テストセット）")
    print("="*60)
    print(f"テストサンプル数: {test_samples}")
    print(f"Top-1精度:        {metrics['top1_accuracy']:.2f}%")
    print(f"Top-3精度:        {metrics['top3_accuracy']:.2f}%")
    print("="*60)
    
    # クラス別精度の上位・下位を表示
    report = metrics['classification_report']
    class_accuracies = []
    for i in range(37):
        key = f"{'a' if i < 26 else '0' if i < 36 else ' '}"
        if key in report:
            class_accuracies.append((key, report[key]['recall'] * 100))
    
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 クラス別精度（上位5位）:")
    for i, (key, acc) in enumerate(class_accuracies[:5]):
        print(f"   {i+1}. {key}: {acc:.2f}%")
    
    print(f"\n📉 クラス別精度（下位5位）:")
    for i, (key, acc) in enumerate(class_accuracies[-5:]):
        print(f"   {len(class_accuracies)-4+i}. {key}: {acc:.2f}%")
    
    print("="*60)
    print(f"\n📁 保存されたファイル:")
    print(f"   詳細結果: {output_dir}/offline_evaluation_*.json")
    print(f"   混同行列: {output_dir}/confusion_matrix.png")
    print(f"   クラス別精度: {output_dir}/per_class_accuracy.png")
    print("="*60)


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='テストセットでのオフライン評価')
    parser.add_argument('--model', type=str, required=True,
                        help='モデルパス（.pthファイル）')
    parser.add_argument('--data-dir', type=str, default='data/training',
                        help='データディレクトリ')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='結果保存先（指定しない場合はモデル名で自動生成）')
    
    args = parser.parse_args()
    
    # モデルファイルの存在確認
    if not os.path.exists(args.model):
        print(f"❌ モデルファイルが存在しません: {args.model}")
        return False
    
    # データディレクトリの存在確認
    if not os.path.exists(args.data_dir):
        print(f"❌ データディレクトリが存在しません: {args.data_dir}")
        return False
    
    # 評価実行
    success = evaluate_on_testset(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n✅ オフライン評価が正常に完了しました")
        return True
    else:
        print("\n❌ オフライン評価が失敗しました")
        return False


if __name__ == "__main__":
    main()
