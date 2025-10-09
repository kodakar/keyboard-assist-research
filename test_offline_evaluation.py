#!/usr/bin/env python3
"""
オフライン評価のテストスクリプト
"""

import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_offline import evaluate_on_testset


def test_offline_evaluation():
    """オフライン評価のテスト"""
    
    # モデルパスの確認
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ modelsディレクトリが存在しません")
        print("学習を先に実行してください")
        return False
    
    # 最新モデルを検索
    model_dirs = [d for d in os.listdir(models_dir)
                  if d.startswith('intent_model_') and os.path.isdir(os.path.join(models_dir, d))]
    
    if not model_dirs:
        print("❌ 学習済みモデルが見つかりません")
        print("学習を先に実行してください")
        return False
    
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(models_dir, latest_model_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが存在しません: {model_path}")
        return False
    
    # データディレクトリの確認
    data_dir = "data/training"
    if not os.path.exists(data_dir):
        print(f"❌ データディレクトリが存在しません: {data_dir}")
        print("データ収集を先に実行してください")
        return False
    
    print(f"🔍 使用するモデル: {model_path}")
    print(f"🔍 データディレクトリ: {data_dir}")
    
    # オフライン評価の実行
    try:
        success = evaluate_on_testset(
            model_path=model_path,
            data_dir=data_dir,
            output_dir="evaluation_results/test_offline"
        )
        
        if success:
            print("✅ テストが正常に完了しました")
            return True
        else:
            print("❌ テストが失敗しました")
            return False
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False


if __name__ == "__main__":
    print("🚀 オフライン評価のテストを開始します")
    print("=" * 50)
    
    success = test_offline_evaluation()
    
    print("=" * 50)
    if success:
        print("✅ テスト完了")
    else:
        print("❌ テスト失敗")
