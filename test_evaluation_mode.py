#!/usr/bin/env python3
"""
評価モードのテストスクリプト
"""

import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.modes.evaluation_mode import EvaluationMode


def test_evaluation_mode():
    """評価モードのテスト"""
    
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
    
    print(f"🔍 使用するモデル: {model_path}")
    
    # 評価モードの初期化
    try:
        evaluator = EvaluationMode(model_path)
        
        # 簡単なテストテキスト
        test_texts = ["hello", "test"]
        
        print("🧪 評価モードのテストを開始します")
        print("   テストテキスト:", test_texts)
        print("   被験者ID: TEST")
        
        # 評価セッション実行
        success = evaluator.run_evaluation_session(
            participant_id="TEST",
            target_texts=test_texts
        )
        
        if success:
            print("✅ テストが正常に完了しました")
            return True
        else:
            print("❌ テストが中断されました")
            return False
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False


if __name__ == "__main__":
    print("🚀 評価モードのテストを開始します")
    print("=" * 50)
    
    success = test_evaluation_mode()
    
    print("=" * 50)
    if success:
        print("✅ テスト完了")
    else:
        print("❌ テスト失敗")
