#!/usr/bin/env python3
"""
学習システムの動作確認用テストスクリプト
"""

import os
import sys
import json
import numpy as np
import time
from datetime import datetime

# パスを追加
sys.path.append('src')

def test_enhanced_data_collector():
    """拡張データ収集システムのテスト"""
    print("🧪 拡張データ収集システムのテスト")
    print("=" * 50)
    
    try:
        from processing.enhanced_data_collector import EnhancedDataCollector
        
        # データ収集システムを作成
        collector = EnhancedDataCollector(
            trajectory_buffer_size=30,
            data_dir="test_data",
            user_id="test_user"
        )
        
        print("✅ EnhancedDataCollectorの作成に成功")
        
        # セッション開始
        collector.start_collection_session("test text")
        print("✅ データ収集セッション開始に成功")
        
        # ダミーの手のランドマークデータを作成
        dummy_landmarks = type('Landmarks', (), {
            'landmark': [type('Landmark', (), {
                'x': np.random.random(),
                'y': np.random.random(),
                'z': np.random.random()
            })() for _ in range(21)]
        })()
        
        # 手の位置を追加
        for i in range(35):  # 30フレーム分 + 余分
            collector.add_hand_position(dummy_landmarks, time.time())
        
        print(f"✅ 手の位置データ追加に成功（{len(collector.trajectory_buffer)}フレーム）")
        
        # キーサンプルを追加
        collector.add_key_sample("a", "a", dummy_landmarks, "test text")
        collector.add_key_sample("b", "v", dummy_landmarks, "test text")  # 誤入力
        
        print("✅ キーサンプル追加に成功")
        
        # セッション停止
        collector.stop_collection_session()
        print("✅ データ収集セッション停止に成功")
        
        # データセット情報を取得
        dataset_info = collector.get_training_dataset_info()
        print(f"✅ データセット情報取得に成功: {dataset_info['total_samples']}サンプル")
        
        # 学習データセットをエクスポート
        dataset_file = collector.export_training_dataset()
        if dataset_file:
            print(f"✅ 学習データセットエクスポートに成功: {dataset_file}")
        
        print("✅ 拡張データ収集システムのテスト完了")
        return True
        
    except Exception as e:
        print(f"❌ 拡張データ収集システムのテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lstm_model():
    """LSTMモデルのテスト"""
    print("\n🧪 LSTMモデルのテスト")
    print("=" * 50)
    
    try:
        from processing.models.hand_lstm import BasicHandLSTM, HandLSTMTrainer
        
        # モデルを作成
        model = BasicHandLSTM(
            input_size=63,
            hidden_size=32,  # テスト用に小さく
            num_layers=1,
            num_classes=37,
            dropout=0.1
        )
        
        print("✅ BasicHandLSTMの作成に成功")
        print(f"   パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
        
        # ダミーの入力データを作成
        dummy_input = np.random.random((30, 63))  # 30フレーム × 63次元
        
        # 予測テスト
        predicted_key, confidence = model.predict_key(dummy_input)
        print(f"✅ 予測テスト成功: {predicted_key} (確信度: {confidence:.3f})")
        
        # 学習器を作成
        trainer = HandLSTMTrainer(model, learning_rate=0.001)
        print("✅ HandLSTMTrainerの作成に成功")
        
        # モデルを保存・読み込みテスト
        test_model_file = "test_data/test_model.pth"
        os.makedirs(os.path.dirname(test_model_file), exist_ok=True)
        
        model.save_model(test_model_file)
        print("✅ モデル保存に成功")
        
        loaded_model = BasicHandLSTM.load_model(test_model_file)
        print("✅ モデル読み込みに成功")
        
        # 読み込んだモデルで予測テスト
        loaded_predicted_key, loaded_confidence = loaded_model.predict_key(dummy_input)
        print(f"✅ 読み込みモデルの予測テスト成功: {loaded_predicted_key} (確信度: {loaded_confidence:.3f})")
        
        # テストファイルを削除
        os.remove(test_model_file)
        print("✅ テストファイルを削除")
        
        print("✅ LSTMモデルのテスト完了")
        return True
        
    except Exception as e:
        print(f"❌ LSTMモデルのテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_structures():
    """データ構造のテスト"""
    print("\n🧪 データ構造のテスト")
    print("=" * 50)
    
    try:
        # サンプルデータの構造を確認
        sample_data = {
            'timestamp': datetime.now().isoformat(),
            'user_id': 'test_user',
            'target_text': 'hello world',
            'intended_key': 'h',
            'actual_key': 'h',
            'current_context': 'h',
            'hand_landmarks': [0.5] * 63,  # 21点 × 3座標
            'trajectory_data': [
                {
                    'timestamp': time.time(),
                    'landmarks': [0.5] * 63,
                    'frame_index': i
                }
                for i in range(30)
            ],
            'trajectory_length': 30,
            'session_duration': 5.0
        }
        
        print("✅ サンプルデータ構造の作成に成功")
        
        # 軌跡データの構造を確認
        trajectory_data = [
            {
                'timestamp': time.time() + i * 0.033,  # 30fps
                'landmarks': [0.5 + np.random.normal(0, 0.01) for _ in range(63)],
                'frame_index': i
            }
            for i in range(30)
        ]
        
        print("✅ 軌跡データ構造の作成に成功")
        
        # データの整合性チェック
        assert len(sample_data['hand_landmarks']) == 63, "ランドマーク次元数が不正"
        assert len(sample_data['trajectory_data']) == 30, "軌跡フレーム数が不正"
        assert sample_data['trajectory_length'] == 30, "軌跡長が不正"
        
        print("✅ データ整合性チェックに成功")
        
        # JSONシリアライゼーションテスト
        json_str = json.dumps(sample_data, indent=2)
        loaded_data = json.loads(json_str)
        
        assert loaded_data['intended_key'] == 'h', "JSON読み込み後のデータが不正"
        print("✅ JSONシリアライゼーションテストに成功")
        
        print("✅ データ構造のテスト完了")
        return True
        
    except Exception as e:
        print(f"❌ データ構造のテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("🚀 学習システムの動作確認テスト開始")
    print("=" * 60)
    
    # テスト結果を記録
    test_results = []
    
    # 各テストを実行
    test_results.append(("拡張データ収集システム", test_enhanced_data_collector()))
    test_results.append(("LSTMモデル", test_lstm_model()))
    test_results.append(("データ構造", test_data_structures()))
    
    # 結果を表示
    print("\n📊 テスト結果サマリー")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\n結果: {passed}/{total} テストが成功")
    
    if passed == total:
        print("🎉 全てのテストが成功しました！")
        print("学習システムは正常に動作します。")
    else:
        print("⚠️ 一部のテストが失敗しました。")
        print("エラーログを確認してください。")
    
    # テストデータディレクトリをクリーンアップ
    if os.path.exists("test_data"):
        import shutil
        shutil.rmtree("test_data")
        print("🧹 テストデータディレクトリをクリーンアップしました")

if __name__ == "__main__":
    main()
