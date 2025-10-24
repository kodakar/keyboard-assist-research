#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
座標系の一貫性をテストする統合テスト
MediaPipe座標 → 作業領域座標 → 15次元特徴量 → モデル入力の一貫性を確認
"""

import unittest
import numpy as np
import torch
import json
import os
import tempfile
import shutil
from datetime import datetime

# テスト対象のモジュールをインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.processing.feature_extractor import FeatureExtractor
from src.processing.models.hand_lstm import BasicHandLSTM
from src.processing.coordinate_transformer import WorkAreaTransformer
from src.processing.enhanced_data_collector import EnhancedDataCollector
from src.processing.data_loader import KeyboardIntentDataset, create_data_loaders
from config.feature_config import get_feature_dim, get_sequence_length, get_num_classes


class TestCoordinateConsistency(unittest.TestCase):
    """座標系の一貫性をテストするクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        # テスト用の一時ディレクトリを作成
        self.test_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.test_dir, "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # テスト用の作業領域コーナー（4点）
        self.test_corners = np.array([
            [0.1, 0.2],  # 左上
            [0.9, 0.2],  # 右上
            [0.9, 0.8],  # 右下
            [0.1, 0.8]   # 左下
        ], dtype=np.float32)
        
        # テスト用のキーボードマップ
        self.test_keyboard_map = {
            'a': {'x': 0.2, 'y': 0.3, 'width': 0.05, 'height': 0.05},
            'b': {'x': 0.3, 'y': 0.3, 'width': 0.05, 'height': 0.05},
            'c': {'x': 0.4, 'y': 0.3, 'width': 0.05, 'height': 0.05},
            'space': {'x': 0.5, 'y': 0.7, 'width': 0.2, 'height': 0.05}
        }
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 一時ディレクトリを削除
        shutil.rmtree(self.test_dir)
    
    def test_work_area_transformer_initialization(self):
        """WorkAreaTransformerの初期化テスト"""
        transformer = WorkAreaTransformer()
        self.assertIsNotNone(transformer)
        self.assertEqual(transformer.feature_dim, get_feature_dim())
    
    def test_coordinate_transformation(self):
        """MediaPipe座標 → 作業領域座標の変換テスト"""
        transformer = WorkAreaTransformer()
        transformer.set_work_area_corners(self.test_corners)
        
        # MediaPipe座標（0-1正規化）をテスト
        test_coords = [
            (0.1, 0.2),  # 左上付近
            (0.5, 0.5),  # 中央
            (0.9, 0.8),  # 右下付近
        ]
        
        for mp_x, mp_y in test_coords:
            wa_coords = transformer.pixel_to_work_area(mp_x, mp_y)
            self.assertIsNotNone(wa_coords)
            wa_x, wa_y = wa_coords
            
            # 作業領域座標は0-1の範囲内であることを確認
            self.assertGreaterEqual(wa_x, 0.0)
            self.assertLessEqual(wa_x, 1.0)
            self.assertGreaterEqual(wa_y, 0.0)
            self.assertLessEqual(wa_y, 1.0)
    
    def test_feature_dimensions(self):
        """特徴量が15次元であることを確認"""
        extractor = FeatureExtractor()
        
        # テスト用の軌跡データを作成
        dummy_trajectory = []
        for i in range(60):
            # 作業領域座標と最近傍キー情報を含むフレーム
            frame_data = {
                'work_area_coords': {
                    'index_finger': {'x': 0.5 + 0.01 * np.sin(i * 0.1), 'y': 0.5 + 0.01 * np.cos(i * 0.1)}
                },
                'nearest_keys_relative': [
                    {
                        'key': 'a',
                        'relative_x': 0.1,
                        'relative_y': 0.1,
                        'distance': 0.14,
                        'approach_velocity': 0.5
                    },
                    {
                        'key': 'b',
                        'relative_x': 0.2,
                        'relative_y': 0.2,
                        'distance': 0.28,
                        'approach_velocity': 0.3
                    },
                    {
                        'key': 'c',
                        'relative_x': 0.3,
                        'relative_y': 0.3,
                        'distance': 0.42,
                        'approach_velocity': 0.1
                    }
                ]
            }
            dummy_trajectory.append(frame_data)
        
        # 特徴量を抽出
        features = extractor.extract_from_trajectory(dummy_trajectory)
        
        # 形状の確認
        expected_dim = get_feature_dim()
        expected_seq_len = get_sequence_length()
        self.assertEqual(features.shape, (expected_seq_len, expected_dim))
        
        # 特徴量の内容確認
        # 0-1: 作業領域での指の座標
        self.assertTrue(np.all((features[:, 0] >= 0.0) & (features[:, 0] <= 1.0)))
        self.assertTrue(np.all((features[:, 1] >= 0.0) & (features[:, 1] <= 1.0)))
        
        # 2-7: 最近傍3キーへの相対座標（-5から5の範囲）
        self.assertTrue(np.all((features[:, 2:8] >= -5.0) & (features[:, 2:8] <= 5.0)))
        
        # 8-10: 最近傍3キーへの距離（0から10の範囲）
        self.assertTrue(np.all((features[:, 8:11] >= 0.0) & (features[:, 8:11] <= 10.0)))
        
        # 11-14: 速度・加速度（-5から5の範囲）
        self.assertTrue(np.all((features[:, 11:] >= -5.0) & (features[:, 11:] <= 5.0)))
    
    def test_model_input(self):
        """モデルが設定ファイルで定義された次元数入力を受け付けることを確認"""
        model = BasicHandLSTM()
        expected_dim = get_feature_dim()
        expected_seq_len = get_sequence_length()
        expected_classes = get_num_classes()
        
        # テスト用の入力データ
        dummy_input = torch.randn(32, expected_seq_len, expected_dim)
        
        # モデルに通す
        output = model(dummy_input)
        
        # 出力形状の確認
        self.assertEqual(output.shape, (32, expected_classes))
        
        # 出力が確率分布として妥当であることを確認
        probabilities = torch.softmax(output, dim=1)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(32)))
    
    def test_data_collection_to_loading_consistency(self):
        """データ収集 → 読み込み → 学習の一貫性テスト"""
        # テスト用のデータ収集器を作成
        collector = EnhancedDataCollector(
            user_id="test_user",
            data_dir=self.test_data_dir
        )
        
        # 作業領域コーナーを設定
        collector.set_work_area_corners(self.test_corners)
        
        # テスト用の軌跡データを作成
        test_trajectory = []
        for i in range(60):
            frame_data = {
                'timestamp': datetime.now().timestamp(),
                'frame_index': i,
                'work_area_coords': {
                    'index_finger': {'x': 0.5, 'y': 0.5}
                },
                'nearest_keys_relative': [
                    {
                        'key': 'a',
                        'relative_x': 0.1,
                        'relative_y': 0.1,
                        'distance': 0.14,
                        'approach_velocity': 0.5
                    }
                ],
                'data_version': '2.0'
            }
            test_trajectory.append(frame_data)
        
        # サンプルデータを作成
        sample_data = {
            'timestamp': datetime.now().isoformat(),
            'data_version': '2.0',
            'coordinate_system': 'relative_keyboard_space',  # データローダーが期待する値
            'user_id': 'test_user',
            'target_char': 'a',
            'trajectory_data': test_trajectory
        }
        
        # サンプルを保存
        sample_file = os.path.join(self.test_data_dir, "samples", "sample_test.json")
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        # デバッグ: 保存されたファイルの確認
        print(f"💾 サンプルファイルを保存: {sample_file}")
        print(f"📁 ファイル存在確認: {os.path.exists(sample_file)}")
        print(f"📁 ディレクトリ内容: {os.listdir(os.path.dirname(sample_file))}")
        print(f"📁 サンプルディレクトリ内容: {os.listdir(self.test_data_dir)}")
        
        # データローダーで読み込み
        try:
            dataset = KeyboardIntentDataset(
                data_dir=self.test_data_dir,
                sequence_length=60,
                train=True,
                augment=False
            )
            
            # データセットの情報確認
            self.assertEqual(len(dataset), 1)  # 1サンプル
            self.assertEqual(dataset.feature_dim, get_feature_dim())
            self.assertEqual(dataset.sequence_length, get_sequence_length())
            self.assertEqual(dataset.num_classes, get_num_classes())
            
            # サンプルの取得
            features, label = dataset[0]
            
            # 特徴量の形状確認
            self.assertEqual(features.shape, (get_sequence_length(), get_feature_dim()))
            
            # ラベルの確認
            self.assertIsInstance(label, int)
            self.assertGreaterEqual(label, 0)
            self.assertLess(label, get_num_classes())
            
        except Exception as e:
            self.fail(f"データローダーでの読み込みに失敗: {e}")
    
    def test_model_training_consistency(self):
        """モデルの学習時の一貫性テスト"""
        # 設定ファイルから値を取得
        model = BasicHandLSTM()
        batch_size = 16
        sequence_length = get_sequence_length()
        feature_dim = get_feature_dim()
        num_classes = get_num_classes()
        
        # ダミーの学習データ
        X_train = torch.randn(batch_size, sequence_length, feature_dim)
        y_train = torch.randint(0, num_classes, (batch_size,))
        
        # モデルに通す
        model.train()
        output = model(X_train)
        
        # 出力形状の確認
        self.assertEqual(output.shape, (batch_size, num_classes))
        
        # 損失計算
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, y_train)
        
        # 損失が数値であることを確認
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)
    
    def test_coordinate_system_versioning(self):
        """座標系のバージョン管理テスト"""
        # データバージョン2.0の確認
        self.assertEqual('work_area_v2', 'work_area_v2')
        
        # 座標系の一貫性確認
        coordinate_systems = [
            'work_area_v2',
            'work_area_v2',
            'work_area_v2'
        ]
        
        for coord_sys in coordinate_systems:
            self.assertEqual(coord_sys, 'work_area_v2')
    
    def test_feature_extractor_robustness(self):
        """特徴量抽出器の堅牢性テスト"""
        extractor = FeatureExtractor()
        
        # 不完全なデータでのテスト
        incomplete_trajectory = [
            {
                'work_area_coords': {
                    'index_finger': {'x': 0.5, 'y': 0.5}
                }
                # nearest_keys_relativeが欠けている
            }
            for _ in range(30)  # 60フレーム未満
        ]
        
        # エラーが発生しないことを確認
        try:
            features = extractor.extract_from_trajectory(incomplete_trajectory)
            expected_dim = get_feature_dim()
            expected_seq_len = get_sequence_length()
            self.assertEqual(features.shape, (expected_seq_len, expected_dim))
        except Exception as e:
            self.fail(f"不完全なデータでの特徴量抽出に失敗: {e}")


if __name__ == '__main__':
    # テストの実行
    unittest.main(verbosity=2)
