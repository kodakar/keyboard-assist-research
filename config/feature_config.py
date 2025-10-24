"""
特徴量設定ファイル
キーボード入力意図推定用の特徴量に関する設定を一元管理
"""

import os
from typing import Dict, Any

# 環境変数で上書き可能な設定
FEATURE_CONFIG: Dict[str, Any] = {
    # 特徴量の次元数（30次元に拡張）
    'feature_dim': int(os.getenv('FEATURE_DIM', 30)),
    
    # 時系列データの長さ（フレーム数）
    'sequence_length': int(os.getenv('SEQUENCE_LENGTH', 60)),
    
    # フレームレート
    'fps': float(os.getenv('FPS', 30.0)),
    
    # 特徴量計算用の窓サイズ（過去フレーム数）
    'window_size': int(os.getenv('WINDOW_SIZE', 10)),
    
    # クラス数（37キー: a-z, 0-9, space）
    'num_classes': int(os.getenv('NUM_CLASSES', 37)),
    
    # 特徴量の正規化範囲
    'normalization_ranges': {
        'finger_coords': (0.0, 1.0),           # 指の座標
        'relative_coords': (-5.0, 5.0),        # 相対座標
        'distances': (0.0, 10.0),              # 距離
        'velocity_acceleration': (-5.0, 5.0),  # 速度・加速度
        'amplitude_direction': (0.0, 1.0),     # 振幅・方向転換頻度
        'new_features': (0.0, 10.0),           # 新規特徴量（全体用）
    },
    
    # 新規特徴量の個別正規化範囲（30次元の18-29番目）
    'new_feature_ranges': [
        (0.0, 1.0),      # 18: elapsed_time
        (-3.14159, 3.14159),  # 19: target_angle
        (-3.14159, 3.14159),  # 20: velocity_angle
        (0.0, 3.14159),  # 21: angle_to_target
        (0.0, 10.0),     # 22: speed
        (0.0, 10.0),     # 23: acceleration_magnitude
        (0.0, 20.0),     # 24: jerk
        (0.0, 2.0),      # 25: trajectory_length
        (-5.0, 5.0),     # 26: approach_velocity
        (0.0, 1.0),      # 27: trajectory_curvature
        (0.0, 5.0),      # 28: speed_std
        (0.0, 2.0),      # 29: velocity_consistency
    ],
    
    # 特徴量の構成（30次元の内訳）
    'feature_breakdown': {
        'spatial_info': 11,      # 空間情報: 指の座標(2) + 相対座標(6) + 距離(3)
        'motion_info': 9,        # 動き情報: 速度(2) + 加速度(2) + 大きさ(2) + ジャーク(1) + 軌跡長(1) + 接近速度(1)
        'direction_info': 4,     # 方向情報: 目標角度(1) + 速度角度(1) + 角度ズレ(1) + 曲率(1)
        'stability_info': 5,     # 震え・安定性: 振幅(2) + 方向転換(1) + 速度標準偏差(1) + 変動係数(1)
        'time_info': 1,          # 時間情報: 経過時間(1)
    },
    
    # データ拡張の設定
    'augmentation': {
        'enabled': True,
        'noise_std': 0.01,
        'tremor_probability': 0.5,
        'shift_probability': 0.2,
    },
    
    # 学習設定
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'early_stopping_patience': 10,
    }
}

# 設定値の取得用ヘルパー関数
def get_feature_dim() -> int:
    """特徴量の次元数を取得"""
    return FEATURE_CONFIG['feature_dim']

def get_sequence_length() -> int:
    """時系列データの長さを取得"""
    return FEATURE_CONFIG['sequence_length']

def get_fps() -> float:
    """フレームレートを取得"""
    return FEATURE_CONFIG['fps']

def get_window_size() -> int:
    """特徴量計算用の窓サイズを取得"""
    return FEATURE_CONFIG['window_size']

def get_num_classes() -> int:
    """クラス数を取得"""
    return FEATURE_CONFIG['num_classes']

def get_normalization_range(feature_type: str) -> tuple:
    """特徴量タイプの正規化範囲を取得"""
    return FEATURE_CONFIG['normalization_ranges'].get(feature_type, (0.0, 1.0))

def get_feature_breakdown() -> Dict[str, int]:
    """特徴量の構成を取得"""
    return FEATURE_CONFIG['feature_breakdown']

def get_new_feature_ranges() -> list:
    """新規特徴量の個別正規化範囲を取得"""
    return FEATURE_CONFIG['new_feature_ranges']

def get_new_feature_range(index: int) -> tuple:
    """新規特徴量の指定インデックスの正規化範囲を取得"""
    ranges = FEATURE_CONFIG['new_feature_ranges']
    if 0 <= index < len(ranges):
        return ranges[index]
    else:
        return (0.0, 1.0)  # デフォルト範囲

# 設定の検証
def validate_config() -> bool:
    """設定値の妥当性を検証"""
    try:
        # 特徴量次元数の検証
        total_dim = sum(FEATURE_CONFIG['feature_breakdown'].values())
        if total_dim != FEATURE_CONFIG['feature_dim']:
            print(f"⚠️ 特徴量次元数の不整合: 設定値={FEATURE_CONFIG['feature_dim']}, 計算値={total_dim}")
            return False
        
        # 正の値の検証
        if FEATURE_CONFIG['feature_dim'] <= 0:
            print(f"⚠️ 特徴量次元数が無効: {FEATURE_CONFIG['feature_dim']}")
            return False
            
        if FEATURE_CONFIG['sequence_length'] <= 0:
            print(f"⚠️ 時系列長が無効: {FEATURE_CONFIG['sequence_length']}")
            return False
            
        if FEATURE_CONFIG['fps'] <= 0:
            print(f"⚠️ フレームレートが無効: {FEATURE_CONFIG['fps']}")
            return False
            
        if FEATURE_CONFIG['num_classes'] <= 0:
            print(f"⚠️ クラス数が無効: {FEATURE_CONFIG['num_classes']}")
            return False
        
        print(f"✅ 設定検証完了: {FEATURE_CONFIG['feature_dim']}次元特徴量")
        return True
        
    except Exception as e:
        print(f"❌ 設定検証エラー: {e}")
        return False

# 設定情報の表示
def print_config_info():
    """設定情報を表示"""
    print("📊 特徴量設定情報:")
    print(f"   特徴量次元数: {FEATURE_CONFIG['feature_dim']}")
    print(f"   時系列長: {FEATURE_CONFIG['sequence_length']}")
    print(f"   フレームレート: {FEATURE_CONFIG['fps']}")
    print(f"   クラス数: {FEATURE_CONFIG['num_classes']}")
    print("   特徴量構成:")
    for category, dim in FEATURE_CONFIG['feature_breakdown'].items():
        print(f"     {category}: {dim}次元")

# モジュール読み込み時の検証
if __name__ == "__main__":
    print_config_info()
    validate_config()
else:
    # インポート時の自動検証
    validate_config()
