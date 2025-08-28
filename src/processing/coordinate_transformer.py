# src/processing/coordinate_transformer.py
"""
座標変換クラス
カメラのピクセル座標をキーボード基準の相対座標に変換する機能を提供
"""

import json
import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KeyInfo:
    """キー情報を格納するデータクラス"""
    key: str
    center_x: float  # キーボード空間での中心X座標 (0-1)
    center_y: float  # キーボード空間での中心Y座標 (0-1)
    width: float     # キーサイズ (キーボード空間での幅)
    height: float    # キーサイズ (キーボード空間での高さ)
    relative_x: float  # 指定キーからの相対X座標 (キーサイズ単位)
    relative_y: float  # 指定キーからの相対Y座標 (キーサイズ単位)


class CoordinateTransformer:
    """
    座標変換クラス
    カメラのピクセル座標をキーボード基準の相対座標に変換
    """
    
    def __init__(self, keyboard_map_path: str = 'keyboard_map.json'):
        """
        座標変換クラスの初期化
        
        Args:
            keyboard_map_path: キーボードマップファイルのパス
        """
        self.keyboard_map_path = keyboard_map_path
        self.keyboard_map = None
        self.homography_matrix = None
        self.keyboard_corners = None
        
        # キーボードマップを読み込み
        self._load_keyboard_map()
        
        # キーボードの4隅の座標を設定（デフォルト値）
        self._set_default_keyboard_corners()
    
    def _load_keyboard_map(self):
        """キーボードマップを読み込み"""
        try:
            if os.path.exists(self.keyboard_map_path):
                with open(self.keyboard_map_path, 'r', encoding='utf-8') as f:
                    self.keyboard_map = json.load(f)
                print(f"✅ キーボードマップを読み込みました: {self.keyboard_map_path}")
            else:
                print(f"⚠️ キーボードマップファイルが見つかりません: {self.keyboard_map_path}")
                self.keyboard_map = {}
        except Exception as e:
            print(f"⚠️ キーボードマップ読み込みエラー: {e}")
            self.keyboard_map = {}
    
    def _set_default_keyboard_corners(self):
        """キーボードの4隅の座標をデフォルト値で設定"""
        # デフォルトでは画面の中央付近にキーボードがあると仮定
        # 実際の使用時は set_keyboard_corners で更新される
        self.keyboard_corners = np.array([
            [0.2, 0.3],  # 左上
            [0.8, 0.3],  # 右上
            [0.8, 0.7],  # 右下
            [0.2, 0.7]   # 左下
        ], dtype=np.float32)
    
    def set_keyboard_corners(self, corners: np.ndarray):
        """
        キーボードの4隅の座標を設定
        
        Args:
            corners: 4隅の座標 (左上, 右上, 右下, 左下) の配列
                    各座標は (x, y) の形式で、0-1の正規化座標
        """
        if corners.shape != (4, 2):
            raise ValueError("キーボードの4隅の座標は4x2の配列である必要があります")
        
        self.keyboard_corners = corners.astype(np.float32)
        
        # ホモグラフィ行列を再計算
        self._compute_homography()
        
        print(f"✅ キーボードの4隅を設定しました")
        print(f"   左上: ({corners[0][0]:.3f}, {corners[0][1]:.3f})")
        print(f"   右上: ({corners[1][0]:.3f}, {corners[1][1]:.3f})")
        print(f"   右下: ({corners[2][0]:.3f}, {corners[2][1]:.3f})")
        print(f"   左下: ({corners[3][0]:.3f}, {corners[3][1]:.3f})")
    
    def _compute_homography(self):
        """4隅の座標からホモグラフィ行列を計算"""
        try:
            # キーボード空間の4隅（0-1正規化）
            dst_corners = np.array([
                [0.0, 0.0],  # 左上
                [1.0, 0.0],  # 右上
                [1.0, 1.0],  # 右下
                [0.0, 1.0]   # 左下
            ], dtype=np.float32)
            
            # ホモグラフィ行列を計算
            self.homography_matrix = cv2.findHomography(
                self.keyboard_corners, 
                dst_corners, 
                cv2.RANSAC,
                ransacReprojThreshold=0.1
            )[0]
            
            # デバッグ情報
            print(f"🔍 ホモグラフィ計算:")
            print(f"   入力座標: {self.keyboard_corners}")
            print(f"   目標座標: {dst_corners}")
            print(f"   行列: {self.homography_matrix}")
            
            print(f"✅ ホモグラフィ行列を計算しました")
            
        except Exception as e:
            print(f"⚠️ ホモグラフィ行列計算エラー: {e}")
            self.homography_matrix = None
    
    def pixel_to_keyboard_space(self, pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float]]:
        """
        MediaPipeの正規化座標（0-1）をキーボード空間（0,0）〜（1,1）の正規化座標に変換
        
        Args:
            pixel_x: MediaPipeの正規化X座標（0-1）
            pixel_y: MediaPipeの正規化Y座標（0-1）
            
        Returns:
            キーボード空間座標 (x, y) または None（変換失敗時）
        """
        if self.homography_matrix is None:
            print("⚠️ ホモグラフィ行列が計算されていません")
            return None
        
        try:
            # MediaPipeの座標は既に0-1正規化されているので、そのまま使用
            norm_x = pixel_x
            norm_y = pixel_y
            
            # デバッグ情報
            print(f"🔍 座標変換: MediaPipe座標({pixel_x:.3f}, {pixel_y:.3f}) → キーボード空間変換")
            
            # ホモグラフィ変換
            src_point = np.array([[[norm_x, norm_y]]], dtype=np.float32)
            dst_point = cv2.perspectiveTransform(src_point, self.homography_matrix)
            
            kb_x = dst_point[0][0][0]
            kb_y = dst_point[0][0][1]
            
            # デバッグ情報
            print(f"🔍 ホモグラフィ変換: MediaPipe座標({norm_x:.3f}, {norm_y:.3f}) → キーボード空間({kb_x:.3f}, {kb_y:.3f})")
            
            # キーボード空間内かチェック
            if 0.0 <= kb_x <= 1.0 and 0.0 <= kb_y <= 1.0:
                return (kb_x, kb_y)
            else:
                # キーボード空間外の場合は警告とクリッピング
                print(f"⚠️ キーボード空間外: ({kb_x:.3f}, {kb_y:.3f}) → クリッピング")
                kb_x = np.clip(kb_x, 0.0, 1.0)
                kb_y = np.clip(kb_y, 0.0, 1.0)
                return (kb_x, kb_y)
                
        except Exception as e:
            print(f"⚠️ 座標変換エラー: {e}")
            return None
    
    def keyboard_to_key_relative(self, kb_x: float, kb_y: float, target_key: str) -> Optional[Tuple[float, float]]:
        """
        キーボード空間の座標を、特定キーからの相対座標に変換（キーサイズ単位で表現）
        
        Args:
            kb_x: キーボード空間X座標 (0-1)
            kb_y: キーボード空間Y座標 (0-1)
            target_key: 基準となるキー
            
        Returns:
            相対座標 (x, y) または None（変換失敗時）
        """
        if target_key not in self.keyboard_map:
            print(f"⚠️ キー '{target_key}' がキーボードマップに見つかりません")
            return None
        
        try:
            target_info = self.keyboard_map[target_key]
            target_center_x = target_info['x']
            target_center_y = target_info['y']
            target_width = target_info.get('width', 0.05)  # デフォルト値
            target_height = target_info.get('height', 0.05)  # デフォルト値
            
            # 相対座標を計算（キーサイズ単位）
            relative_x = (kb_x - target_center_x) / target_width
            relative_y = (kb_y - target_center_y) / target_height
            
            return (relative_x, relative_y)
            
        except Exception as e:
            print(f"⚠️ 相対座標計算エラー: {e}")
            return None
    
    def get_nearest_keys_with_relative_coords(self, kb_x: float, kb_y: float, top_k: int = 3) -> List[KeyInfo]:
        """
        最も近いk個のキーとその相対座標を返す
        
        Args:
            kb_x: キーボード空間X座標 (0-1)
            kb_y: キーボード空間Y座標 (0-1)
            top_k: 取得するキーの数
            
        Returns:
            キー情報のリスト（距離順）
        """
        if not self.keyboard_map:
            return []
        
        try:
            key_distances = []
            
            for key, key_info in self.keyboard_map.items():
                key_center_x = key_info['x']
                key_center_y = key_info['y']
                key_width = key_info.get('width', 0.05)
                key_height = key_info.get('height', 0.05)
                
                # ユークリッド距離を計算
                distance = np.sqrt((kb_x - key_center_x)**2 + (kb_y - key_center_y)**2)
                
                # 相対座標を計算
                relative_x = (kb_x - key_center_x) / key_width
                relative_y = (kb_y - key_center_y) / key_height
                
                key_info_obj = KeyInfo(
                    key=key,
                    center_x=key_center_x,
                    center_y=key_center_y,
                    width=key_width,
                    height=key_height,
                    relative_x=relative_x,
                    relative_y=relative_y
                )
                
                key_distances.append((distance, key_info_obj))
            
            # 距離順にソート
            key_distances.sort(key=lambda x: x[0])
            
            # top_k個のキーを返す
            return [key_info for _, key_info in key_distances[:top_k]]
            
        except Exception as e:
            print(f"⚠️ 最近傍キー取得エラー: {e}")
            return []
    
    def set_screen_size(self, width: int, height: int):
        """
        画面サイズを設定（ピクセル座標の正規化に使用）
        
        Args:
            width: 画面幅（ピクセル）
            height: 画面高さ（ピクセル）
        """
        self.screen_width = width
        self.screen_height = height
        print(f"✅ 画面サイズを設定しました: {width}x{height}")
    
    def get_keyboard_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        キーボードの境界を取得
        
        Returns:
            (min_x, min_y, max_x, max_y) または None
        """
        if self.keyboard_corners is None:
            return None
        
        min_x = np.min(self.keyboard_corners[:, 0])
        min_y = np.min(self.keyboard_corners[:, 1])
        max_x = np.max(self.keyboard_corners[:, 0])
        max_y = np.max(self.keyboard_corners[:, 1])
        
        return (min_x, min_y, max_x, max_y)
    
    def is_point_in_keyboard_space(self, kb_x: float, kb_y: float) -> bool:
        """
        点がキーボード空間内にあるかチェック
        
        Args:
            kb_x: キーボード空間X座標
            kb_y: キーボード空間Y座標
            
        Returns:
            キーボード空間内の場合True
        """
        return 0.0 <= kb_x <= 1.0 and 0.0 <= kb_y <= 1.0
