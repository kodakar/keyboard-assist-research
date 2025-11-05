# src/input/calibration/simple_mapper.py
"""
4点クリックによる簡易キーボードマッピング
グリッド検出やパースペクティブ補正なしで、直接キー配置を計算
"""

import numpy as np
from typing import Dict, Tuple, List


class SimpleKeyboardMapper:
    """4点からキー配置を計算する簡易マッパー"""
    
    def __init__(self):
        # 各行のキー配列（研究対象の37キーのみ）
        self.key_layout = {
            0: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],           # 数字行
            1: ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],           # QWERTY行
            2: ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],                # ASDF行（9キー）
            3: ['z', 'x', 'c', 'v', 'b', 'n', 'm'],                          # ZXCV行（7キー）
            4: ['space']                                                       # スペース行
        }
        
        # 各行の横方向オフセット（タブキー分のずれ）
        self.row_offsets = {
            0: 0.0,    # 数字行：オフセットなし
            1: 0.5,    # QWERTY行：0.5キー分右にずれ
            2: 0.7,   # ASDF行：0.75キー分右にずれ  
            3: 1.25,   # ZXCV行：1.25キー分右にずれ
            4: 0.0     # スペース行：オフセットなし
        }
        
        # 各行のキー幅の調整（キー数が少ない行は幅を広く）
        self.row_key_widths = {
            0: 1.0,    # 10キー
            1: 1.0,    # 10キー
            2: 1.0,    # 9キー（少し広く）
            3: 1.0,    # 7キー（もっと広く）
            4: 6.0     # スペース（特別な幅）
        }
    
    def map_from_corners(self, corners: List[Tuple[float, float]]) -> Dict:
        """
        4隅の座標からキー配置を計算
        
        Args:
            corners: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 
                    1: 左上（1キーの左上）
                    2: 右上（-キーの右上）
                    3: 右下（スペースキーの右上）
                    4: 左下（スペースキーの左上）
                    全て正規化座標（0.0-1.0）
        
        Returns:
            dict: {key: {'x': x, 'y': y, 'width': w, 'height': h}}
                  全て正規化座標（0.0-1.0）
        """
        if len(corners) != 4:
            raise ValueError("4つの角が必要です")
        
        corners = np.array(corners)
        
        # 文字キー領域の定義
        top_left_x = corners[0][0]   # 1点目のX（1キーの左）
        top_right_x = corners[1][0]  # 2点目のX（-キーの右）
        top_y = (corners[0][1] + corners[1][1]) / 2  # 上辺のY
        
        # スペースキーの上端（点3と点4のY座標）
        space_right_x = corners[2][0]  # 3点目のX（スペースの右上）
        space_left_x = corners[3][0]   # 4点目のX（スペースの左上）
        bottom_y = (corners[2][1] + corners[3][1]) / 2  # 下辺のY（スペースの上端）
        
        # キーボード領域
        keyboard_width = top_right_x - top_left_x
        keyboard_height = bottom_y - top_y  # 文字キー4行の高さ
        
        # 4行分の高さ
        row_height = keyboard_height / 4.0
        
        key_positions = {}
        
        # 文字キーの配置（4行）
        for row_idx, keys in self.key_layout.items():
            # 行のY座標（中心）
            row_y = top_y + (row_idx + 0.5) * row_height
            
            num_keys = len(keys)
            offset = self.row_offsets[row_idx]
            
            # 基準幅の計算（数字行の11キーを基準）
            key_width_base = keyboard_width / 11.0
            key_width = key_width_base * self.row_key_widths[row_idx]
            
            for key_idx, key in enumerate(keys):
                # キーのX座標（オフセット考慮）
                key_x = top_left_x + (offset + key_idx) * key_width_base + key_width / 2
                
                # -キーは研究対象外なのでスキップ
                if key == '-':
                    continue
                    
                key_positions[key] = {
                    'x': key_x,
                    'y': row_y,
                    'width': key_width * 0.9,    # キー間に隙間を作る
                    'height': row_height * 0.8    # 高さの80%
                }
        
        # スペースキーの配置（矩形の外、下に配置）
        space_width = space_right_x - space_left_x
        space_center_x = (space_left_x + space_right_x) / 2
        
        # スペースのY座標は点3,4（スペースの上端）より下
        space_y = bottom_y + row_height * 0.4  # スペースの上端から下に配置
        
        key_positions['space'] = {
            'x': space_center_x,
            'y': space_y,
            'width': space_width * 0.95,       # 実際の幅の95%
            'height': row_height * 0.7         # 通常キーより低め
        }
        
        return key_positions
    
    def map_from_rectangle(self, x1: float, y1: float, x2: float, y2: float) -> Dict:
        """
        矩形領域からキー配置を計算（簡易版）
        
        Args:
            x1, y1: 左上座標（正規化）
            x2, y2: 右下座標（正規化）
        
        Returns:
            dict: キー配置情報
        """
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return self.map_from_corners(corners)
    
    def find_key_at_position(self, x: float, y: float, key_positions: Dict) -> Tuple[str, float]:
        """
        指定座標に最も近いキーを検索
        
        Args:
            x, y: 検索座標（正規化）
            key_positions: キー配置辞書
        
        Returns:
            tuple: (キー名, 距離) or (None, inf)
        """
        # まずスペースキーをチェック（Y座標のみで判定）
        if 'space' in key_positions:
            space_pos = key_positions['space']
            space_top = space_pos['y'] - space_pos['height'] / 2
            space_bottom = space_pos['y'] + space_pos['height'] / 2
            
            if space_top <= y <= space_bottom:
                # スペース領域内
                return 'space', 0.0
        
        # 通常のキーをチェック
        min_distance = float('inf')
        nearest_key = None
        
        for key, pos in key_positions.items():
            if key == 'space':
                continue
                
            # キーの矩形内かチェック
            left = pos['x'] - pos['width'] / 2
            right = pos['x'] + pos['width'] / 2
            top = pos['y'] - pos['height'] / 2
            bottom = pos['y'] + pos['height'] / 2
            
            if left <= x <= right and top <= y <= bottom:
                # 矩形内なら距離0
                return key, 0.0
            
            # 矩形外なら中心からの距離を計算
            distance = ((pos['x'] - x) ** 2 + (pos['y'] - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_key = key
        
        return nearest_key, min_distance
    
    def visualize_layout(self, key_positions: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        キー配置を可視化（デバッグ用）
        
        Args:
            key_positions: キー配置辞書
            frame_shape: (height, width) 画像サイズ
        
        Returns:
            numpy.array: 可視化画像
        """
        import cv2
        
        h, w = frame_shape
        img = np.ones((h, w, 3), dtype=np.uint8) * 255  # 白背景
        
        for key, pos in key_positions.items():
            # 正規化座標から画像座標に変換
            cx = int(pos['x'] * w)
            cy = int(pos['y'] * h)
            kw = int(pos['width'] * w)
            kh = int(pos['height'] * h)
            
            # 矩形を描画
            x1 = cx - kw // 2
            y1 = cy - kh // 2
            x2 = cx + kw // 2
            y2 = cy + kh // 2
            
            # スペースキーは別色
            if key == 'space':
                cv2.rectangle(img, (x1, y1), (x2, y2), (100, 200, 100), 2)
                cv2.putText(img, 'SPACE', (cx - 30, cy + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                cv2.putText(img, key.upper(), (cx - 8, cy + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img


class CalibrationHelper:
    """キャリブレーション作業を支援するヘルパークラス"""
    
    def __init__(self):
        self.clicked_points = []
        self.mapper = SimpleKeyboardMapper()
    
    def add_click(self, x: int, y: int, frame_shape: Tuple[int, int]) -> bool:
        """
        クリック座標を追加
        
        Args:
            x, y: クリック座標（ピクセル）
            frame_shape: (height, width) 画像サイズ
        
        Returns:
            bool: 4点揃ったらTrue
        """
        h, w = frame_shape
        # 正規化座標に変換
        norm_x = x / w
        norm_y = y / h
        
        self.clicked_points.append((norm_x, norm_y))
        
        if len(self.clicked_points) > 4:
            self.clicked_points = self.clicked_points[-4:]  # 最新4点のみ保持
        
        return len(self.clicked_points) == 4
    
    def get_key_positions(self) -> Dict:
        """
        クリックした4点からキー配置を計算
        
        Returns:
            dict: キー配置情報
        """
        if len(self.clicked_points) != 4:
            return None
        
        return self.mapper.map_from_corners(self.clicked_points)
    
    def reset(self):
        """クリック情報をリセット"""
        self.clicked_points = []
    
    def draw_clicks(self, frame: np.ndarray) -> np.ndarray:
        """
        クリック点を画像に描画
        
        Args:
            frame: 入力画像
        
        Returns:
            numpy.array: 描画後の画像
        """
        import cv2
        
        result = frame.copy()
        h, w = frame.shape[:2]

        # マッピング中は常時表示する水平ガイド線（1/3 高さ、半透明、細線）
        overlay = result.copy()
        guide_y = int(h / 6)
        cv2.line(overlay, (0, guide_y), (w - 1, guide_y), (0, 180, 0), 2)
        cv2.addWeighted(overlay, 0.4, result, 0.6, 0, result)
        
        labels = ["1:Top-Left", "2:Top-Right", "3:Bottom-Right", "4:Bottom-Left"]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, (norm_x, norm_y) in enumerate(self.clicked_points):
            x = int(norm_x * w)
            y = int(norm_y * h)
            
            # 点を描画
            cv2.circle(result, (x, y), 8, colors[i], -1)
            cv2.circle(result, (x, y), 10, (255, 255, 255), 2)
            
            # ラベルを描画
            if i < len(labels):
                cv2.putText(result, labels[i], (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        
        # 4点揃ったら矩形とキーボードレイアウトを描画
        if len(self.clicked_points) == 4:
            pts = []
            for norm_x, norm_y in self.clicked_points:
                pts.append([int(norm_x * w), int(norm_y * h)])
            pts = np.array(pts, np.int32)
            
            # 矩形を描画
            cv2.polylines(result, [pts], True, (0, 255, 255), 2)
            
            # キーボードレイアウトを描画
            result = self._draw_keyboard_layout(result, pts)
        
        return result
    
    def _draw_keyboard_layout(self, frame, corners):
        """
        キーボードレイアウトを描画
        
        Args:
            frame: 描画対象の画像
            corners: 4隅の座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        Returns:
            描画後の画像
        """
        import cv2
        import numpy as np
        
        # 4隅からキー配置を計算
        try:
            # frameのサイズ情報を渡してピクセル座標を計算
            h, w = frame.shape[:2]
            key_positions = self._calculate_key_positions_from_corners(corners, (h, w))
            
            # 半透明描画用オーバーレイ
            overlay = frame.copy()

            # 各キーを描画
            for key, (center_x, center_y, width, height) in key_positions.items():
                # キーの四角形を描画
                x1 = int(center_x - width/2)
                y1 = int(center_y - height/2)
                x2 = int(center_x + width/2)
                y2 = int(center_y + height/2)
                
                # 細め・半透明の枠（オーバーレイ側に1pxで描画）
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), 2)
                
                # キーの文字を描画
                if key == ' ':
                    key_text = 'SPACE'
                else:
                    key_text = key.upper()
                
                # 文字サイズを調整
                font_scale = min(width/30, height/30, 0.6)
                font_scale = max(font_scale, 0.3)  # 最小サイズ
                
                # 文字の位置を中央に調整
                text_size = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                text_x = int(center_x - text_size[0]/2)
                text_y = int(center_y + text_size[1]/2)
                
                # 文字もオーバーレイ側に描画（やや淡い緑白）
                cv2.putText(overlay, key_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 255, 200), 1)
            
            # 半透明で合成（矩形と文字を薄く表示）
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # 重要キーのハイライトは不要のため非表示
        
        except Exception as e:
            # エラー時は何もしない
            pass
        
        return frame
    
    def _calculate_key_positions_from_corners(self, corners, frame_shape):
        """
        4隅の座標からキー位置を計算（描画用）
        
        Args:
            corners: 4隅の座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        Returns:
            dict: {key: (center_x, center_y, width, height)}
        """
        import numpy as np
        
        # 4隅を正規化座標に変換（frame_shapeを使用）
        h, w = frame_shape
        normalized_corners = []
        for corner in corners:
            norm_x = corner[0] / w
            norm_y = corner[1] / h
            normalized_corners.append((norm_x, norm_y))
        
        # 既存のマッパーで正規化キー配置を取得
        key_positions_norm = self.mapper.map_from_corners(normalized_corners)
        
        # ピクセル座標に変換
        pixel_positions = {}
        for key, pos in key_positions_norm.items():
            norm_x = pos['x']
            norm_y = pos['y']
            norm_width = pos['width']
            norm_height = pos['height']
            center_x = int(norm_x * w)
            center_y = int(norm_y * h)
            width = int(norm_width * w)
            height = int(norm_height * h)
            pixel_positions[key] = (center_x, center_y, width, height)
        
        return pixel_positions