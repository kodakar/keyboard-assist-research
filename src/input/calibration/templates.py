# src/input/calibration/templates.py
"""
キーボード配列のテンプレート定義
JIS配列の相対位置情報を保持
"""

class KeyboardTemplate:
    """キーボード配列の基底クラス"""
    
    def __init__(self):
        self.layout = self.get_layout()
        self.rows = self.get_row_structure()
        
    def get_layout(self):
        """配列を返す（サブクラスで実装）"""
        raise NotImplementedError
        
    def get_row_structure(self):
        """行構造を返す（サブクラスで実装）"""
        raise NotImplementedError
    
    def get_key_relative_position(self, key):
        """キーの相対位置を取得"""
        if key in self.layout:
            return self.layout[key]
        return None


class JISLayoutTemplate(KeyboardTemplate):
    """JIS配列のテンプレート"""
    
    def get_layout(self):
        """
        JIS配列の相対位置（単位：キー1個分）
        基準点：'1'キーを(0, 0)とする
        """
        return {
            # 数字行 (y=0)
            '1': (0.0, 0.0),   '2': (1.0, 0.0),   '3': (2.0, 0.0),
            '4': (3.0, 0.0),   '5': (4.0, 0.0),   '6': (5.0, 0.0),
            '7': (6.0, 0.0),   '8': (7.0, 0.0),   '9': (8.0, 0.0),
            '0': (9.0, 0.0),
            
            # QWERTY行 (y=1, 少し右にオフセット)
            'q': (0.5, 1.0),   'w': (1.5, 1.0),   'e': (2.5, 1.0),
            'r': (3.5, 1.0),   't': (4.5, 1.0),   'y': (5.5, 1.0),
            'u': (6.5, 1.0),   'i': (7.5, 1.0),   'o': (8.5, 1.0),
            'p': (9.5, 1.0),
            
            # ASDF行 (y=2, さらに右にオフセット)
            'a': (0.75, 2.0),  's': (1.75, 2.0),  'd': (2.75, 2.0),
            'f': (3.75, 2.0),  'g': (4.75, 2.0),  'h': (5.75, 2.0),
            'j': (6.75, 2.0),  'k': (7.75, 2.0),  'l': (8.75, 2.0),
            
            # ZXCV行 (y=3, さらに右にオフセット)
            'z': (1.25, 3.0),  'x': (2.25, 3.0),  'c': (3.25, 3.0),
            'v': (4.25, 3.0),  'b': (5.25, 3.0),  'n': (6.25, 3.0),
            'm': (7.25, 3.0),
            
            # スペースキー (y=4, 中央, 幅が広い)
            'space': (5.0, 4.0),  # 中心座標
        }
    
    def get_row_structure(self):
        """各行のキー配列"""
        return {
            0: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            1: ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            2: ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            3: ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
            4: ['space']
        }
    
    def get_key_dimensions(self):
        """キーの標準的なサイズ（相対値）"""
        return {
            'default': (1.0, 1.0),      # 通常キー
            'space': (6.0, 1.0),        # スペースキー
        }
    
    def get_row_offsets(self):
        """各行の横方向オフセット（タブ分）"""
        return {
            0: 0.0,    # 数字行
            1: 0.5,    # QWERTY行
            2: 0.75,   # ASDF行
            3: 1.25,   # ZXCV行
            4: 0.0     # スペース行
        }


class LayoutEstimator:
    """検出したグリッドから実際のキー位置を推定"""
    
    def __init__(self, template):
        self.template = template
        
    def estimate_key_positions(self, keyboard_bounds, grid_points=None):
        """
        キーボード領域から各キーの位置を推定
        
        Args:
            keyboard_bounds: (x1, y1, x2, y2) キーボードの矩形領域
            grid_points: 検出されたグリッドポイント（オプション）
        
        Returns:
            dict: 各キーの推定位置 {key: (x, y, width, height)}
        """
        x1, y1, x2, y2 = keyboard_bounds
        keyboard_width = x2 - x1
        keyboard_height = y2 - y1
        
        # 基準単位の計算
        # 横方向: 10個のキー + 余白
        key_width = keyboard_width / 12.0
        # 縦方向: 5行
        key_height = keyboard_height / 5.5
        
        key_positions = {}
        key_dims = self.template.get_key_dimensions()
        
        for key, (rel_x, rel_y) in self.template.layout.items():
            # キーのサイズ
            if key == 'space':
                width = key_width * key_dims['space'][0]
                height = key_height * key_dims['space'][1]
            else:
                width = key_width * key_dims['default'][0]
                height = key_height * key_dims['default'][1]
            
            # 中心座標の計算
            center_x = x1 + key_width * (rel_x + 1.0)
            center_y = y1 + key_height * (rel_y + 0.5)
            
            # 正規化座標として保存
            key_positions[key] = {
                'x': center_x,
                'y': center_y,
                'width': width,
                'height': height
            }
        
        return key_positions
    
    def refine_with_grid(self, initial_positions, grid_points):
        """
        グリッド検出結果を使って位置を精緻化
        
        Args:
            initial_positions: 初期推定位置
            grid_points: 検出されたグリッドポイント
        
        Returns:
            dict: 精緻化された位置
        """
        # TODO: グリッドポイントとテンプレートをマッチング
        # 現在は初期推定をそのまま返す
        return initial_positions