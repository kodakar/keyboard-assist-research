# src/filters/moving_average.py
from collections import deque
from .base_filter import BaseFilter

class MovingAverageFilter(BaseFilter):
    """移動平均フィルター"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.position_history = deque(maxlen=window_size)
    
    def filter(self, landmark):
        """
        ランドマークに移動平均フィルターを適用
        
        Parameters:
        landmark: MediaPipeのランドマークオブジェクト
        
        Returns:
        tuple: フィルタリング後の(x, y, z)座標
        """
        # 位置履歴に追加
        self.position_history.append((landmark.x, landmark.y, landmark.z))
        
        # 履歴が十分にたまるまでは元の値を返す
        if len(self.position_history) < self.window_size:
            return landmark.x, landmark.y, landmark.z
        
        # 移動平均を計算
        filtered_x = sum(p[0] for p in self.position_history) / len(self.position_history)
        filtered_y = sum(p[1] for p in self.position_history) / len(self.position_history)
        filtered_z = sum(p[2] for p in self.position_history) / len(self.position_history)
        
        return filtered_x, filtered_y, filtered_z
    
    def reset(self):
        """フィルターの状態をリセット"""
        self.position_history.clear()