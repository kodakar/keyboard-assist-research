# src/filters/base_filter.py
from abc import ABC, abstractmethod

class BaseFilter(ABC):
    """フィルターの基底クラス"""
    
    @abstractmethod
    def filter(self, value):
        """
        入力値にフィルターを適用する
        
        Parameters:
        value: フィルタリングする値
        
        Returns:
        フィルタリング後の値
        """
        pass
    
    @abstractmethod
    def reset(self):
        """フィルターの状態をリセットする"""
        pass