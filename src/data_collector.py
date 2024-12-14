# src/data_collector.py
import json
from datetime import datetime
import numpy as np

class DataCollector:
    def __init__(self):
        self.data = []
    
    def add_sample(self, key, hand_landmarks):
        """キー入力と手の位置を記録"""
        # ランドマークを配列に変換
        landmarks_array = []
        for landmark in hand_landmarks.landmark:
            landmarks_array.extend([landmark.x, landmark.y, landmark.z])
            
        sample = {
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'hand_position': landmarks_array
        }
        self.data.append(sample)
    
    def save_to_file(self, filename='data/collected_data.json'):
        """収集したデータをJSONファイルとして保存"""
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)