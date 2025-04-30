# src/keyboard_detector.py
import cv2
import numpy as np

class KeyboardDetector:
    def __init__(self):
        self.key_layout = [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']
        ]
        
    def detect_keyboard(self, frame):
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 表示用にコピー
        debug_frame = frame.copy()
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 大きい輪郭のみをフィルタリング
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        
        # 検出された輪郭を描画
        cv2.drawContours(debug_frame, large_contours, -1, (0, 255, 0), 2)
        
        # エッジ検出結果も表示
        cv2.imshow('Edges', edges)
        cv2.imshow('Keyboard Detection', debug_frame)
        
        return debug_frame
        
    def get_key_positions(self):
        # 検出したキーボード領域から
        # 各キーの位置を推定
        pass