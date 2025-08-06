import cv2
import numpy as np

class DisplayManager:
    def __init__(self, frame_width=1280, frame_height=720):
        self.width = frame_width
        self.height = frame_height
        
        # カラーパレット（BGR形式）
        self.colors = {
            'primary': (0, 255, 0),      # 緑 - メイン情報
            'secondary': (255, 255, 0),   # 黄 - 補助情報  
            'success': (0, 255, 0),       # 緑 - 成功
            'error': (0, 0, 255),         # 赤 - エラー
            'warning': (0, 165, 255),     # オレンジ - 警告
            'info': (255, 255, 255),      # 白 - 一般情報
            'debug': (255, 0, 255),       # マゼンタ - デバッグ
            'prediction': (0, 255, 255),  # シアン - 予測情報
        }
        
        # フォント設定
        self.fonts = {
            'small': (cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1),
            'medium': (cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2),
            'large': (cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2),
            'extra_large': (cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3),
        }
        
        # レイアウト設定
        self.layouts = {
            'top_area': {'y_start': 10, 'y_end': 150, 'x_margin': 10},
            'center_area': {'y_start': 150, 'y_end': 450, 'x_margin': 10},
            'bottom_area': {'y_start': 450, 'y_end': None, 'x_margin': 10},
        }
        
        # 行の高さ設定
        self.line_heights = {
            'small': 20,
            'medium': 30,
            'large': 40,
            'extra_large': 50,
        }
    
    def update_frame_size(self, frame):
        """フレームサイズを更新"""
        self.height, self.width = frame.shape[:2]
        # bottom_areaの終点を更新
        self.layouts['bottom_area']['y_end'] = self.height - 10
    
    def draw_text(self, frame, text, position, font_size='medium', color='info', 
                  background=False, background_color=(0, 0, 0), background_alpha=0.7):
        """統一的なテキスト描画"""
        font, scale, thickness = self.fonts[font_size]
        text_color = self.colors[color]
        x, y = position
        
        # 背景を描画する場合
        if background:
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]
            bg_x1 = x - 5
            bg_y1 = y - text_size[1] - 5
            bg_x2 = x + text_size[0] + 5
            bg_y2 = y + 5
            
            # 半透明背景の描画
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1)
            cv2.addWeighted(overlay, background_alpha, frame, 1 - background_alpha, 0, frame)
        
        cv2.putText(frame, text, (x, y), font, scale, text_color, thickness)
        return y + self.line_heights[font_size]
    
    def draw_top_area(self, frame, info_dict):
        """上部エリアの描画（リアルタイム情報）"""
        area = self.layouts['top_area']
        x_start = area['x_margin']
        y_current = area['y_start'] + 25  # 最初の行の位置
        
        # システム状態
        if 'system_status' in info_dict:
            y_current = self.draw_text(frame, f"System: {info_dict['system_status']}", 
                                     (x_start, y_current), 'small', 'info')
        
        # 現在の予測
        if 'current_prediction' in info_dict:
            y_current = self.draw_text(frame, f"Current Prediction: {info_dict['current_prediction']}", 
                                     (x_start, y_current), 'large', 'primary', 
                                     background=True)
        
        # 最後の予測（一定時間表示）
        if 'last_prediction' in info_dict and info_dict['last_prediction']:
            y_current = self.draw_text(frame, f"Last: {info_dict['last_prediction']}", 
                                     (x_start, y_current), 'medium', 'prediction')
        
        return y_current
    
    def draw_center_area(self, frame, info_dict):
        """中央エリアの描画（メイン情報）"""
        area = self.layouts['center_area']
        x_start = area['x_margin']
        y_current = area['y_start'] + 30
        
        # テストモード用の表示
        if 'test_instruction' in info_dict:
            y_current = self.draw_text(frame, f"Type this key: {info_dict['test_instruction']}", 
                                     (x_start, y_current), 'extra_large', 'warning',
                                     background=True, background_color=(0, 0, 100))
        
        if 'test_progress' in info_dict:
            y_current = self.draw_text(frame, f"Progress: {info_dict['test_progress']}", 
                                     (x_start, y_current), 'medium', 'info')
        
        # 結果表示
        if 'test_result' in info_dict:
            result_color = 'success' if info_dict['test_result'] == 'CORRECT' else 'error'
            y_current = self.draw_text(frame, f"Result: {info_dict['test_result']}", 
                                     (x_start, y_current), 'large', result_color,
                                     background=True)
        
        if 'accuracy' in info_dict:
            y_current = self.draw_text(frame, f"Accuracy: {info_dict['accuracy']:.1f}%", 
                                     (x_start, y_current), 'medium', 'secondary')
        
        return y_current
    
    def draw_bottom_area(self, frame, info_dict):
        """下部エリアの描画（履歴・統計）"""
        area = self.layouts['bottom_area']
        x_start = area['x_margin']
        y_start = area['y_end'] - 80  # 下から80px上から開始
        
        # 入力履歴
        if 'actual_history' in info_dict:
            self.draw_text(frame, f"Actual: {info_dict['actual_history']}", 
                          (x_start, y_start), 'medium', 'error')
        
        if 'predicted_history' in info_dict:
            self.draw_text(frame, f"Predicted: {info_dict['predicted_history']}", 
                          (x_start, y_start + self.line_heights['medium']), 'medium', 'success')
    
    def draw_debug_info(self, frame, info_dict):
        """デバッグ情報の描画"""
        if 'fps' in info_dict:
            self.draw_text(frame, f"FPS: {info_dict['fps']:.1f}", 
                          (self.width - 100, 30), 'small', 'debug')
        
        if 'hand_detected' in info_dict:
            status_text = "Hand: OK" if info_dict['hand_detected'] else "Hand: NO"
            color = 'success' if info_dict['hand_detected'] else 'error'
            self.draw_text(frame, status_text, (self.width - 100, 50), 'small', color)
    
    def render_frame(self, frame, info_dict, mode='debug'):
        """フレーム全体のレンダリング"""
        self.update_frame_size(frame)
        
        # 各エリアを描画
        self.draw_top_area(frame, info_dict)
        self.draw_center_area(frame, info_dict)
        self.draw_bottom_area(frame, info_dict)
        
        # デバッグ情報（デバッグモードのみ）
        if mode == 'debug':
            self.draw_debug_info(frame, info_dict)
        
        return frame