# src/input/keyboard_map.py - 4点クリック方式を統合した改修版
"""
キーボードマッピングの管理
4点クリック方式を追加
"""

import cv2
import numpy as np
import json
import os
from dotenv import load_dotenv

# 新しいモジュールをインポート
try:
    from .calibration.simple_mapper import SimpleKeyboardMapper, CalibrationHelper
except ImportError:
    from calibration.simple_mapper import SimpleKeyboardMapper, CalibrationHelper


class KeyboardMap:
    def __init__(self, config_file='keyboard_map.json'):
        self.config_file = config_file
        self.key_positions = {}
        self.frame_shape = (720, 1280)
        self.calibration_mode = False
        
        # 4点クリック用
        self.calibration_helper = CalibrationHelper()
        self.simple_mapper = SimpleKeyboardMapper()
        
        load_dotenv()
        self.gemini_available = self._setup_gemini()
        self.target_keys = self._get_target_keys()
        
        if os.path.exists(config_file):
            self.load_map()
        else:
            print("⚠️ キーボードマップが見つかりません。キャリブレーションが必要です。")
            self.key_positions = {}
    
    def _setup_gemini(self):
        """Gemini API の初期化（互換性のため残す）"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return False
            
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            return True
            
        except:
            return False
    
    def _get_target_keys(self):
        """現在の研究対象キーを取得"""
        target_keys = set()
        target_keys.update('abcdefghijklmnopqrstuvwxyz')
        target_keys.update('0123456789')
        target_keys.add('space')
        return target_keys
    
    def save_map(self):
        """キーマップをJSONファイルに保存"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.key_positions, f, indent=2, ensure_ascii=False)
            print(f"✓ キーマップを {self.config_file} に保存しました")
            print(f"  検出キー数: {len(self.key_positions)}")
            return True
        except Exception as e:
            print(f"エラー: キーマップの保存に失敗しました: {e}")
            return False
    
    def load_map(self):
        """JSONファイルからキーマップを読み込み"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.key_positions = json.load(f)
            print(f"✓ {self.config_file} からキーマップを読み込みました")
            print(f"  読み込みキー数: {len(self.key_positions)}")
            return True
        except Exception as e:
            print(f"エラー: キーマップの読み込みに失敗しました: {e}")
            return False
    
    def setup_four_point_calibration(self):
        """4点クリックによる簡易キャリブレーション"""
    def setup_four_point_calibration(self):
        """4点クリックによる簡易キャリブレーション"""
        print("=== 4点クリック キャリブレーション ===")
        print("以下の4点をクリックしてください")
        print()
        print("クリック順序:")
        print("  1. 左上: 1キーの左上")
        print("  2. 右上: ハイフン(-)キーの右上（0の右の記号）")
        print("  3. 右下: スペースキーの右上")
        print("  4. 左下: スペースキーの左上")
        print()
        print("※ 文字キー4行は矩形内に、スペースは矩形の下に配置されます")
        print()
        print("操作:")
        print("  左クリック: 点を配置")
        print("  R: リセット")
        print("  SPACE: 確定")
        print("  ESC: キャンセル")
        
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("エラー: カメラを開けませんでした")
            return False
        
        cv2.namedWindow('4-Point Calibration')
        
        # キャリブレーションヘルパーをリセット
        self.calibration_helper.reset()
        clicked_count = [0]  # クリック数カウンタ
        
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                frame_shape = param['frame_shape']
                if self.calibration_helper.add_click(x, y, frame_shape):
                    print("✓ 4点指定完了！SPACEキーで確定してください")
                else:
                    clicked_count[0] = len(self.calibration_helper.clicked_points)
                    print(f"点 {clicked_count[0]}/4 を配置しました")
        
        param = {'frame_shape': None}
        cv2.setMouseCallback('4-Point Calibration', on_mouse_click, param)
        
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    break
                
                param['frame_shape'] = frame.shape[:2]
                
                # クリック点を描画
                display_frame = self.calibration_helper.draw_clicks(frame)
                
                # 4点揃ったらキー配置をプレビュー
                if len(self.calibration_helper.clicked_points) == 4:
                    key_positions = self.calibration_helper.get_key_positions()
                    if key_positions:
                        # キーをオーバーレイ表示
                        h, w = frame.shape[:2]
                        for key, pos in key_positions.items():
                            cx = int(pos['x'] * w)
                            cy = int(pos['y'] * h)
                            
                            if key == 'space':
                                # スペースキーは特別表示
                                width = int(pos['width'] * w)
                                height = int(pos['height'] * h)
                                cv2.rectangle(display_frame,
                                            (cx - width//2, cy - height//2),
                                            (cx + width//2, cy + height//2),
                                            (0, 255, 0), 1)
                                cv2.putText(display_frame, "SPACE", (cx - 30, cy + 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                            else:
                                # 通常のキー
                                cv2.circle(display_frame, (cx, cy), 2, (0, 200, 0), -1)
                                cv2.putText(display_frame, key, (cx - 5, cy + 3),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)
                
                # 情報表示
                info_text = f"Points: {len(self.calibration_helper.clicked_points)}/4"
                cv2.putText(display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if len(self.calibration_helper.clicked_points) == 4:
                    cv2.putText(display_frame, "Press SPACE to confirm", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('4-Point Calibration', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE: 確定
                    if len(self.calibration_helper.clicked_points) == 4:
                        key_positions = self.calibration_helper.get_key_positions()
                        if key_positions:
                            self.key_positions = key_positions
                            self.save_map()
                            print("✓ キャリブレーション完了！")
                            camera.release()
                            cv2.destroyAllWindows()  # 全ウィンドウを閉じる
                            return True
                    else:
                        print(f"まだ{4 - len(self.calibration_helper.clicked_points)}点必要です")
                
                elif key == ord('r') or key == ord('R'):  # R: リセット
                    self.calibration_helper.reset()
                    clicked_count[0] = 0
                    print("リセットしました")
                
                elif key == 27:  # ESC: キャンセル
                    print("キャリブレーションをキャンセルしました")
                    break
        
        finally:
            camera.release()
            cv2.destroyAllWindows()  # 全ウィンドウを閉じる
        
        return False
    
    def start_calibration(self):
        """キャリブレーション開始（4点クリックをデフォルトに）"""
        print("キーボードキャリブレーションを開始します")
        print(f"対象キー: {len(self.target_keys)}個")
        print()
        
        # 4点クリック方式をデフォルトに
        return self.setup_four_point_calibration()
    
    def get_nearest_key(self, x, y):
        """指定された座標に最も近いキーを返す（改良版）"""
        if not self.key_positions:
            return None, float('inf')
        
        # SimpleKeyboardMapperの検索機能を使用
        return self.simple_mapper.find_key_at_position(x, y, self.key_positions)
    
    def visualize(self, frame):
        """キーボードマップを画像上に可視化"""
        h, w, _ = frame.shape
        self.frame_shape = (h, w)
        
        for key, pos in self.key_positions.items():
            center_x = int(pos['x'] * w)
            center_y = int(pos['y'] * h)
            
            if 'width' in pos and 'height' in pos:
                width = int(pos['width'] * w)
                height = int(pos['height'] * h)
                
                x1 = center_x - width // 2
                y1 = center_y - height // 2
                x2 = center_x + width // 2
                y2 = center_y + height // 2
                
                if key == 'space':
                    # スペースキーは特別な色
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'SP', (center_x - 10, center_y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # 通常のキー
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
                    cv2.putText(frame, key, (center_x - 5, center_y + 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            else:
                # 旧形式（点のみ）の場合
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
                cv2.putText(frame, key, (center_x - 8, center_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    # 以下、互換性のために残す旧メソッド
    def setup_manual_calibration(self):
        """手動キャリブレーション（37キー全クリック）- 互換性のため残す"""
        print("=== 旧式：手動キャリブレーション（37キー） ===")
        print("4点クリック方式を推奨します。")
        print("続行しますか？ (y/n)")
        
        response = input().lower()
        if response != 'y':
            return self.setup_four_point_calibration()
        
        # 既存の37キー手動実装...（省略）
        return False