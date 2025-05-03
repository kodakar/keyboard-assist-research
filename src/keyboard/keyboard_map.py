# src/keyboard_map.py を修正（OCR機能を統合）
import cv2
import numpy as np
import pytesseract
import json
import os

class KeyboardMap:
    def __init__(self, config_file='keyboard_map.json'):
        self.config_file = config_file
        self.key_positions = {}
        self.frame_shape = (720, 1280)  # デフォルトのフレームサイズ
        self.calibration_mode = False
        
        # Windowsの場合はTesseractのパスを設定
        # Macやlinuxでは不要な場合があります
        if os.name == 'nt':  # Windowsの場合
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # 設定ファイルが存在する場合は読み込む
        if os.path.exists(config_file):
            self.load_map()
        else:
            # JIS配列キーボードの初期マップ
            self.initialize_jis_keyboard()
    
    def initialize_jis_keyboard(self):
        """JIS配列キーボードの初期マップを設定"""
        # 1段目 - 数字キー行
        row1 = ['Key.esc', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '^', ']']
        
        # 2段目 - QWERTY行
        row2 = ['Key.tab', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '@', '[', 'Key.backspace']
        
        # 3段目 - ASDF行
        row3 = ['Key.ctrl_l', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', ':', 'Key.enter']
        
        # 4段目 - ZXCV行
        row4 = ['Key.shift', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 'Key.shift_r']
        
        # 5段目 - スペース行
        row5 = ['Key.alt_l', 'Key.cmd', 'Key.space', 'Key.cmd_r', 'Key.alt_gr']
        
        # 各キーの初期位置を設定（後でキャリブレーションで更新される）
        y_start = 0.1
        y_step = 0.15
        
        # 各行のキー位置を仮設定
        all_rows = [row1, row2, row3, row4, row5]
        for i, row in enumerate(all_rows):
            y = y_start + i * y_step
            x_start = 0.05
            x_step = 0.85 / len(row)  # 行の長さに応じて均等に配置
            
            for j, key in enumerate(row):
                x = x_start + j * x_step
                self.key_positions[key] = {'x': x, 'y': y}
    
    def start_calibration(self):
        """キャリブレーションモードを開始"""
        self.calibration_mode = True
        print("キーボードキャリブレーションを開始します")
        print("各キーを順番に押してください")
    
    def detect_keyboard_ocr(self, frame):
        """OCRを使用してキーボードのキーを検出"""
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 閾値処理でコントラストを強調
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # ノイズ除去
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 検出したキーの情報を保存
        detected_keys = []
        
        # 各輪郭を処理
        for contour in contours:
            # 小さすぎる輪郭は無視
            if cv2.contourArea(contour) < 100:
                continue
                
            # 輪郭の外接矩形を取得
            x, y, w, h = cv2.boundingRect(contour)
            
            # キーの候補領域を切り出し
            key_roi = gray[y:y+h, x:x+w]
            
            # OCRでテキスト認識
            config = '--psm 10'  # 単一文字モード
            text = pytesseract.image_to_string(key_roi, config=config).strip()
            
            # 空でなければキー情報を追加
            if text and len(text) == 1:  # 単一文字のみ対象
                key = text.lower()  # 小文字に統一
                detected_keys.append({
                    'key': key,
                    'x': (x + w/2) / frame.shape[1],  # 正規化座標
                    'y': (y + h/2) / frame.shape[0]
                })
                
                # フレームに検出結果を描画
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, key, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # キー位置を更新
                self.key_positions[key] = {
                    'x': (x + w/2) / frame.shape[1],
                    'y': (y + h/2) / frame.shape[0]
                }
        
        # 結果を保存
        if detected_keys:
            self.save_map()
        
        return frame, detected_keys
    
    def update_from_collected_data(self, collected_data):
        """収集したデータからキーボードマップを更新"""
        for sample in collected_data:
            key = sample['key']
            # 人差し指の先端（ランドマーク8）の位置を取得
            landmarks = np.array(sample['hand_position']).reshape(-1, 3)
            index_finger_tip = landmarks[8]  # インデックス8が人差し指の先端
            
            # キー位置を更新
            self.key_positions[key] = {
                'x': float(index_finger_tip[0]),
                'y': float(index_finger_tip[1])
            }
        
        # 更新したマップを保存
        self.save_map()
    
    def save_map(self):
        """キーマップをJSONファイルに保存"""
        with open(self.config_file, 'w') as f:
            json.dump(self.key_positions, f, indent=2)
        print(f"キーマップを {self.config_file} に保存しました")
    
    def load_map(self):
        """JSONファイルからキーマップを読み込み"""
        try:
            with open(self.config_file, 'r') as f:
                self.key_positions = json.load(f)
            print(f"{self.config_file} からキーマップを読み込みました")
            return True
        except Exception as e:
            print(f"キーマップの読み込みに失敗しました: {e}")
            self.initialize_jis_keyboard()
            return False
    
    def get_nearest_key(self, x, y):
        """指定された座標に最も近いキーを返す"""
        min_distance = float('inf')
        nearest_key = None
        
        for key, pos in self.key_positions.items():
            distance = ((pos['x'] - x) ** 2 + (pos['y'] - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_key = key
                
        return nearest_key, min_distance
    
    def visualize(self, frame):
        """キーボードマップを画像上に可視化"""
        h, w, _ = frame.shape
        self.frame_shape = (h, w)
        
        # 各キーの位置を描画
        for key, pos in self.key_positions.items():
            # 正規化座標を画像サイズに変換
            x, y = int(pos['x'] * w), int(pos['y'] * h)
            
            # キーの種類によって表示方法を変える
            if key.startswith('Key.'):
                # 特殊キーは大きめの四角で表示
                cv2.rectangle(frame, (x-20, y-10), (x+20, y+10), (0, 255, 0), 2)
                # キー名を短く表示
                short_name = key.replace('Key.', '')
                cv2.putText(frame, short_name, (x-15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # 通常のキーは円で表示
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, key, (x-5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


    def setup_manual_calibration(self):
        """マウスクリックによる手動キャリブレーション"""
        print("手動キャリブレーションを開始します")
        print("順番にキーをクリックしてください")
        
        # キーの順序（JIS配列）
        key_rows = [
            # 1段目
            ['Key.esc', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '^', ']'],
            # 2段目 - QWERTY行
            ['Key.tab', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '@', '[', 'Key.backspace'],
            # 3段目 - ASDF行
            ['Key.ctrl_l', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', ':', 'Key.enter'],
            # 4段目 - ZXCV行
            ['Key.shift', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 'Key.shift_r'],
            # 5段目 - スペース行
            ['Key.alt_l', 'Key.cmd', 'Key.space', 'Key.cmd_r', 'Key.alt_gr']
        ]
        
        # 全キーのリスト
        all_keys = []
        for row in key_rows:
            all_keys.extend(row)
        
        # カメラセットアップ
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Manual Calibration')
        
        current_key_index = 0
        current_key = all_keys[current_key_index]
        positions = {}
        
        # フレームの形状を共有するための変数
        frame_shape = [None]  # リストで包むことで参照渡しにする
        
        def on_mouse_click(event, x, y, flags, param):
            nonlocal current_key_index, current_key
            
            if event == cv2.EVENT_LBUTTONDOWN and frame_shape[0] is not None:
                h, w, _ = frame_shape[0]
                
                # クリック位置を記録（正規化座標）
                normalized_x = x / w
                normalized_y = y / h
                
                # キー位置を更新
                positions[current_key] = {'x': normalized_x, 'y': normalized_y}
                print(f"キー '{current_key}' の位置を記録: ({normalized_x:.3f}, {normalized_y:.3f})")
                
                # 次のキーへ
                current_key_index += 1
                if current_key_index < len(all_keys):
                    current_key = all_keys[current_key_index]
                else:
                    # すべてのキーが完了
                    print("すべてのキーの位置を記録しました")
        
        cv2.setMouseCallback('Manual Calibration', on_mouse_click)
        
        # キャリブレーションループ
        while current_key_index < len(all_keys):
            ret, frame = camera.read()
            if not ret:
                break
            
            # フレームの形状を更新
            frame_shape[0] = frame.shape
            
            # 位置を可視化
            for key, pos in positions.items():
                x, y = int(pos['x'] * frame.shape[1]), int(pos['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, key, (x-5, y+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 現在のキーを表示
            cv2.putText(frame, f"クリックでキー '{current_key}' の位置を記録 ({current_key_index+1}/{len(all_keys)})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Manual Calibration', frame)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # リソースを解放
        camera.release()
        cv2.destroyWindow('Manual Calibration')
        
        # キー位置を保存
        self.key_positions = positions
        self.save_map()
        
        return True