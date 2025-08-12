# src/input/keyboard_map.py - Enhanced with Gemini API integration
import cv2
import numpy as np
import json
import os
import time
from dotenv import load_dotenv

class KeyboardMap:
    def __init__(self, config_file='keyboard_map.json'):
        self.config_file = config_file
        self.key_positions = {}
        self.frame_shape = (720, 1280)  # デフォルトのフレームサイズ
        self.calibration_mode = False
        
        # .envファイルを読み込み
        load_dotenv()
        
        # Gemini API設定
        self.gemini_available = self._setup_gemini()
        
        # 現在の研究対象キー（KeyFormatterと一致）
        self.target_keys = self._get_target_keys()
        
        # 設定ファイルが存在する場合は読み込む
        if os.path.exists(config_file):
            self.load_map()
        else:
            # JIS配列キーボードの初期マップ
            self.initialize_jis_keyboard()
    
    def _setup_gemini(self):
        """Gemini API の初期化"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("警告: .envファイルにGEMINI_API_KEYが設定されていません")
                print("手動キャリブレーションのみ利用可能です")
                return False
            
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✓ Gemini API が利用可能です")
            return True
            
        except ImportError:
            print("警告: google-generativeaiがインストールされていません")
            print("pip install google-generativeai を実行してください")
            return False
        except Exception as e:
            print(f"警告: Gemini API初期化に失敗しました: {e}")
            return False
    
    def _get_target_keys(self):
        """現在の研究対象キーを取得（KeyFormatterと同期）"""
        target_keys = set()
        
        # 英字 (a-z)
        target_keys.update('abcdefghijklmnopqrstuvwxyz')
        
        # 数字 (0-9)  
        target_keys.update('0123456789')
        
        # スペース
        target_keys.add('space')
        
        return target_keys
    
    def initialize_jis_keyboard(self):
        """JIS配列キーボードの初期マップを設定"""
        # 研究対象キーのみの基本配置
        basic_layout = {
            # 数字行
            '1': {'x': 0.08, 'y': 0.2}, '2': {'x': 0.12, 'y': 0.2}, '3': {'x': 0.16, 'y': 0.2},
            '4': {'x': 0.20, 'y': 0.2}, '5': {'x': 0.24, 'y': 0.2}, '6': {'x': 0.28, 'y': 0.2},
            '7': {'x': 0.32, 'y': 0.2}, '8': {'x': 0.36, 'y': 0.2}, '9': {'x': 0.40, 'y': 0.2},
            '0': {'x': 0.44, 'y': 0.2},
            
            # QWERTY行
            'q': {'x': 0.10, 'y': 0.35}, 'w': {'x': 0.14, 'y': 0.35}, 'e': {'x': 0.18, 'y': 0.35},
            'r': {'x': 0.22, 'y': 0.35}, 't': {'x': 0.26, 'y': 0.35}, 'y': {'x': 0.30, 'y': 0.35},
            'u': {'x': 0.34, 'y': 0.35}, 'i': {'x': 0.38, 'y': 0.35}, 'o': {'x': 0.42, 'y': 0.35},
            'p': {'x': 0.46, 'y': 0.35},
            
            # ASDF行
            'a': {'x': 0.12, 'y': 0.50}, 's': {'x': 0.16, 'y': 0.50}, 'd': {'x': 0.20, 'y': 0.50},
            'f': {'x': 0.24, 'y': 0.50}, 'g': {'x': 0.28, 'y': 0.50}, 'h': {'x': 0.32, 'y': 0.50},
            'j': {'x': 0.36, 'y': 0.50}, 'k': {'x': 0.40, 'y': 0.50}, 'l': {'x': 0.44, 'y': 0.50},
            
            # ZXCV行
            'z': {'x': 0.16, 'y': 0.65}, 'x': {'x': 0.20, 'y': 0.65}, 'c': {'x': 0.24, 'y': 0.65},
            'v': {'x': 0.28, 'y': 0.65}, 'b': {'x': 0.32, 'y': 0.65}, 'n': {'x': 0.36, 'y': 0.65},
            'm': {'x': 0.40, 'y': 0.65},
            
            # スペース
            'space': {'x': 0.30, 'y': 0.80}
        }
        
        self.key_positions = basic_layout
    
    def detect_keyboard_with_gemini(self, image_data):
        """Gemini APIを使用してキーボード検出"""
        if not self.gemini_available:
            raise Exception("Gemini API が利用できません")
        
        # 研究対象キーに絞ったプロンプト
        target_keys_str = ', '.join(sorted(self.target_keys))
        
        prompt = f"""
        この画像にはキーボードがあります。以下の手順で正確にキー位置を検出してください：

        対象キー: {target_keys_str}

        手順：
        1. キーボードの物理的な境界を特定
        2. キーボードを行に分割：
           - 最上段：数字キー（1,2,3,4,5,6,7,8,9,0）
           - 2段目：QWERTY行（q,w,e,r,t,y,u,i,o,p）
           - 3段目：ASDF行（a,s,d,f,g,h,j,k,l）
           - 4段目：ZXCV行（z,x,c,v,b,n,m）
           - 最下段：スペースキー（space）
        3. 各キーの文字を読み取り、対象キーのみ選択
        4. 各キーの正確な矩形境界を測定

        重要：
        - 画像の実際のキーボード位置から測定（推測禁止）
        - キーキャップの物理的境界を正確に
        - 数字は最上段、英字は該当する行に配置

        出力（JSONのみ）：
        {{
            "keys": {{
                "1": {{"x1": 実測値, "y1": 実測値, "x2": 実測値, "y2": 実測値}},
                "q": {{"x1": 実測値, "y1": 実測値, "x2": 実測値, "y2": 実測値}},
                "space": {{"x1": 実測値, "y1": 実測値, "x2": 実測値, "y2": 実測値}}
            }}
        }}

        座標は0.0-1.0の正規化座標で、x1,y1=左上角、x2,y2=右下角
        """
        
        try:
            # Gemini APIに送信
            response = self.gemini_model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_data}
            ])
            
            # レスポンス解析
            response_text = response.text.strip()
            
            # JSONを抽出
            if '```json' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            result = json.loads(json_text)
            
            # 結果の検証とフィルタリング
            if 'keys' in result:
                filtered_keys = {}
                for key, coords in result['keys'].items():
                    if key in self.target_keys:
                        # 座標の妥当性チェック
                        if self._validate_coordinates(coords):
                            filtered_keys[key] = coords
                        else:
                            print(f"警告: キー '{key}' の座標が無効です: {coords}")
                    else:
                        print(f"情報: 対象外キー '{key}' をスキップしました")
                
                result['keys'] = filtered_keys
                print(f"✓ Gemini検出成功: {len(filtered_keys)}個のキーを検出")
                return result
            else:
                print("エラー: Geminiレスポンスに'keys'フィールドがありません")
                return None
                
        except json.JSONDecodeError as e:
            print(f"エラー: Gemini APIのレスポンスがJSONとして解析できません: {e}")
            print(f"レスポンス: {response_text[:200]}...")
            return None
        except Exception as e:
            print(f"エラー: Gemini API呼び出しに失敗しました: {e}")
            return None
    
    def _validate_coordinates(self, coords):
        """座標の妥当性をチェック（強化版）"""
        required_keys = ['x1', 'y1', 'x2', 'y2']
        
        # 必要なキーが存在するかチェック
        for key in required_keys:
            if key not in coords:
                return False
        
        # 座標値の範囲チェック（0.0-1.0）
        for key in required_keys:
            val = coords[key]
            if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
                return False
        
        # 論理的整合性チェック（左上 < 右下）
        if coords['x1'] >= coords['x2'] or coords['y1'] >= coords['y2']:
            return False
        
        # サイズの妥当性チェック（キーが極端に小さい/大きくないか）
        width = coords['x2'] - coords['x1']
        height = coords['y2'] - coords['y1']
        
        # 通常のキーサイズ範囲（経験値）
        if width < 0.01 or width > 0.3:  # 幅が1%未満 or 30%超は異常
            return False
        if height < 0.01 or height > 0.15:  # 高さが1%未満 or 15%超は異常
            return False
        
        return True
    
    def setup_gemini_calibration(self):
        """Gemini APIを使用した自動キーボード検出"""
        if not self.gemini_available:
            print("Gemini APIが利用できません。手動キャリブレーションを実行します。")
            return self.setup_manual_calibration()
        
        print("=== Gemini自動キーボード検出 ===")
        print("カメラでキーボードを撮影し、AIが自動でキー位置を検出します")
        print()
        print("操作方法:")
        print("- キーボード全体がカメラに映るように調整してください")
        print("- SPACEキー: 撮影・検出実行")
        print("- ESCキー: 手動キャリブレーションに切り替え") 
        print("- Qキー: 終了")
        
        # カメラ初期化
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("エラー: カメラを開けませんでした")
            return False
        
        cv2.namedWindow('Gemini Keyboard Detection')
        
        detection_result = None
        
        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("エラー: カメラからフレームを取得できませんでした")
                    break
                
                # フレーム情報を更新
                self.frame_shape = frame.shape
                
                # 指示を画面に表示
                display_frame = frame.copy()
                
                # 背景矩形
                cv2.rectangle(display_frame, (10, 10), (600, 120), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (10, 10), (600, 120), (0, 255, 0), 2)
                
                # テキスト表示
                cv2.putText(display_frame, "Gemini Keyboard Detection", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, "SPACE: Capture & Detect  ESC: Manual Mode  Q: Quit", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Target Keys: {len(self.target_keys)} keys (a-z, 0-9, space)", (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(display_frame, f"Gemini API: {'Ready' if self.gemini_available else 'Not Available'}", (20, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.gemini_available else (0, 0, 255), 1)
                
                cv2.imshow('Gemini Keyboard Detection', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # スペースキー: 検出実行
                    print("📸 キーボードを撮影中...")
                    
                    # 画像をエンコード
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_data = buffer.tobytes()
                    
                    print("🤖 Gemini APIで解析中...")
                    
                    # Gemini APIで検出
                    detection_result = self.detect_keyboard_with_gemini(image_data)
                    
                    if detection_result and 'keys' in detection_result:
                        detected_count = len(detection_result['keys'])
                        target_count = len(self.target_keys)
                        
                        print(f"✓ 検出完了: {detected_count}/{target_count} キーを検出")
                        
                        # 結果を可視化
                        self._visualize_detection_result(frame, detection_result)
                        
                        # 保存するか確認
                        print("\n結果を保存しますか？ (y/n/r)")
                        print("y: 保存して完了  n: 再撮影  r: 再検出")
                        
                        while True:
                            confirm_key = cv2.waitKey(0) & 0xFF
                            if confirm_key == ord('y'):
                                # 座標を変換して保存
                                if self._convert_and_save_detection(detection_result):
                                    print("✓ キーボードマップを保存しました")
                                    return True
                                else:
                                    print("❌ 保存に失敗しました")
                                    break
                            elif confirm_key == ord('n') or confirm_key == ord('r'):
                                print("再撮影します...")
                                break
                            elif confirm_key == 27:  # ESC
                                break
                    else:
                        print("❌ 検出に失敗しました。再撮影してください。")
                        print("💡 ヒント: キーボード全体がはっきり見えるように調整してください")
                
                elif key == 27:  # ESCキー: 手動モードに切り替え
                    print("手動キャリブレーションモードに切り替えます...")
                    camera.release()
                    cv2.destroyWindow('Gemini Keyboard Detection')
                    return self.setup_manual_calibration()
                    
                elif key == ord('q'):  # Qキー: 終了
                    break
        
        finally:
            camera.release()
            cv2.destroyWindow('Gemini Keyboard Detection')
        
        return False
    
    def _visualize_detection_result(self, frame, detection_result):
        """検出結果を可視化"""
        if not detection_result or 'keys' not in detection_result:
            return
        
        display_frame = frame.copy()
        h, w, _ = frame.shape
        
        for key, coords in detection_result['keys'].items():
            # 矩形座標を画像座標に変換
            x1 = int(coords['x1'] * w)
            y1 = int(coords['y1'] * h)
            x2 = int(coords['x2'] * w)
            y2 = int(coords['y2'] * h)
            
            # 矩形を描画
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # キー名を描画
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.putText(display_frame, key, (center_x - 10, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 結果ウィンドウを表示
        cv2.namedWindow('Detection Result')
        cv2.imshow('Detection Result', display_frame)
        print("検出結果を確認してください（別ウィンドウ）")
    
    def _convert_and_save_detection(self, detection_result):
        """検出結果を既存フォーマットに変換して保存"""
        if not detection_result or 'keys' not in detection_result:
            return False
        
        try:
            converted_positions = {}
            
            for key, coords in detection_result['keys'].items():
                # 中心座標を計算
                center_x = (coords['x1'] + coords['x2']) / 2
                center_y = (coords['y1'] + coords['y2']) / 2
                
                # 幅と高さを計算
                width = coords['x2'] - coords['x1']
                height = coords['y2'] - coords['y1']
                
                converted_positions[key] = {
                    'x': center_x,
                    'y': center_y,
                    'width': width,
                    'height': height
                }
            
            # 検出されなかったキーの警告
            missing_keys = self.target_keys - set(converted_positions.keys())
            if missing_keys:
                print(f"警告: 以下のキーが検出されませんでした: {sorted(missing_keys)}")
            
            # 保存
            self.key_positions = converted_positions
            self.save_map()
            return True
            
        except Exception as e:
            print(f"エラー: 検出結果の変換に失敗しました: {e}")
            return False
    
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
            self.initialize_jis_keyboard()
            return False
    
    def get_nearest_key(self, x, y):
        """指定された座標に最も近いキーを返す（既存の実装を維持）"""
        min_distance = float('inf')
        nearest_key = None
        
        for key, pos in self.key_positions.items():
            # 中心座標ベースの距離計算
            distance = ((pos['x'] - x) ** 2 + (pos['y'] - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_key = key
                
        return nearest_key, min_distance
    
    def visualize(self, frame):
        """キーボードマップを画像上に可視化（既存の実装を拡張）"""
        h, w, _ = frame.shape
        self.frame_shape = (h, w)
        
        # 各キーの位置を描画
        for key, pos in self.key_positions.items():
            # 正規化座標を画像サイズに変換
            center_x = int(pos['x'] * w)
            center_y = int(pos['y'] * h)
            
            # 幅と高さがある場合は矩形で表示
            if 'width' in pos and 'height' in pos:
                width = int(pos['width'] * w)
                height = int(pos['height'] * h)
                
                x1 = center_x - width // 2
                y1 = center_y - height // 2
                x2 = center_x + width // 2
                y2 = center_y + height // 2
                
                # 矩形を描画
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # 古い形式（点のみ）の場合は円で表示
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
            
            # キー名を表示
            if key == 'space':
                display_key = 'SP'
            else:
                display_key = key
            
            cv2.putText(frame, display_key, (center_x - 8, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def setup_manual_calibration(self):
        """マウスクリックによる手動キャリブレーション（既存実装を改良）"""
        print("=== 手動キーボードキャリブレーション ===")
        print("対象キーを順番にクリックしてください")
        print()
        
        # 対象キーを論理的順序でソート
        sorted_keys = []
        
        # 数字 (0-9)
        for i in range(10):
            key = str(i)
            if key in self.target_keys:
                sorted_keys.append(key)
        
        # 英字 (a-z)
        for char in 'abcdefghijklmnopqrstuvwxyz':
            if char in self.target_keys:
                sorted_keys.append(char)
        
        # スペース
        if 'space' in self.target_keys:
            sorted_keys.append('space')
        
        print(f"キー順序: {' → '.join(sorted_keys)}")
        print(f"総数: {len(sorted_keys)}個")
        
        # カメラセットアップ
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Manual Calibration')
        
        current_key_index = 0
        positions = {}
        frame_shape = [None]
        
        def on_mouse_click(event, x, y, flags, param):
            nonlocal current_key_index
            
            if event == cv2.EVENT_LBUTTONDOWN and frame_shape[0] is not None:
                h, w, _ = frame_shape[0]
                
                # クリック位置を記録（正規化座標）
                normalized_x = x / w
                normalized_y = y / h
                
                current_key = sorted_keys[current_key_index]
                
                # キー位置を更新（点座標として保存）
                positions[current_key] = {'x': normalized_x, 'y': normalized_y}
                print(f"✓ キー '{current_key}' の位置を記録: ({normalized_x:.3f}, {normalized_y:.3f})")
                
                # 次のキーへ
                current_key_index += 1
        
        cv2.setMouseCallback('Manual Calibration', on_mouse_click)
        
        try:
            while current_key_index < len(sorted_keys):
                ret, frame = camera.read()
                if not ret:
                    break
                
                frame_shape[0] = frame.shape
                
                # 既に記録された位置を可視化
                for key, pos in positions.items():
                    x, y = int(pos['x'] * frame.shape[1]), int(pos['y'] * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, key, (x-5, y+15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 現在のキーを表示
                current_key = sorted_keys[current_key_index]
                progress = f"{current_key_index+1}/{len(sorted_keys)}"
                
                # UI表示
                cv2.rectangle(frame, (10, 10), (500, 80), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (500, 80), (0, 255, 0), 2)
                cv2.putText(frame, f"Click key: {current_key.upper()}", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Progress: {progress}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Manual Calibration', frame)
                
                # ESCキーで終了
                if cv2.waitKey(1) & 0xFF == 27:
                    print("手動キャリブレーションを中断しました")
                    return False
        
        finally:
            camera.release()
            cv2.destroyWindow('Manual Calibration')
        
        # 結果を保存
        if len(positions) > 0:
            self.key_positions = positions
            self.save_map()
            print(f"✓ 手動キャリブレーション完了: {len(positions)}個のキーを記録")
            return True
        else:
            print("❌ キャリブレーションが完了していません")
            return False
    
    def start_calibration(self):
        """キャリブレーション開始（自動でGemini→手動の順に試行）"""
        print("キーボードキャリブレーションを開始します")
        print(f"対象キー: {len(self.target_keys)}個 ({', '.join(sorted(self.target_keys))})")
        print()
        
        if self.gemini_available:
            print("💡 Gemini自動検出を使用します（推奨）")
            if self.setup_gemini_calibration():
                return True
            else:
                print("Gemini検出に失敗しました。手動キャリブレーションに切り替えます。")
        else:
            print("💡 手動キャリブレーションを使用します")
        
        return self.setup_manual_calibration()