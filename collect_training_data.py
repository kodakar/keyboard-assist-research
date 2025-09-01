#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実践的なデータ収集スクリプト
ユーザーがキーボード入力を行い、学習用データを収集
"""

import cv2
import numpy as np
import argparse
import os
import json
import time
from datetime import datetime
from collections import deque
import sys

# 既存のモジュールをインポート
sys.path.append('src')
from src.core.camera import Camera
from src.core.hand_tracker import HandTracker
from src.input.keyboard_map import KeyboardMap
from src.input.keyboard_tracker import KeyboardTracker
from src.processing.enhanced_data_collector import EnhancedDataCollector


class TrainingDataCollector:
    """学習用データ収集クラス"""
    
    def __init__(self, user_id: str, session_text: str, repetitions: int):
        """
        データ収集クラスの初期化
        
        Args:
            user_id: ユーザーID
            session_text: 入力してもらうテキスト
            repetitions: 繰り返し回数
        """
        self.user_id = user_id
        self.session_text = session_text
        self.repetitions = repetitions
        
        # セッション情報
        self.current_repetition = 0
        self.current_char_index = 0
        self.correct_inputs = 0
        self.total_inputs = 0
        
        # コンポーネントの初期化
        self.camera = None
        self.hand_tracker = None
        self.keyboard_map = None
        self.keyboard_tracker = None
        self.data_collector = None
        
        # データ収集の状態
        self.is_collecting = False
        self.collection_start_time = None
        self.trajectory_buffer = deque(maxlen=60)  # 2秒分（30fps × 2秒）
        
        # セッションデータの保存先
        self.session_dir = self._create_session_directory()
        
        print(f"🎯 データ収集セッション開始")
        print(f"   ユーザーID: {user_id}")
        print(f"   目標テキスト: {session_text}")
        print(f"   繰り返し回数: {repetitions}")
        print(f"   セッションディレクトリ: {self.session_dir}")
    
    def _create_session_directory(self) -> str:
        """セッションディレクトリを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join("data", "training", self.user_id, f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def initialize_components(self) -> bool:
        """コンポーネントの初期化"""
        try:
            print("🔧 コンポーネントを初期化中...")
            
            # カメラの初期化
            self.camera = Camera()
            if not self.camera.is_opened():
                print("❌ カメラの初期化に失敗しました")
                return False
            
            # 画面サイズを設定
            frame = self.camera.read_frame()
            if frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ カメラ初期化完了: {width}x{height}")
            else:
                print("❌ カメラからのフレーム取得に失敗しました")
                return False
            
            # 手追跡の初期化
            self.hand_tracker = HandTracker()
            print("✅ 手追跡初期化完了")
            
            # キーボードマップの初期化
            self.keyboard_map = KeyboardMap()
            
            # keyboard_map.jsonファイルの存在確認
            keyboard_map_file = "keyboard_map.json"
            if os.path.exists(keyboard_map_file):
                print("\n📁 保存済みのキーボード設定ファイル (keyboard_map.json) が見つかりました。")
                print("\n1: 保存した設定を再利用する")
                print("2: 新しくキャリブレーションをやり直す")
                
                while True:
                    try:
                        choice = input("\nどちらにしますか？ (1/2): ").strip()
                        
                        if choice == "1":
                            print("✅ 保存した設定を再利用します。")
                            # KeyboardMapのコンストラクタが自動で読み込んでいるため、追加処理は不要
                            break
                        elif choice == "2":
                            print("🔄 新しいキャリブレーションを開始します...")
                            if not self.keyboard_map.start_calibration(self.camera):
                                print("❌ キーボードマッピングに失敗しました")
                                return False
                            
                            # キャリブレーション完了後のカメラ状態確認・復旧
                            print("🔄 キャリブレーション完了後のカメラ状態を確認中...")
                            if not self._verify_camera_after_calibration():
                                print("❌ キャリブレーション後のカメラ状態確認に失敗しました")
                                return False
                            break
                        else:
                            print("❌ 無効な選択です。1 または 2 を入力してください。")
                            
                    except KeyboardInterrupt:
                        print("\n❌ 無効な選択です。プログラムを終了します。")
                        return False
                    except EOFError:
                        print("\n❌ 無効な選択です。プログラムを終了します。")
                        return False
            else:
                print("⚠️ キーボード設定ファイルが見つかりません。")
                print("   新しいキャリブレーションを開始します...")
                if not self.keyboard_map.start_calibration(self.camera):
                    print("❌ キーボードマッピングに失敗しました")
                    return False
                
                # キャリブレーション完了後のカメラ状態確認・復旧
                print("🔄 キャリブレーション完了後のカメラ状態を確認中...")
                if not self._verify_camera_after_calibration():
                    print("❌ キャリブレーション後のカメラ状態確認に失敗しました")
                    return False
            
            # キーボードトラッカーの初期化
            self.keyboard_tracker = KeyboardTracker()
            self.keyboard_tracker.start()
            print("✅ キーボードトラッカー初期化完了")
            
            # データ収集の初期化
            self.data_collector = EnhancedDataCollector(user_id=self.user_id)
            self.data_collector.set_screen_size(width, height)
            
            # 作業領域の4隅を設定
            work_area_corners = self.keyboard_map.get_work_area_corners()
            if work_area_corners is not None:
                self.data_collector.set_work_area_corners(work_area_corners)
                print("✅ 作業領域の4隅を設定しました")
            else:
                print("⚠️ 作業領域の4隅の取得に失敗しました")
                return False
            
            print("✅ データ収集初期化完了")
            
            return True
            
        except Exception as e:
            print(f"❌ コンポーネント初期化エラー: {e}")
            return False
    
    def _verify_camera_after_calibration(self) -> bool:
        """キャリブレーション完了後のカメラ状態を確認・復旧"""
        try:
            print("   カメラの状態を確認中...")
            
            # より長い待機時間でカメラが安定するのを待つ
            print("   ⏳ カメラの安定化を待機中...")
            time.sleep(3)
            
            # カメラが開いているか確認
            if not self.camera.is_opened():
                print("   ⚠️ カメラが閉じられています。再初期化します...")
                self.camera = Camera()
                if not self.camera.is_opened():
                    print("   ❌ カメラの再初期化に失敗しました")
                    return False
            
            # フレーム取得テスト（より多くの試行）
            print("   📷 フレーム取得テスト中...")
            for attempt in range(5):  # 最大5回試行
                frame = self.camera.read_frame()
                if frame is not None:
                    height, width = frame.shape[:2]
                    print(f"   ✅ カメラ状態確認完了: {width}x{height}")
                    
                    # 追加の安定性テスト
                    print("   🔍 追加の安定性テスト中...")
                    stable_frames = 0
                    for stability_test in range(3):
                        test_frame = self.camera.read_frame()
                        if test_frame is not None:
                            stable_frames += 1
                        time.sleep(0.2)
                    
                    if stable_frames >= 2:  # 3回中2回以上成功
                        print(f"   ✅ カメラの安定性確認完了 ({stable_frames}/3)")
                        return True
                    else:
                        print(f"   ⚠️ カメラの安定性が不十分 ({stable_frames}/3)")
                        continue
                else:
                    print(f"   ⚠️ フレーム取得失敗 (試行 {attempt + 1}/5)")
                    time.sleep(1)  # より長い待機時間
            
            print("   ❌ フレーム取得に失敗しました")
            return False
            
        except Exception as e:
            print(f"   ❌ カメラ状態確認エラー: {e}")
            return False
    
    def _verify_camera_before_collection(self) -> bool:
        """データ収集開始前のカメラ状態を確認"""
        try:
            print("      🔧 カメラの状態を詳細確認中...")
            
            # カメラが開いているか確認
            if not self.camera.is_opened():
                print("      ⚠️ カメラが閉じられています。再初期化します...")
                self.camera = Camera()
                if not self.camera.is_opened():
                    print("      ❌ カメラの再初期化に失敗しました")
                    return False
            
            # より長い待機時間でカメラの安定化を待つ
            print("      ⏳ カメラの安定化を待機中...")
            time.sleep(2)
            
            # フレーム取得テスト（より多くの試行）
            print("      📷 フレーム取得テスト中...")
            for attempt in range(5):  # 最大5回試行
                frame = self.camera.read_frame()
                if frame is not None:
                    height, width = frame.shape[:2]
                    print(f"      ✅ カメラ状態確認完了: {width}x{height}")
                    
                    # 追加の安定性テスト
                    print("      🔍 追加の安定性テスト中...")
                    stable_frames = 0
                    for stability_test in range(3):
                        test_frame = self.camera.read_frame()
                        if test_frame is not None:
                            stable_frames += 1
                        time.sleep(0.1)
                    
                    if stable_frames >= 2:  # 3回中2回以上成功
                        print(f"      ✅ カメラの安定性確認完了 ({stable_frames}/3)")
                        return True
                    else:
                        print(f"      ⚠️ カメラの安定性が不十分 ({stable_frames}/3)")
                        continue
                else:
                    print(f"      ⚠️ フレーム取得失敗 (試行 {attempt + 1}/5)")
                    time.sleep(1)  # より長い待機時間
            
            print("      ❌ フレーム取得に失敗しました")
            return False
            
        except Exception as e:
            print(f"      ❌ カメラ状態確認エラー: {e}")
            return False
    
    def run_data_collection(self) -> bool:
        """データ収集の実行"""
        try:
            print("🚀 データ収集を開始します")
            print("   操作説明:")
            print("   - SPACE: データ収集開始/停止")
            print("   - R: 現在の文字をリトライ")
            print("   - ESC: 終了")
            
            # メインループ
            while self.current_repetition < self.repetitions:
                if not self._run_single_repetition():
                    break
                
                self.current_repetition += 1
                if self.current_repetition < self.repetitions:
                    print(f"\n🔄 {self.current_repetition}/{self.repetitions} 回目完了")
                    print("   次の繰り返しを開始します...")
                    time.sleep(2)
            
            # 最終結果を表示
            self._show_final_results()
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ ユーザーによって中断されました")
            return False
        except Exception as e:
            print(f"❌ データ収集エラー: {e}")
            return False
        finally:
            self._cleanup()
    
    def _run_single_repetition(self) -> bool:
        """1回の繰り返しを実行"""
        print(f"\n📝 {self.current_repetition + 1}回目の入力開始")
        print(f"   目標テキスト: {self.session_text}")
        
        # データ収集開始前のカメラ最終確認
        print("   🔍 データ収集開始前のカメラ状態を確認中...")
        if not self._verify_camera_before_collection():
            print("   ❌ データ収集開始前のカメラ状態確認に失敗しました")
            return False
        
        # データ収集セッションを開始
        self.data_collector.start_collection_session(self.session_text)
        self.current_char_index = 0
        self.is_collecting = True
        self.collection_start_time = datetime.now()  # time.time()ではなくdatetime.now()
        
        # 文字入力ループ
        while self.current_char_index < len(self.session_text):
            if not self._process_single_character():
                return False
        
        # データ収集セッションを停止
        self.data_collector.stop_collection_session()
        self.is_collecting = False
        
        # 結果を表示
        accuracy = (self.correct_inputs / self.total_inputs * 100) if self.total_inputs > 0 else 0
        print(f"✅ {self.current_repetition + 1}回目完了 - 正解率: {accuracy:.1f}%")
        
        return True
    
    def _process_single_character(self) -> bool:
        """1文字の入力処理"""
        target_char = self.session_text[self.current_char_index]
        print(f"   次に入力する文字: '{target_char}' (位置: {self.current_char_index + 1}/{len(self.session_text)})")
        
        # フレーム処理ループ
        frame_count = 0
        start_time = time.time()
        
        while True:
            # フレーム取得（リトライ処理付き）
            frame = None
            for retry in range(3):  # 最大3回リトライ
                frame = self.camera.read_frame()
                if frame is not None:
                    break
                else:
                    print(f"   ⚠️ フレーム取得失敗 (リトライ {retry + 1}/3)")
                    time.sleep(0.5)
            
            if frame is None:
                print("❌ カメラからのフレーム取得に失敗しました（3回リトライ後）")
                return False
            
            frame_count += 1
            current_time = datetime.now()  # time.time()ではなくdatetime.now()
            
            # 手の検出
            results = self.hand_tracker.detect_hands(frame)
            
            # 手の軌跡データを収集
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.data_collector.add_hand_position(hand_landmarks, current_time.timestamp())
                
                # 手のランドマークを描画
                self.hand_tracker.draw_landmarks(frame, results)
            
            # キーボードマップを可視化
            frame = self.keyboard_map.visualize(frame)
            
            # 画面表示の更新
            frame = self._update_display(frame, target_char)
            
            # 画面に表示
            cv2.imshow('Training Data Collection', frame)
            
            # キー入力の処理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return False
            elif key == ord('r') or key == ord('R'):  # Rキーでリトライ
                print(f"   リトライ: '{target_char}' を再度入力してください")
                continue
            
            # キーボード入力の検出
            keyboard_key = self.keyboard_tracker.get_key_event()
            if keyboard_key:
                if self._process_keyboard_input(keyboard_key, target_char, current_time):
                    break
            
            # FPS制限
            if frame_count % 30 == 0:  # 30fps
                time.sleep(0.01)
        
        return True
    
    def _process_keyboard_input(self, input_key: str, target_key: str, timestamp: datetime) -> bool:
        """キーボード入力を処理"""
        print(f"   入力検出: '{input_key}' (目標: '{target_key}')")
        
        # 正解判定
        is_correct = input_key.lower() == target_key.lower()
        if is_correct:
            self.correct_inputs += 1
            print(f"   ✅ 正解!")
        else:
            print(f"   ❌ 不正解 (目標: '{target_key}')")
        
        self.total_inputs += 1
        
        # 軌跡データを取得（前60フレーム）
        trajectory_data = list(self.data_collector.trajectory_buffer)
        
        # デバッグ情報を追加
        print(f"   🔍 軌跡データ収集状況:")
        print(f"      - バッファサイズ: {len(self.data_collector.trajectory_buffer)}")
        print(f"      - 軌跡データ長: {len(trajectory_data)}")
        if trajectory_data:
            print(f"      - 最初のフレーム: {trajectory_data[0].get('frame_index', 'N/A')}")
            print(f"      - 最後のフレーム: {trajectory_data[-1].get('frame_index', 'N/A')}")
        
        # サンプルデータを作成
        sample_data = {
            'timestamp': datetime.now().isoformat(),
            'data_version': '2.0',  # 追加
            'user_id': self.user_id,
            'session_id': os.path.basename(self.session_dir),
            'repetition': self.current_repetition + 1,
            'char_index': self.current_char_index,
            'target_char': target_key,
            'input_char': input_key,
            'is_correct': is_correct,
            'target_text': self.session_text,
            'trajectory_data': trajectory_data,
            'trajectory_length': len(trajectory_data),
            'coordinate_system': 'work_area_v2'  # 変更
        }
        
        # サンプルを保存
        self._save_sample(sample_data, target_key)
        
        # 次の文字に進む
        self.current_char_index += 1
        
        return True
    
    def _update_display(self, frame: np.ndarray, target_char: str) -> np.ndarray:
        """画面表示を更新"""
        h, w = frame.shape[:2]
        
        # 上部：目標テキストを表示
        target_text = f"目標テキスト: {self.session_text}"
        cv2.putText(frame, target_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 次に入力すべき文字をハイライト
        next_char_text = f"次に入力: '{target_char}'"
        cv2.putText(frame, next_char_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # 進捗状況を表示
        progress_text = f"進捗: {self.current_char_index + 1}/{len(self.session_text)} 文字"
        cv2.putText(frame, progress_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 繰り返し回数を表示
        repetition_text = f"繰り返し: {self.current_repetition + 1}/{self.repetitions} 回"
        cv2.putText(frame, repetition_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 正解率を表示
        if self.total_inputs > 0:
            accuracy = (self.correct_inputs / self.total_inputs * 100)
            accuracy_text = f"正解率: {accuracy:.1f}%"
            cv2.putText(frame, accuracy_text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 操作説明を表示
        instruction_text = "SPACE: 開始/停止 | R: リトライ | ESC: 終了"
        cv2.putText(frame, instruction_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def _save_sample(self, sample_data: dict, target_char: str):
        """サンプルデータを保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sample_{timestamp}_{target_char}_{self.current_char_index:02d}.json"
            filepath = os.path.join(self.session_dir, filename)
            
            # デバッグ情報を追加
            print(f"   🔍 保存前チェック:")
            print(f"      - 軌跡データ長: {len(sample_data.get('trajectory_data', []))}")
            print(f"      - ファイルパス: {filepath}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            
            # 保存後の確認
            file_size = os.path.getsize(filepath)
            print(f"   💾 サンプル保存: {filename} (サイズ: {file_size} バイト)")
            
            # ファイルの内容を簡単に確認
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.endswith('"x":'):
                    print(f"   ⚠️ 警告: ファイルが不完全です")
                    print(f"      最後の行: {content[-20:]}")
            
        except Exception as e:
            print(f"   ⚠️ サンプル保存エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_session_metadata(self):
        """セッション全体のメタデータを保存"""
        try:
            metadata = {
                'session_id': os.path.basename(self.session_dir),
                'user_id': self.user_id,
                'target_text': self.session_text,
                'repetitions': self.repetitions,
                'completed_repetitions': self.current_repetition,
                'total_inputs': self.total_inputs,
                'correct_inputs': self.correct_inputs,
                'accuracy': (self.correct_inputs / self.total_inputs * 100) if self.total_inputs > 0 else 0,
                'session_start': self.collection_start_time.isoformat() if self.collection_start_time else None,
                'session_end': datetime.now().isoformat(),
                'coordinate_system': 'work_area_v2'
            }
            
            metadata_file = os.path.join(self.session_dir, 'session_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ セッションメタデータを保存: {metadata_file}")
            
        except Exception as e:
            print(f"⚠️ メタデータ保存エラー: {e}")
    
    def _show_final_results(self):
        """最終結果を表示"""
        print("\n" + "="*50)
        print("🎯 データ収集完了！")
        print("="*50)
        print(f"ユーザーID: {self.user_id}")
        print(f"目標テキスト: {self.session_text}")
        print(f"完了回数: {self.current_repetition}/{self.repetitions}")
        print(f"総入力数: {self.total_inputs}")
        print(f"正解数: {self.correct_inputs}")
        
        if self.total_inputs > 0:
            accuracy = (self.correct_inputs / self.total_inputs * 100)
            print(f"最終正解率: {accuracy:.1f}%")
        
        print(f"セッションディレクトリ: {self.session_dir}")
        print("="*50)
    
    def _cleanup(self):
        """リソースのクリーンアップ"""
        try:
            # セッションメタデータを保存
            self._save_session_metadata()
            
            # コンポーネントのクリーンアップ
            if self.keyboard_tracker:
                self.keyboard_tracker.stop()
            
            if self.camera:
                self.camera.release()
            
            cv2.destroyAllWindows()
            
            print("🧹 リソースのクリーンアップ完了")
            
        except Exception as e:
            print(f"⚠️ クリーンアップエラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='学習用データ収集スクリプト')
    parser.add_argument('--user-id', default='user_001', help='ユーザーID (default: user_001)')
    parser.add_argument('--session-text', default='hello world', help='入力してもらうテキスト (default: "hello world")')
    parser.add_argument('--repetitions', type=int, default=10, help='繰り返し回数 (default: 10)')
    
    args = parser.parse_args()
    
    # データ収集クラスの作成
    collector = TrainingDataCollector(
        user_id=args.user_id,
        session_text=args.session_text,
        repetitions=args.repetitions
    )
    
    # コンポーネントの初期化
    if not collector.initialize_components():
        print("❌ 初期化に失敗しました")
        return 1
    
    # データ収集の実行
    if not collector.run_data_collection():
        print("❌ データ収集に失敗しました")
        return 1
    
    print("✅ データ収集が正常に完了しました")
    return 0


if __name__ == "__main__":
    exit(main())
