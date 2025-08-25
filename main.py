import argparse
import os
from src.modes.debug_mode import run_debug_mode
from src.modes.test_mode import run_test_mode
from src.modes.learning_mode import LearningMode
from src.input.keyboard_map import KeyboardMap

def parse_args():
    parser = argparse.ArgumentParser(description='キーボード入力支援システム')
    parser.add_argument('--mode', choices=['debug', 'test', 'learning'], default='debug',
                        help='実行モード (default: debug)')
    parser.add_argument('--text', default='hello world',
                        help='テストモード時のテスト文字列 (default: "hello world")')
    parser.add_argument('--user-id', default='user_001',
                        help='学習モード時のユーザーID (default: user_001)')
    parser.add_argument('--no-mapping', action='store_true',
                        help='マッピングをスキップして既存のキーボード座標を使用')
    parser.add_argument('--gemini-v2', action='store_true',
                   help='改善版Gemini検出を使用')
    return parser.parse_args()

def check_keyboard_mapping_exists():
    """キーボードマッピングファイルの存在確認"""
    return os.path.exists('keyboard_map.json')

def main():
    args = parse_args()
    
    # キーボードマッピングの存在確認
    mapping_exists = check_keyboard_mapping_exists()
    
    # --no-mappingが指定されているが、マッピングファイルが存在しない場合
    if args.no_mapping and not mapping_exists:
        print("エラー: --no-mappingが指定されましたが、keyboard_map.jsonが見つかりません。")
        print("以下のいずれかを実行してください：")
        print("1. --no-mappingオプションを外してマッピングを実行")
        print("2. 既存のkeyboard_map.jsonファイルを配置")
        return
    
    # --no-mappingが指定されていて、マッピングファイルが存在する場合
    if args.no_mapping and mapping_exists:
        print("既存のキーボードマッピングを使用します (--no-mapping指定)")
    
    # --no-mappingが指定されていない場合（通常動作）
    if not args.no_mapping and not mapping_exists:
        print("キーボードマッピングファイルが見つかりません。")
        print("マッピングモードを開始します...")
    
    # モード実行
    if args.mode == 'debug':
        run_debug_mode(skip_mapping=args.no_mapping)
    elif args.mode == 'test':
        run_test_mode(args.text, skip_mapping=args.no_mapping)
    elif args.mode == 'learning':
        print(f"🎓 学習モードを開始します")
        print(f"   ユーザーID: {args.user_id}")
        print(f"   目標テキスト: {args.text}")
        
        # 学習モードでもキーボードマッピングを実行（--no-mappingが指定されていない限り）
        if not args.no_mapping:
            print("キーボードマッピングを開始します")
            keyboard_map = KeyboardMap()
            if not keyboard_map.start_calibration():
                print("エラー: キーボードマッピングに失敗しました")
                return
        else:
            print("マッピングをスキップし、既存のキーボード座標を使用します")
        
        learning_mode = LearningMode(user_id=args.user_id)
        try:
            learning_mode.run_learning_mode(args.text)
        except KeyboardInterrupt:
            print("\n⚠️ ユーザーによって中断されました")
        except Exception as e:
            print(f"\n⚠️ エラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        finally:
            learning_mode._cleanup()

if __name__ == "__main__":
    main()