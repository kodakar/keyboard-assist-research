import argparse
from src.modes.debug_mode import run_debug_mode
from src.modes.test_mode import run_test_mode

def parse_args():
    parser = argparse.ArgumentParser(description='キーボード入力支援システム')
    parser.add_argument('--mode', choices=['debug', 'test'], default='debug',
                        help='実行モード (default: debug)')
    parser.add_argument('--text', default='hello world',
                        help='テストモード時のテスト文字列 (default: "hello world")')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == 'debug':
        run_debug_mode()
    elif args.mode == 'test':
        run_test_mode(args.text)

if __name__ == "__main__":
    main()