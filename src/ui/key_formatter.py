class KeyFormatter:
    """キー表示の統一管理クラス"""
    
    # 英字・数字・スペースのみ対応（研究用最小限）
    SUPPORTED_KEYS = {
        # 基本文字・数字
        'letters': 'abcdefghijklmnopqrstuvwxyz',
        'numbers': '0123456789',
        
        # スペースのみ
        'basic_special': {
            'Key.space': ' ',
        }
    }
    
    @staticmethod
    def is_supported_key(key):
        """サポートされているキーかチェック"""
        if isinstance(key, str):
            # 文字・数字の場合
            if len(key) == 1:
                return (key.lower() in KeyFormatter.SUPPORTED_KEYS['letters'] or
                        key in KeyFormatter.SUPPORTED_KEYS['numbers'])
            
            # スペースキーの場合
            if key == 'Key.space':
                return True
        
        return False
    
    @staticmethod
    def format_for_display(key):
        """一般表示用のキー名整形"""
        if not KeyFormatter.is_supported_key(key):
            return f"[{key}]"  # サポート外は括弧で表示
        
        # スペースキーの表示名変換
        if key == 'Key.space':
            return 'Space'
        
        # 文字・数字はそのまま
        return str(key)
    
    @staticmethod
    def format_for_test_display(key):
        """テスト表示用（見やすさ重視）"""
        if not KeyFormatter.is_supported_key(key):
            return None  # サポート外は表示しない
        
        # スペースは実際の空白文字に
        if key == 'Key.space':
            return ' '
        
        # 文字・数字はそのまま
        return str(key)
    
    @staticmethod
    def format_for_test_history(key):
        """テスト履歴表示用（連続表示）"""
        if not KeyFormatter.is_supported_key(key):
            return ""  # サポート外は履歴に含めない
        
        # スペースキーの履歴表示
        if key == 'Key.space':
            return ' '
        
        # 文字・数字はそのまま
        return str(key)
    
    @staticmethod
    def is_valid_for_test(key, test_text):
        """テストに使用可能なキーかチェック"""
        if not KeyFormatter.is_supported_key(key):
            return False
        
        # テスト文字列に含まれる文字のみ許可
        if isinstance(key, str) and len(key) == 1:
            return key.lower() in test_text.lower()
        
        # スペースはテスト文字列にスペースが含まれる場合のみ
        if key == 'Key.space':
            return ' ' in test_text
        
        return False
    
    @staticmethod
    def create_test_filter(test_text):
        """特定のテスト文字列に対するフィルター関数を作成"""
        allowed_chars = set(test_text.lower())
        
        def filter_func(key):
            return KeyFormatter.is_valid_for_test(key, test_text)
        
        return filter_func
    
    @staticmethod
    def get_supported_keys_info():
        """サポートされているキーの情報を取得（デバッグ用）"""
        return {
            'total_letters': len(KeyFormatter.SUPPORTED_KEYS['letters']),
            'total_numbers': len(KeyFormatter.SUPPORTED_KEYS['numbers']),
            'total_supported': len(KeyFormatter.SUPPORTED_KEYS['letters']) + len(KeyFormatter.SUPPORTED_KEYS['numbers']) + 1,
            'special_keys': ['Key.space']
        }
    
    # テスト用のサンプルテキスト
    SAMPLE_TEXTS = [
        "hello world",
        "test 123",
        "quick brown fox",
        "abc def ghi",
        "the lazy dog",
        "python code"
    ]