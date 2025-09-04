# src/keyboard.py
from pynput import keyboard

class KeyboardTracker:
    def __init__(self):
        self.current_key = None
        self.listener = None
        self.is_key_new = False  # 新しいキー入力かどうかのフラグ
        
    def on_press(self, key):
        try:
            self.current_key = key.char
            self.is_key_new = True  # キーが押されたときにフラグを立てる
        except AttributeError:
            # スペースキーの特別処理
            if key == keyboard.Key.space:
                self.current_key = ' '
            else:
                self.current_key = str(key)
            self.is_key_new = True

    def get_key_event(self):
        """新しいキー入力があれば、それを返して状態をリセット"""
        if self.is_key_new:
            key = self.current_key
            self.is_key_new = False  # フラグをリセット
            return key
        return None
        
    def start(self):
        """キーボードの監視を開始"""
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
    def stop(self):
        """キーボードの監視を停止"""
        if self.listener:
            self.listener.stop()