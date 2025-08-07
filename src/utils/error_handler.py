import cv2
import time
from enum import Enum
from typing import Optional, Dict, Any

class SystemStatus(Enum):
    """システム状態の定義"""
    INITIALIZING = "Initializing"
    RUNNING = "Running"
    ERROR = "Error"
    WARNING = "Warning"
    CALIBRATING = "Calibrating"
    PAUSED = "Paused"
    COMPLETED = "Completed"

class ErrorType(Enum):
    """エラータイプの定義"""
    CAMERA_ERROR = "Camera Error"
    KEYBOARD_MAP_ERROR = "Keyboard Map Error"
    HAND_DETECTION_ERROR = "Hand Detection Error"
    FILE_ERROR = "File Error"
    SYSTEM_ERROR = "System Error"

class ErrorHandler:
    """エラー処理と状態管理クラス"""
    
    def __init__(self):
        self.current_status = SystemStatus.INITIALIZING
        self.error_log = []
        self.warnings = []
        self.status_history = []
        self.last_status_change = time.time()
        
        # エラー表示設定
        self.error_display_duration = 300  # 10秒間表示
        self.warning_display_duration = 180  # 6秒間表示
        self.status_display_timer = 0
        self.current_message = None
        
    def set_status(self, status: SystemStatus, message: str = None):
        """システム状態を設定"""
        if self.current_status != status:
            self.status_history.append({
                'from': self.current_status,
                'to': status,
                'timestamp': time.time(),
                'message': message
            })
            self.current_status = status
            self.last_status_change = time.time()
            
            if message:
                print(f"[{status.value}] {message}")
    
    def add_error(self, error_type: ErrorType, message: str, show_duration: int = None):
        """エラーを記録・表示"""
        error_info = {
            'type': error_type,
            'message': message,
            'timestamp': time.time(),
            'status_at_error': self.current_status
        }
        self.error_log.append(error_info)
        
        # エラー表示設定
        self.current_message = f"ERROR: {message}"
        self.status_display_timer = show_duration or self.error_display_duration
        self.set_status(SystemStatus.ERROR, message)
        
        print(f"[ERROR] {error_type.value}: {message}")
    
    def add_warning(self, message: str, show_duration: int = None):
        """警告を記録・表示"""
        warning_info = {
            'message': message,
            'timestamp': time.time(),
            'status_at_warning': self.current_status
        }
        self.warnings.append(warning_info)
        
        # 警告表示設定
        self.current_message = f"WARNING: {message}"
        self.status_display_timer = show_duration or self.warning_display_duration
        
        if self.current_status != SystemStatus.ERROR:
            self.set_status(SystemStatus.WARNING, message)
        
        print(f"[WARNING] {message}")
    
    def check_camera_status(self, camera) -> bool:
        """カメラ状態をチェック"""
        if not hasattr(camera, 'cap') or camera.cap is None:
            self.add_error(ErrorType.CAMERA_ERROR, "Camera object not initialized")
            return False
        
        if not camera.cap.isOpened():
            self.add_error(ErrorType.CAMERA_ERROR, "Camera not connected or failed to open")
            return False
        
        # カメラからフレーム読み取りテスト
        ret, frame = camera.cap.read()
        if not ret or frame is None:
            self.add_error(ErrorType.CAMERA_ERROR, "Failed to read frame from camera")
            return False
        
        return True
    
    def check_keyboard_map_status(self, keyboard_map) -> bool:
        """キーボードマップ状態をチェック"""
        if not hasattr(keyboard_map, 'key_positions') or not keyboard_map.key_positions:
            self.add_error(ErrorType.KEYBOARD_MAP_ERROR, "Keyboard map not loaded or empty")
            return False
        
        # 最小限のキー数チェック（英字のみでも20キー以上あれば有効とみなす）
        if len(keyboard_map.key_positions) < 20:
            self.add_warning(f"Keyboard map has only {len(keyboard_map.key_positions)} keys")
            return True  # 警告だが動作継続
        
        return True
    
    def check_hand_detection_status(self, results, consecutive_failures: int) -> bool:
        """手検出状態をチェック"""
        if not results.multi_hand_landmarks:
            if consecutive_failures > 150:  # 5秒間検出失敗（30fps想定）
                self.add_warning("Hand not detected for 5 seconds")
            return False
        return True
    
    def update_display_timer(self):
        """表示タイマーを更新"""
        if self.status_display_timer > 0:
            self.status_display_timer -= 1
            if self.status_display_timer == 0:
                self.current_message = None
                # エラーや警告状態から回復
                if self.current_status in [SystemStatus.ERROR, SystemStatus.WARNING]:
                    self.set_status(SystemStatus.RUNNING)
    
    def get_display_info(self) -> Dict[str, Any]:
        """表示用の状態情報を取得"""
        return {
            'system_status': self.current_status.value,
            'status_message': self.current_message,
            'error_count': len(self.error_log),
            'warning_count': len(self.warnings),
            'status_color': self._get_status_color(),
            'show_message': self.status_display_timer > 0
        }
    
    def _get_status_color(self) -> str:
        """状態に応じた色を取得"""
        color_map = {
            SystemStatus.INITIALIZING: 'info',
            SystemStatus.RUNNING: 'success',
            SystemStatus.ERROR: 'error',
            SystemStatus.WARNING: 'warning',
            SystemStatus.CALIBRATING: 'info',
            SystemStatus.PAUSED: 'secondary',
            SystemStatus.COMPLETED: 'success'
        }
        return color_map.get(self.current_status, 'info')
    
    def get_system_health(self) -> Dict[str, Any]:
        """システムヘルス情報を取得"""
        recent_errors = [e for e in self.error_log if time.time() - e['timestamp'] < 300]  # 5分以内
        recent_warnings = [w for w in self.warnings if time.time() - w['timestamp'] < 300]
        
        return {
            'status': self.current_status.value,
            'uptime': time.time() - (self.status_history[0]['timestamp'] if self.status_history else time.time()),
            'recent_errors': len(recent_errors),
            'recent_warnings': len(recent_warnings),
            'total_errors': len(self.error_log),
            'total_warnings': len(self.warnings)
        }
    
    def save_error_log(self, filename: str = None):
        """エラーログをファイルに保存"""
        if not filename:
            filename = f"data/error_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        log_data = {
            'system_health': self.get_system_health(),
            'status_history': self.status_history,
            'errors': self.error_log,
            'warnings': self.warnings
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"Error log saved to: {filename}")
        except Exception as e:
            print(f"Failed to save error log: {e}")
    
    def reset_errors(self):
        """エラー・警告をクリア"""
        self.error_log.clear()
        self.warnings.clear()
        self.current_message = None
        self.status_display_timer = 0
        if self.current_status in [SystemStatus.ERROR, SystemStatus.WARNING]:
            self.set_status(SystemStatus.RUNNING, "Errors cleared")
    
    def __str__(self):
        """状態の文字列表現"""
        return f"ErrorHandler(Status: {self.current_status.value}, Errors: {len(self.error_log)}, Warnings: {len(self.warnings)})"