# src/utils/logger.py
"""
ログ管理機能
"""

import logging
import os
from datetime import datetime
from typing import Optional

class LearningLogger:
    def __init__(self, name: str = "learning_system", log_dir: str = "logs"):
        """
        学習システム用ロガーの初期化
        
        Args:
            name: ロガー名
            log_dir: ログ保存ディレクトリ
        """
        self.name = name
        self.log_dir = log_dir
        
        # ログディレクトリの作成
        os.makedirs(log_dir, exist_ok=True)
        
        # ロガーの設定
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 既存のハンドラーをクリア
        self.logger.handlers.clear()
        
        # ファイルハンドラーの設定
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラーの設定
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマッターの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラーの追加
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # ログファイルパスを保存
        self.log_file = log_file
        
        self.logger.info(f"ロガー '{name}' を初期化しました")
        self.logger.info(f"ログファイル: {log_file}")
    
    def info(self, message: str):
        """情報ログを出力"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告ログを出力"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """エラーログを出力"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """デバッグログを出力"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """重大エラーログを出力"""
        self.logger.critical(message)
    
    def log_training_start(self, user_id: str, target_text: str, samples_count: int):
        """学習開始をログに記録"""
        self.info(f"学習開始 - ユーザー: {user_id}, 目標テキスト: {target_text}, サンプル数: {samples_count}")
    
    def log_training_complete(self, user_id: str, epochs: int, final_accuracy: float):
        """学習完了をログに記録"""
        self.info(f"学習完了 - ユーザー: {user_id}, エポック数: {epochs}, 最終精度: {final_accuracy:.2f}%")
    
    def log_data_collection(self, user_id: str, samples_count: int, trajectories_count: int):
        """データ収集状況をログに記録"""
        self.info(f"データ収集 - ユーザー: {user_id}, サンプル: {samples_count}, 軌跡: {trajectories_count}")
    
    def log_error(self, error_type: str, error_message: str, user_id: Optional[str] = None):
        """エラーをログに記録"""
        user_info = f" (ユーザー: {user_id})" if user_id else ""
        self.error(f"エラー発生 - タイプ: {error_type}, メッセージ: {error_message}{user_info}")
    
    def get_log_file_path(self) -> str:
        """ログファイルパスを取得"""
        return self.log_file
    
    def cleanup_old_logs(self, max_files: int = 10):
        """古いログファイルを削除"""
        try:
            log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.log')]
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)), reverse=True)
            
            if len(log_files) > max_files:
                for old_log in log_files[max_files:]:
                    old_log_path = os.path.join(self.log_dir, old_log)
                    os.remove(old_log_path)
                    self.info(f"古いログファイルを削除: {old_log}")
        except Exception as e:
            self.error(f"ログクリーンアップエラー: {e}")


# デフォルトロガーの作成
default_logger = LearningLogger()

def get_logger(name: str = "learning_system") -> LearningLogger:
    """ロガーを取得（シングルトンパターン）"""
    return LearningLogger(name)
