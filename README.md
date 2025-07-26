# キーボード入力支援システム

## 概要

カメラを用いて運動機能障害者向けのキーボード入力を支援するシステム。手の動きを認識し、ユーザーの入力意図を推測して、誤って押されたキーを自動的に修正する。

## 技術スタック

- Python 3.9+
- OpenCV
- MediaPipe
- PyTorch
- pynput

## 開発環境セットアップ

1. 仮想環境の作成

```bash
python3 -m venv venv # MacOS/Linux
py -3.10 -m venv venv_py310 # Windows

conda env create -f environment.yml
conda activate <env_name>
```

2. 仮想環境の有効化

```bash
source venv/bin/activate  # MacOS/Linux
.\venv\Scripts\Activate.ps1 # Windows

```

3. 必要なパッケージのインストール

```bash
pip install -r requirements.txt
```

4. 仮想環境の無効化

```bash
deactivate
conda deactivate
```

## フォルダ構成(最終案)

```
keyboard-assist-research/
├── .gitignore
├── README.md
├── main.py                      # エントリーポイント（引数解析のみ）
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py              # 設定管理（実験条件、パラメータ等）
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── system.py            # メインシステム（全体統合）
│   │   ├── camera.py            # カメラ制御
│   │   ├── hand_tracker.py      # 手検出
│   │   └── intent_estimator.py  # 意図推定（距離計算等）
│   ├── input/
│   │   ├── __init__.py
│   │   ├── keyboard_tracker.py  # キー入力監視
│   │   └── keyboard_map.py      # キーボードマッピング
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── filters/
│   │   │   ├── __init__.py
│   │   │   ├── base_filter.py   # フィルター基底クラス
│   │   │   ├── moving_average.py # 移動平均フィルター
│   │   │   ├── kalman_filter.py # カルマンフィルター
│   │   │   └── savitzky_golay.py # Savitzky-Golayフィルター
│   │   └── models/              # 機械学習モデル
│   │       ├── __init__.py
│   │       ├── base_model.py    # モデル基底クラス
│   │       ├── lstm_model.py    # LSTMモデル
│   │       ├── gru_model.py     # GRUモデル
│   │       └── attention.py     # Attention機構
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py         # データ収集
│   │   └── exporter.py          # データ出力
│   ├── modes/
│   │   ├── __init__.py
│   │   ├── debug_mode.py        # デバッグモード
│   │   └── test_mode.py         # テストモード
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── display_manager.py   # 表示統合管理
│   │   ├── debug_display.py     # デバッグ用表示
│   │   └── test_display.py      # テスト用表示
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # ログ管理
│       └── metrics.py           # 評価指標計算
├── data/
│   ├── raw/                     # 生データ
│   ├── processed/               # 処理済みデータ
│   ├
```

## 主な機能

1. リアルタイムの手の動き認識
2. キーボード入力の意図推定
3. 入力の自動修正

## モデルについて

- 手の動き認識：MediaPipe Hands
- 意図推定：LSTM/GRU ベースのカスタムモデル
- 文脈考慮：自然言語処理による補助

## 研究目的

運動機能障害を持つユーザーのデジタルアクセシビリティを向上させ、より正確なキーボード入力を支援する。

## 開発環境

- OS: macOS
- Python: 3.9+
- IDE: VSCode 推奨

## ライセンス

研究用・学術用途に限る

## 注意事項

- カメラへのアクセス権限が必要
- キーボード制御の権限が必要
