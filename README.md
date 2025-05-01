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

## フォルダ構成

```
keyboard-assist-research/
├── src/                   # ソースコードディレクトリ
│   ├── camera.py          # カメラ制御
│   ├── hand_track.py      # 手の認識処理
│   ├── keyboard.py        # キーボード制御
│   └── logger.py          # ログ処理
├── data/                  # データ保存用ディレクトリ
│   ├── models/            # 学習済みモデル
│   └── logs/              # ログファイル
├── tests/                 # テストコード
├── requirements.txt       # 依存パッケージリスト
├── main.py                # メインプログラム
└── README.md              # 本ファイル
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
