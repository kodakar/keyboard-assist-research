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
├── main.py                   # メインのアプリケーションエントリポイント
├── config.py                 # 設定ファイル（パラメータ、定数など）
├── requirements.txt
├── src/
│   ├── camera.py             # カメラ制御
│   ├── data_collector.py     # データ収集
│   ├── hand_tracker.py       # 手の認識処理
│   ├── keyboard/
│   │   ├── __init__.py
│   │   ├── keyboard_mapper.py  # キーボードレイアウトとマッピング
│   │   ├── keyboard_tracker.py # キーボード入力検出（現在のkeyboard.py）
│   │   └── keyboard_detector.py # キーボード検出・認識
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── base_filter.py    # フィルター基底クラス
│   │   ├── moving_average.py # 移動平均フィルター
│   │   ├── kalman_filter.py  # カルマンフィルター
│   │   └── savitzky_golay.py # Savitzky-Golayフィルター
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py     # モデル基底クラス
│   │   ├── lstm_model.py     # LSTMモデル
│   │   ├── gru_model.py      # GRUモデル
│   │   └── attention.py      # Attention機構
│   ├── intent/
│   │   ├── __init__.py
│   │   ├── estimator.py      # 意図推測ベースクラス
│   │   ├── distance_based.py # 距離ベースの意図推測
│   │   ├── trajectory_based.py # 軌跡ベースの意図推測
│   │   └── nlp_correction.py # 言語モデルによる補正
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── visualization.py  # 可視化コンポーネント
│   │   └── feedback.py       # ユーザーフィードバック
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py         # ロギング機能
│   │   ├── metrics.py        # 評価指標
│   │   └── preprocessing.py  # データ前処理
│   └── experiments/
│       ├── __init__.py
│       ├── data_analysis.py  # データ分析
│       └── evaluation.py     # 性能評価
├── data/
│   ├── raw/                  # 生データ
│   ├── processed/            # 処理済みデータ
│   ├── models/               # 学習済みモデル
│   └── results/              # 評価結果
├── notebooks/                # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── result_analysis.ipynb
├── scripts/                  # スクリプト類
│   ├── setup.sh              # 環境設定
│   ├── collect_data.py       # データ収集専用スクリプト
│   └── train_model.py        # モデル学習専用スクリプト
├── tests/                    # テストコード
│   ├── test_filters.py
│   ├── test_models.py
│   └── test_intent.py
└── docs/                     # ドキュメント
    ├── architecture.md       # システム設計図
    ├── installation.md       # インストール手順
    ├── api.md                # API仕様書
    └── user_manual.md        # ユーザーマニュアル
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
