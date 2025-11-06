# キーボード入力支援システム

## 概要

カメラを用いて運動機能障害者向けのキーボード入力を支援するシステム。手の動きを認識し、ユーザーの入力意図を推測して、誤って押されたキーを自動的に修正する。

## 技術スタック

- Python 3.9+
- OpenCV
- MediaPipe
- PyTorch
- pynput
- scikit-learn
- TensorBoard

## 🚀 クイックスタート

### 1. 環境セットアップ

#### macOS/Linux の場合

```bash
# 1. リポジトリをクローン
git clone <repository-url>
cd keyboard-assist-research

# 2. Python環境を作成（推奨：Python 3.10）
conda create -n py310 python=3.10
conda activate py310

# 3. 必要なパッケージをインストール
pip install -r requirements.txt
```

#### Windows の場合

```powershell
# 1. リポジトリをクローン
git clone <repository-url>
cd keyboard-assist-research

# 2. Python環境を作成（推奨：Python 3.10）
conda create -n py310 python=3.10
conda activate py310

# 3. 必要なパッケージをインストール
pip install -r requirements.txt
```

### 2. 基本的な使い方

#### macOS/Linux の場合

```bash
# 1. データ収集（手の動きを学習用データとして収集、可変長対応）
python collect_training_data.py --user-id test_user --session-text "hello" --repetitions 3

# 2. 学習（収集したデータでモデルを学習、CNNモデル推奨）
python train_intent_model.py --data-dir data/training/test_user --model-type cnn --variable-length --epochs 50

# 3. 予測モード（学習済みモデルでリアルタイム予測）
# 最新モデルを自動選択
python src/modes/prediction_mode.py

# モデルを明示指定（例: ディレクトリ名 or .pth のフルパス）
python src/modes/prediction_mode.py --model intent_model_YYYYMMDD_HHMMSS
python src/modes/prediction_mode.py --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth --map keyboard_map.json
```

#### Windows の場合

```powershell
# 1. データ収集（手の動きを学習用データとして収集、可変長対応）
python collect_training_data.py --user-id test_user --session-text "hello" --repetitions 3

# 2. 学習（収集したデータでモデルを学習、CNNモデル推奨）
python train_intent_model.py --data-dir data/training/test_user --model-type cnn --variable-length --epochs 50

# 3. 予測モード（学習済みモデルでリアルタイム予測）
# 最新モデルを自動選択
python src/modes/prediction_mode.py

# モデルを明示指定（例: ディレクトリ名 or .pth のフルパス）
python src/modes/prediction_mode.py --model intent_model_YYYYMMDD_HHMMSS
python src/modes/prediction_mode.py --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth --map keyboard_map.json
```

### 🔀 特徴量次元の切り替え（18/24/30）

特徴量次元は `config/feature_config.py` で管理しています。切り替え方法は 2 通りあります。

1. 一時的に切り替え（環境変数で上書き、ファイル変更なし）

```powershell
# Windows PowerShell の例（実行中のみ有効）
$env:FEATURE_DIM=30; python src/modes/evaluation_mode.py --model models/AB_f30/best_model.pth --participant P001 --texts "hello"
$env:FEATURE_DIM=24; python src/modes/evaluation_mode.py --model models/AB_f24/best_model.pth --participant P001 --texts "hello"
Remove-Item Env:FEATURE_DIM  # 元に戻す
```

```bash
# macOS/Linux の例（実行中のみ有効）
FEATURE_DIM=30 python src/modes/evaluation_mode.py --model models/AB_f30/best_model.pth --participant P001 --texts "hello"
FEATURE_DIM=24 python src/modes/evaluation_mode.py --model models/AB_f24/best_model.pth --participant P001 --texts "hello"
```

2. 恒久的に切り替え（ファイルを直接変更）

```python
# config/feature_config.py
FEATURE_CONFIG = {
    # 'feature_dim': int(os.getenv('FEATURE_DIM', 18)) を 24 や 30 に書き換え
    'feature_dim': int(os.getenv('FEATURE_DIM', 30)),
}
```

注意:

- 使うモデルの入力次元と `feature_dim` を一致させてください（不一致だと入力サイズエラー）
- 例: 18 次元なら `models/AB/best_model.pth`、30 次元なら `models/AB_f30/best_model.pth`

## 📖 詳細な使い方

### データ収集（collect_training_data.py）

手の動きを学習用データとして収集します。

```bash
python collect_training_data.py [オプション]
```

**オプション：**

- `--user-id`: ユーザー ID（例：test_user, user_001）
- `--session-text`: 入力してもらうテキスト（例："hello", "abc123"）
- `--repetitions`: 繰り返し回数（例：3, 5, 10）

**例：**

#### macOS/Linux の場合

```bash
# 基本的なデータ収集
python collect_training_data.py --user-id test_user --session-text "hello" --repetitions 3

# 短いテキストでテスト
python collect_training_data.py --user-id test_user --session-text "a" --repetitions 1

# 長いテキストで本格収集
python collect_training_data.py --user-id user_001 --session-text "the quick brown fox" --repetitions 10
```

#### Windows の場合

```powershell
# 基本的なデータ収集
python collect_training_data.py --user-id test_user --session-text "hello" --repetitions 3

# 短いテキストでテスト
python collect_training_data.py --user-id test_user --session-text "a" --repetitions 1

# 長いテキストで本格収集
python collect_training_data.py --user-id user_001 --session-text "the quick brown fox" --repetitions 10
```

**操作説明：**

- **SPACE**: データ収集開始/停止
- **R**: 現在の文字をリトライ
- **ESC**: 終了

**画面表示：**

- 上部：目標テキスト（次に入力すべき文字をハイライト）
- 中央：カメラ映像＋手の検出結果
- 下部：進捗状況（正解率、完了回数など）

**可変長対応（重要）：**

データ収集は**可変長軌跡対応**で動作します：
- 各キー押下時にバッファをクリア（各キーの軌跡を独立させる）
- 軌跡長：最小5フレーム、最大90フレーム（ユーザーの入力速度に応じて自動）
- 速い入力（15フレーム）から遅い入力（90フレーム）まで対応

### 学習（train_intent_model.py）

収集したデータを使って機械学習モデルを学習します。複数のモデルタイプから選択可能です。

```bash
python train_intent_model.py [オプション]
```

**オプション：**

- `--data-dir`: データディレクトリ（例：data/training/test_user）
- `--epochs`: エポック数（例：10, 50, 100）
- `--batch-size`: バッチサイズ（例：16, 32, 64）
- `--learning-rate`: 学習率（例：0.001, 0.0001）
- `--model-type`: モデルタイプ（`cnn`, `gru`, `lstm`, `tcn`）
- `--variable-length`: 可変長対応を有効にする

**モデルタイプの選択：**

| モデル | 特徴 | パラメータ数 | 推奨用途 |
|--------|------|--------------|----------|
| **CNN** | 高速、可変長対応が簡単 | 167,653 | **推奨：速度重視** |
| **GRU** | LSTMより高速・省メモリ | 166,565 | LSTMの代替 |
| **LSTM** | 長期依存を学習 | 218,533 | 従来型 |
| **TCN** | 並列計算可能、最新手法 | 207,909 | 実験的 |

**例：**

#### macOS/Linux の場合

```bash
# 基本的な学習（CNN、可変長対応）
python train_intent_model.py --data-dir data/training/test_user --model-type cnn --variable-length --epochs 50

# GRUで学習
python train_intent_model.py --data-dir data/training/user_001 --model-type gru --variable-length --epochs 100

# LSTMで学習（従来型）
python train_intent_model.py --data-dir data/training/user_001 --model-type lstm --variable-length --epochs 100

# TCNで学習（実験的）
python train_intent_model.py --data-dir data/training/user_001 --model-type tcn --variable-length --epochs 100

# モデル比較実験（同じデータで4つのモデルを比較）
python train_intent_model.py --data-dir data/training/user_001 --model-type cnn --variable-length --epochs 50
python train_intent_model.py --data-dir data/training/user_001 --model-type gru --variable-length --epochs 50
python train_intent_model.py --data-dir data/training/user_001 --model-type lstm --variable-length --epochs 50
python train_intent_model.py --data-dir data/training/user_001 --model-type tcn --variable-length --epochs 50
```

#### Windows の場合

```powershell
# 基本的な学習（CNN、可変長対応）
python train_intent_model.py --data-dir data/training/test_user --model-type cnn --variable-length --epochs 50

# GRUで学習
python train_intent_model.py --data-dir data/training/user_001 --model-type gru --variable-length --epochs 100

# LSTMで学習（従来型）
python train_intent_model.py --data-dir data/training/user_001 --model-type lstm --variable-length --epochs 100

# TCNで学習（実験的）
python train_intent_model.py --data-dir data/training/user_001 --model-type tcn --variable-length --epochs 100

# モデル比較実験（同じデータで4つのモデルを比較）
python train_intent_model.py --data-dir data/training/user_001 --model-type cnn --variable-length --epochs 50
python train_intent_model.py --data-dir data/training/user_001 --model-type gru --variable-length --epochs 50
python train_intent_model.py --data-dir data/training/user_001 --model-type lstm --variable-length --epochs 50
python train_intent_model.py --data-dir data/training/user_001 --model-type tcn --variable-length --epochs 50
```

**学習結果：**

- モデルファイル：`models/intent_model_YYYYMMDD_HHMMSS/`
- 学習曲線：`learning_curves.png`
- 混同行列：`confusion_matrix.png`
- 結果 JSON：`training_results.json`（検証・テスト結果を含む）
- TensorBoard ログ：`runs/intent_training_YYYYMMDD_HHMMSS/`

### 予測モード（prediction_mode.py）

学習済みモデルを使ってリアルタイムでキー入力を予測します。

```bash
# 最新モデルを自動選択
python src/modes/prediction_mode.py

# モデルを明示指定（ディレクトリ名指定 or .pth フルパス指定）
python src/modes/prediction_mode.py --model intent_model_YYYYMMDD_HHMMSS
python src/modes/prediction_mode.py --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth --map keyboard_map.json
```

**主な機能/UI：**

- リアルタイム手追跡
- 常時の Top-3 予測を左上に表示（推論準備中は「Loading predictions...」）
- 画面下部に Actual / Predict（二段）
  - Actual: ユーザーが押したキー（a-z, 0-9, Space）
  - Predict: 押した瞬間の Top-3 予測をスナップショット表示
- 学習済みモデルの読み込み（--model で選択可能）
- デバッグ表示・評価モードは削除し、UI を簡素化

### 評価モード（evaluation_mode.py）

被験者によるリアルタイム評価実験を実施し、システムの実用性能を測定します。

```bash
# 基本的な評価実験
python src/modes/evaluation_mode.py --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth --participant P001 --texts "hello world" "the quick brown fox"

# 複数のタスクで評価
python src/modes/evaluation_mode.py --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth --participant P002 --texts "apple banana cherry" "keyboard assist research" "python machine learning"
```

**主な機能：**

- **構造化された評価**: 指定されたテキストを 1 文字ずつ入力
- **詳細なログ記録**: 目標文字、予測 Top-3、実際の入力、正解/不正解、入力時間を記録
- **自動指標計算**: Top-1 精度、Top-3 精度、平均入力時間、WPM、エラー率を自動計算
- **結果保存**: JSON 形式で詳細な評価結果を保存
- **進捗表示**: リアルタイムで進捗状況と予測結果を表示

**評価指標：**

- **Top-1 精度**: 1 位予測の正解率
- **Top-3 精度**: Top-3 予測に正解が含まれる割合
- **平均入力時間**: 1 文字あたりの平均入力時間（秒）
- **WPM**: Words Per Minute（英語では 5 文字=1 単語として計算）
- **エラー率**: 不正解の割合

**保存されるファイル：**

```
evaluation_results/
├── P001/
│   └── evaluation_20241002_143022.json
└── P002/
    └── evaluation_20241002_150315.json
```

**評価結果の例：**

```json
{
  "participant_id": "P001",
  "timestamp": "2024-10-02T14:30:22.123456",
  "model_path": "models/intent_model_20241002/best_model.pth",
  "evaluation_log": [
    {
      "task_idx": 0,
      "target_text": "hello world",
      "inputs": [
        {
          "target_char": "h",
          "predicted_top3": ["h", "g", "y"],
          "predicted_probs": [85.3, 8.2, 3.1],
          "actual_input": "h",
          "is_correct": true,
          "input_time": 2.34,
          "timestamp": "2024-10-02T14:30:24.456789"
        }
      ]
    }
  ],
  "metrics": {
    "top1_accuracy": 78.5,
    "top3_accuracy": 91.2,
    "avg_input_time": 2.45,
    "wpm": 4.9,
    "error_rate": 21.5,
    "total_inputs": 120,
    "correct_inputs": 94
  }
}
```

### オフライン評価（evaluate_offline.py）

学習済みモデルをテストセット（学習時未使用）で評価し、真の性能を測定します。

```bash
# 基本的なオフライン評価
python evaluate_offline.py --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth

# カスタム設定
python evaluate_offline.py \
  --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth \
  --data-dir data/training \
  --output-dir evaluation_results/test_20241002
```

**主な機能：**

- **テストセット専用評価**: 学習時に使用されなかったテストデータで評価
- **詳細な精度分析**: Top-1 精度、Top-3 精度、クラス別精度を計算
- **可視化**: 混同行列とクラス別精度のグラフを自動生成
- **結果保存**: JSON 形式で詳細結果、PNG 形式で可視化を保存

**評価指標：**

- **Top-1 精度**: 1 位予測の正解率
- **Top-3 精度**: Top-3 予測に正解が含まれる割合
- **クラス別精度**: 各キー（a-z, 0-9, スペース）の個別精度
- **混同行列**: 予測結果の詳細な分析

**保存されるファイル：**

```
evaluation_results/offline/
├── offline_evaluation_20241002_143022.json  # 詳細結果
├── confusion_matrix.png                     # 混同行列
└── per_class_accuracy.png                   # クラス別精度
```

**評価結果の例：**

```
📊 オフライン評価結果（テストセット）
========================================
テストサンプル数: 245
Top-1精度:        78.5%
Top-3精度:        91.2%
========================================

🏆 クラス別精度（上位5位）:
   1. a: 95.2%
   2. e: 92.8%
   3. i: 89.1%
   4. o: 87.3%
   5. u: 85.6%

📉 クラス別精度（下位5位）:
   41. z: 45.2%
   42. x: 42.1%
   43. q: 38.9%
   44. j: 35.6%
   45. 9: 32.1%
========================================
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. カメラが開かない

##### macOS/Linux の場合

```bash
# カメラインデックスを確認
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# カメラ権限を確認
# macOS: システム環境設定 > セキュリティとプライバシー > カメラ
# Linux: カメラデバイスの権限を確認
```

##### Windows の場合

```powershell
# カメラインデックスを確認
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# カメラ権限を確認
# Windows: 設定 > プライバシーとセキュリティ > カメラ
```

#### 2. MediaPipe のエラー

```bash
# MediaPipeを再インストール
pip uninstall mediapipe
pip install mediapipe

# バージョンを指定してインストール
pip install mediapipe==0.10.0
```

#### 3. PyTorch のエラー

```bash
# PyTorchを再インストール
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio

# CPU版のみインストール（GPUがない場合）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 4. データ収集時の JSON 保存エラー

##### macOS/Linux の場合

```bash
# 破損ファイルを削除
rm -rf data/training/*/session_*/sample_*.json

# 再実行
python collect_training_data.py --user-id test_user --session-text "a" --repetitions 1
```

##### Windows の場合

```powershell
# 破損ファイルを削除
Remove-Item data/training/*/session_*/sample_*.json

# 再実行
python collect_training_data.py --user-id test_user --session-text "a" --repetitions 1
```

#### 5. 学習時のメモリ不足

```bash
# バッチサイズを小さくする
python train_intent_model.py --data-dir data/training/test_user --epochs 10 --batch-size 16

# データ拡張を無効にする
# src/processing/data_loader.py で augment=False に設定
```

### ログの確認方法

#### 学習ログ

##### macOS/Linux の場合

```bash
# TensorBoardで学習曲線を確認
tensorboard --logdir runs/

# ブラウザで http://localhost:6006 を開く
```

##### Windows の場合

```powershell
# TensorBoardで学習曲線を確認
tensorboard --logdir runs/

# ブラウザで http://localhost:6006 を開く
```

#### データ収集ログ

##### macOS/Linux の場合

```bash
# 収集されたデータを確認
find data/training/ -name "*.json" -type f

# 特定のセッションを確認
head -10 data/training/test_user/session_*/sample_*.json
```

##### Windows の場合

```powershell
# 収集されたデータを確認
Get-ChildItem data/training/ -Recurse | Where-Object {$_.Name -like "*.json"}

# 特定のセッションを確認
Get-Content data/training/test_user/session_*/sample_*.json | Select-Object -First 10
```

## 📁 ファイル構成

```
keyboard-assist-research/
├── main.py                      # エントリーポイント
├── collect_training_data.py     # データ収集スクリプト（可変長対応）
├── train_intent_model.py        # 学習スクリプト（モデル選択可能）
├── requirements.txt             # 依存パッケージ
├── config/
│   └── feature_config.py       # 特徴量設定
├── src/
│   ├── core/                   # コア機能
│   ├── input/                  # 入力処理
│   ├── processing/             # データ処理・学習
│   │   ├── data_loader.py      # データローダー（可変長対応）
│   │   ├── feature_extractor.py # 特徴量抽出
│   │   └── models/
│   │       ├── hand_lstm.py    # LSTM（従来型）
│   │       └── hand_models.py  # CNN/GRU/LSTM/TCN（新規）
│   ├── modes/                  # 動作モード
│   │   ├── prediction_mode.py  # 予測モード
│   │   └── evaluation_mode.py  # 評価モード
│   └── ui/                     # ユーザーインターフェース
├── data/                       # データ保存
├── models/                     # 学習済みモデル
└── runs/                       # TensorBoardログ
```

## 📊 データ構造

### データ分割方法

このシステムでは**3 分割**を採用して機械学習のベストプラクティスに従っています：

- **訓練データ（60%）**: モデルの学習に使用
- **検証データ（20%）**: ハイパーパラメータ調整と Early Stopping に使用
- **テストデータ（20%）**: 最終的な汎化性能評価に使用（一度も触れない）

#### 分割の特徴

- **ユーザー別分割**: 各ユーザーのデータを個別に分割（データリーク防止）
- **層化分割**: 各キーの分布を訓練・検証・テストで均等に保持
- **再現性**: 固定シード（random_state=42）で同じ分割結果を保証

### 収集されるデータ

- **軌跡データ**: 可変長（5～90フレーム）の手の動き
  - ユーザーの入力速度に応じて自動調整
  - 速い入力: 15-30フレーム
  - 通常入力: 30-60フレーム
  - 遅い入力: 60-90フレーム
- **座標系**: 作業領域座標（0-1 の正規化）
- **特徴量**: 18 次元（座標、相対位置、速度、加速度、方向など）
- **ラベル**: 37 クラス（a-z, 0-9, スペース）
- **キー独立**: 各キー押下後にバッファをクリア（キーごとに独立した軌跡）

### データファイル形式

```json
{
  "timestamp": "2025-08-30T02:02:02.631748",
  "data_version": "2.0",
  "user_id": "test_user",
  "target_char": "a",
  "input_char": "a",
  "trajectory_data": [...],
  "trajectory_length": 45,
  "coordinate_system": "work_area_v2"
}
```

## 🎯 学習のコツ

### 1. データ収集のポイント

- **多様性**: 異なる手の位置、速度、角度で収集
- **一貫性**: 同じキーを複数回収集
- **品質**: 手がはっきり見える環境で収集

### 2. 学習パラメータの調整

- **エポック数**: データ量に応じて調整（少ない場合は 10-50、多い場合は 100-200）
- **バッチサイズ**: メモリに余裕があれば 32-64、なければ 16
- **学習率**: 初期値 0.001、収束しない場合は 0.0001 に下げる

### 3. モデル選択のポイント

- **CNN（推奨）**: 最速、パラメータ最小、可変長対応が簡単
  - 速度重視、実用性重視の場合に最適
  - 学習が速い（並列計算可能）
- **GRU**: LSTMより高速で省メモリ
  - LSTMの代替として優秀
  - 精度とスピードのバランスが良い
- **LSTM**: 従来型、長期依存を学習
  - 系列全体の文脈が重要な場合
  - 学習は遅いが実績がある
- **TCN**: 最新手法、実験的
  - 並列計算可能、長期依存を捉える
  - 研究・実験目的に適している

**推奨：** まずCNNで試して、精度が足りなければGRUやLSTMを検討

### 4. 可変長対応の活用

- **`--variable-length`フラグ**: 必ず有効にする
- **速い入力に対応**: 15フレーム（0.5秒）から予測可能
- **実用性向上**: ユーザーの入力速度に合わせて動作
- **精度の注意**: 短い軌跡は精度が下がる可能性（トレードオフ）

### 5. 過学習の防止

- **3 分割データ**: 訓練データ 60%、検証データ 20%、テストデータ 20%に分割
- **Early Stopping**: 検証損失が 5 エポック改善しない場合に停止
- **データ拡張**: ガウシアンノイズ、時間軸の伸縮
- **真の汎化性能**: テストデータで最終評価（ハイパーパラメータ調整に使用しない）

## 🔍 デバッグ方法

### 1. データ収集のデバッグ

#### macOS/Linux の場合

```bash
# 詳細ログを有効にする
python collect_training_data.py --user-id test_user --session-text "a" --repetitions 1

# 保存されたファイルを確認
head -20 data/training/test_user/session_*/sample_*.json
```

#### Windows の場合

```powershell
# 詳細ログを有効にする
python collect_training_data.py --user-id test_user --session-text "a" --repetitions 1

# 保存されたファイルを確認
Get-Content data/training/test_user/session_*/sample_*.json | Select-Object -First 20
```

### 2. 学習のデバッグ

#### macOS/Linux の場合

```bash
# 少ないエポックでテスト
python train_intent_model.py --data-dir data/training/test_user --epochs 1

# 学習結果を確認
find models/ -type f
```

#### Windows の場合

```powershell
# 少ないエポックでテスト
python train_intent_model.py --data-dir data/training/test_user --epochs 1

# 学習結果を確認
Get-ChildItem models/ -Recurse
```

### 3. 予測のデバッグ

#### macOS/Linux の場合

```bash
# 予測モードでテスト
python src/modes/prediction_mode.py

# エラーログを確認
# コンソール出力でエラーメッセージを確認
```

#### Windows の場合

```powershell
# 予測モードでテスト
python src/modes/prediction_mode.py

# エラーログを確認
# コンソール出力でエラーメッセージを確認
```

## 📞 サポート

### 問題が発生した場合

1. **エラーメッセージを確認**: コンソール出力の最後の行を確認
2. **ログファイルを確認**: 保存されたファイルの内容を確認
3. **環境を確認**: Python バージョン、パッケージバージョンを確認
4. **再実行**: 同じコマンドで再実行してみる

### よくある質問

**Q: カメラが開かない**
A: カメラの権限設定を確認し、他のアプリでカメラが使用されていないか確認してください。

**Q: 学習が遅い**
A: バッチサイズを小さくし、データ拡張を無効にしてください。また、CNNモデル（`--model-type cnn`）を使うと学習が高速化されます。

**Q: 精度が低い**
A: より多くのデータを収集し、学習エポック数を増やしてください。また、GRUやLSTMモデルを試してみてください。

**Q: どのモデルを使えばいい？**
A: 最初はCNNモデル（`--model-type cnn --variable-length`）を推奨します。速度と精度のバランスが良く、可変長対応も簡単です。

**Q: `--variable-length`は必須？**
A: 新しいデータ収集方式では必須です。可変長データ（5-90フレーム）を正しく扱えるようになります。

**Q: 複数のモデルを比較したい**
A: 同じデータで`--model-type`を変えて複数回学習を実行してください。精度を比較できます。

## 📝 更新履歴

- **2025-11-06**: 可変長対応の実装、4つのモデル（CNN, GRU, LSTM, TCN）追加
  - データ収集：可変長軌跡対応（5-90フレーム）
  - モデル選択：`--model-type`オプション追加
  - 可変長学習：`--variable-length`フラグ追加
  - モデル比較実験が可能に
- **2025-08-30**: 学習システムの完成、README の大幅更新
- **2025-08-22**: 基本的なシステムの実装
- **2025-08-15**: プロジェクト開始

## ライセンス

研究用・学術用途に限る

## 注意事項

- カメラへのアクセス権限が必要
- キーボード制御の権限が必要
- 十分なディスク容量が必要（データ収集時）
- GPU があると学習が高速化されます

## 📁 Git 管理について

### 無視されるファイル

以下のファイルは`.gitignore`で無視され、Git にコミットされません：

- **学習済みモデル**: `models/`, `*.pth`, `*.pt`
- **学習ログ**: `runs/`, TensorBoard ログ
- **学習結果**: `*.png`, `training_results.json`
- **個人データ**: `data/training/*/session_*/sample_*.json`

### 理由

- ファイルサイズが大きい（モデルファイルは数百 MB〜数 GB）
- 個人固有のデータ（再現可能）
- リポジトリの軽量化

### 共有したい場合

- **コード**: すべての Python ファイルと設定ファイル
- **データ構造例**: `data/training/README.md`
- **学習済みモデル**: 別途配布（Google Drive、Hugging Face 等）
