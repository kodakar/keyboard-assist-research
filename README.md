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
# 1. データ収集（手の動きを学習用データとして収集）
python collect_training_data.py --user-id test_user --session-text "hello" --repetitions 3

# 2. 学習（収集したデータでモデルを学習）
python train_intent_model.py --data-dir data/training/test_user --epochs 10

# 3. 予測モード（学習済みモデルでリアルタイム予測）
# 最新モデルを自動選択
python src/modes/prediction_mode.py

# モデルを明示指定（例: ディレクトリ名 or .pth のフルパス）
python src/modes/prediction_mode.py --model intent_model_YYYYMMDD_HHMMSS
python src/modes/prediction_mode.py --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth --map keyboard_map.json
```

#### Windows の場合

```powershell
# 1. データ収集（手の動きを学習用データとして収集）
python collect_training_data.py --user-id test_user --session-text "hello" --repetitions 3

# 2. 学習（収集したデータでモデルを学習）
python train_intent_model.py --data-dir data/training/test_user --epochs 10

# 3. 予測モード（学習済みモデルでリアルタイム予測）
# 最新モデルを自動選択
python src/modes/prediction_mode.py

# モデルを明示指定（例: ディレクトリ名 or .pth のフルパス）
python src/modes/prediction_mode.py --model intent_model_YYYYMMDD_HHMMSS
python src/modes/prediction_mode.py --model models/intent_model_YYYYMMDD_HHMMSS/best_model.pth --map keyboard_map.json
```

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

### 学習（train_intent_model.py）

収集したデータを使って LSTM モデルを学習します。

```bash
python train_intent_model.py [オプション]
```

**オプション：**

- `--data-dir`: データディレクトリ（例：data/training/test_user）
- `--epochs`: エポック数（例：10, 50, 100）
- `--batch-size`: バッチサイズ（例：16, 32, 64）
- `--learning-rate`: 学習率（例：0.001, 0.0001）

**例：**

#### macOS/Linux の場合

```bash
# 基本的な学習
python train_intent_model.py --data-dir data/training/test_user --epochs 10

# 本格的な学習
python train_intent_model.py --data-dir data/training/user_001 --epochs 100 --batch-size 64

# 高精度学習
python train_intent_model.py --data-dir data/training/user_001 --epochs 200 --learning-rate 0.0001
```

#### Windows の場合

```powershell
# 基本的な学習
python train_intent_model.py --data-dir data/training/test_user --epochs 10

# 本格的な学習
python train_intent_model.py --data-dir data/training/user_001 --epochs 100 --batch-size 64

# 高精度学習
python train_intent_model.py --data-dir data/training/user_001 --epochs 200 --learning-rate 0.0001
```

**学習結果：**

- モデルファイル：`models/intent_model_YYYYMMDD_HHMMSS/`
- 学習曲線：`learning_curves.png`
- 混同行列：`confusion_matrix.png`
- 結果 JSON：`training_results.json`
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
├── collect_training_data.py     # データ収集スクリプト
├── train_intent_model.py        # 学習スクリプト
├── requirements.txt             # 依存パッケージ
├── src/
│   ├── core/                   # コア機能
│   ├── input/                  # 入力処理
│   ├── processing/             # データ処理・学習
│   ├── modes/                  # 動作モード
│   └── ui/                     # ユーザーインターフェース
├── data/                       # データ保存
├── models/                     # 学習済みモデル
└── runs/                       # TensorBoardログ
```

## 📊 データ構造

### 収集されるデータ

- **軌跡データ**: 60 フレーム分の手の動き
- **座標系**: 作業領域座標（0-1 の正規化）
- **特徴量**: 15 次元（座標、相対位置、速度、加速度）
- **ラベル**: 37 クラス（a-z, 0-9, スペース）

### データファイル形式

```json
{
  "timestamp": "2025-08-30T02:02:02.631748",
  "user_id": "test_user",
  "target_char": "a",
  "input_char": "a",
  "trajectory_data": [...],
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

### 3. 過学習の防止

- **検証データ**: 訓練データの 20%を検証用に分割
- **Early Stopping**: 検証損失が 5 エポック改善しない場合に停止
- **データ拡張**: ガウシアンノイズ、時間軸の伸縮

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
A: バッチサイズを小さくし、データ拡張を無効にしてください。

**Q: 精度が低い**
A: より多くのデータを収集し、学習エポック数を増やしてください。

## 📝 更新履歴

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
