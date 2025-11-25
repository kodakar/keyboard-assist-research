# キーボード入力支援システム論文

## 概要

運動機能障害者向けキーボード入力支援システムに関する研究論文

**論文ファイル**: `keyboard_assist.tex`

## クイックスタート

### Windows環境でのコンパイル

#### 方法1: バッチファイルを使用（簡単）
```cmd
compile.bat
```

#### 方法2: PowerShellスクリプトを使用（推奨）
```powershell
.\compile.ps1
```

オプション：
- `.\compile.ps1 -KeepIntermediate` : 中間ファイルを残す
- `.\compile.ps1 -Clean` : 中間ファイルのみ削除

#### 方法3: 手動コンパイル
```powershell
platex keyboard_assist.tex
pbibtex keyboard_assist
platex keyboard_assist.tex
platex keyboard_assist.tex
dvipdfmx keyboard_assist.dvi
```

## 必要な環境

### TeX環境
- pLaTeX (platex)
- pBibTeX (pbibtex)
- dvipdfmx

### インストール方法
詳細は `paper/SETUP_GUIDE.md` を参照してください。

**推奨**: TeX Live for Windows
- URL: https://www.tug.org/texlive/

## 論文構成

### セクション構成
1. はじめに
2. 関連研究
   - 運動機能障害者向けの入力支援技術
   - カメラベースの入力支援技術
   - 震え補正・運動補正技術
   - 既存研究の課題と本研究の位置づけ
3. 提案システム
   - システム概要
   - 4点キャリブレーション
   - 手の軌跡データ取得
   - 動的特徴量の設計
   - 深層学習モデル
4. 実装
5. 評価実験
6. 考察
7. おわりに

### 使用パッケージ
- `graphicx` (dvipdfmx)
- `latexsym`
- `url`
- IPSJクラスファイル (ipsj.cls)

### 図表（作成予定）
- [ ] システム構成図
- [ ] キーボードマッピングの例
- [ ] 学習曲線
- [ ] リアルタイム予測画面

## トラブルシューティング

### エラー: "platex: command not found"
TeX環境がインストールされていません。`paper/SETUP_GUIDE.md`を参照してTeX Liveをインストールしてください。

### エラー: "! LaTeX Error: File `ipsj.cls' not found"
論文ディレクトリ（`paper/ipsj_v4-1/UTF8/`）で実行してください。

### PDF生成後に中間ファイルを削除したい
```powershell
.\compile.ps1
```
を実行すると、自動的に中間ファイルが削除されます。

手動削除：
```powershell
del *.aux, *.log, *.dvi, *.bbl, *.blg, *.toc, *.out
```

## 参考文献

参考文献は論文内の `\begin{thebibliography}` セクションで管理しています。

## 開発状況

- [x] 論文構成の作成
- [x] 各セクションの執筆
- [x] 参考文献の追加
- [ ] システム構成図の作成
- [ ] 実験結果の図表追加
- [ ] 最終校閲

## 連絡先

小高 大和 (Yamato Kodaka)  
東京電機大学 未来科学部 情報メディア学科


