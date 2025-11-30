# キーボード入力支援システム論文

## 概要

運動機能障害者向けキーボード入力支援システムに関する研究論文

**論文ファイル**: `keyboard_assist.tex`

## クイックスタート

### Windows 環境でのコンパイル

#### 方法 1: バッチファイルを使用（簡単）

```cmd
compile.bat
```

#### 方法 2: PowerShell スクリプトを使用（推奨）

```powershell
.\compile.ps1
```

オプション：

- `.\compile.ps1 -KeepIntermediate` : 中間ファイルを残す
- `.\compile.ps1 -Clean` : 中間ファイルのみ削除

#### 方法 3: 手動コンパイル

```powershell
platex keyboard_assist.tex
pbibtex keyboard_assist
platex keyboard_assist.tex
platex keyboard_assist.tex
dvipdfmx keyboard_assist.dvi
```

## 必要な環境

### TeX 環境

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
   - 4 点キャリブレーション
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
- IPSJ クラスファイル (ipsj.cls)

### 図表（作成予定）

- [ ] システム構成図
- [ ] キーボードマッピングの例
- [ ] 学習曲線
- [ ] リアルタイム予測画面

## トラブルシューティング

### エラー: "platex: command not found"

TeX 環境がインストールされていません。`paper/SETUP_GUIDE.md`を参照して TeX Live をインストールしてください。

### エラー: "! LaTeX Error: File `ipsj.cls' not found"

論文ディレクトリ（`paper/ipsj_v4-1/UTF8/`）で実行してください。

### PDF 生成後に中間ファイルを削除したい

```powershell
.\compile.ps1
```

を実行すると、自動的に中間ファイルが削除されます。

手動削除：

```powershell
del *.aux, *.log, *.dvi, *.bbl, *.blg, *.toc, *.out
```

## 参考文献

参考文献は BibTeX 形式（`references.bib`）で管理しています。

### BibTeX の使用方法

1. **初回コンパイル時**は以下の順序で実行：

   ```powershell
   platex -kanji=utf8 keyboard_assist.tex
   pbibtex -kanji=utf8 keyboard_assist
   platex -kanji=utf8 keyboard_assist.tex
   platex -kanji=utf8 keyboard_assist.tex
   dvipdfmx keyboard_assist.dvi
   ```

2. **参考文献を追加・修正した場合**：
   - `references.bib`を編集
   - 上記の手順を再実行（特に`pbibtex`の実行が必要）

### よくある問題と解決方法

#### 問題 1: `\cite{}`が`?`と表示される

**原因**: BibTeX が実行されていない、または`.bbl`ファイルが生成されていない

**解決方法**:

```powershell
# 1. LaTeXを実行して.auxファイルを生成
platex -kanji=utf8 keyboard_assist.tex

# 2. BibTeXを実行
pbibtex -kanji=utf8 keyboard_assist

# 3. LaTeXを2回実行（参照を解決）
platex -kanji=utf8 keyboard_assist.tex
platex -kanji=utf8 keyboard_assist.tex

# 4. PDF生成
dvipdfmx keyboard_assist.dvi
```

#### 問題 2: `! Undefined control sequence. \newblock`

**原因**: IPSJ スタイルファイルは`\newblock`をサポートしていない

**解決方法**: `references.bib`の`note`フィールドを削除するか、`note`を使わない形式に変更

#### 問題 3: VS Code の LaTeX Workshop で自動コンパイルされない

**原因**: BibTeX レシピが正しく設定されていない

**解決方法**: `.vscode/settings.json`で以下を確認：

- `latex-workshop.latex.recipe.default`が`"lastUsed"`または`"platex (with bibtex)"`になっている
- `pbibtex`の引数が`"%DOC%"`になっている（`"%DOCFILE%"`ではない）

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
