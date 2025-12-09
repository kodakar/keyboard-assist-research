# コンパイルガイド - BibTeX 対応版

## 問題の原因と解決方法

### 発生していた問題

1. **`\cite{}`が`?`と表示される**

   - BibTeX が実行されていない
   - `.bbl`ファイルが生成されていない

2. **参考文献リストが表示されない**

   - BibTeX が実行されていない
   - `.bbl`ファイルが存在しない

3. **`! Undefined control sequence. \newblock`エラー**
   - IPSJ スタイルファイル（`ipsjunsrt.bst`）が`\newblock`をサポートしていない
   - `references.bib`の`note`フィールドに`\newblock`が含まれていた

### 解決方法

#### 1. BibTeX の実行順序

BibTeX を使う場合、以下の順序で実行する必要があります：

```powershell
# ステップ1: LaTeXを実行（.auxファイルを生成）
platex -kanji=utf8 keyboard_assist.tex

# ステップ2: BibTeXを実行（.bblファイルを生成）
pbibtex -kanji=utf8 keyboard_assist

# ステップ3: LaTeXを2回実行（参照を解決）
platex -kanji=utf8 keyboard_assist.tex
platex -kanji=utf8 keyboard_assist.tex

# ステップ4: PDF生成
dvipdfmx keyboard_assist.dvi
```

**なぜ 2 回 LaTeX を実行するのか？**

- 1 回目：`.bbl`ファイルを読み込んで参考文献リストを生成
- 2 回目：相互参照（`\ref{}`、`\cite{}`など）を解決

#### 2. `note`フィールドの問題

IPSJ スタイルファイルは`\newblock`コマンドをサポートしていないため、
`references.bib`の`note`フィールドは削除するか、使わない形式にしてください。

**修正前（エラーになる）:**

```bibtex
@misc{example,
  title = {Example},
  year = {2024},
  note = {Accessed: 2024}  % ← これが\newblockを生成してエラー
}
```

**修正後（正しい）:**

```bibtex
@misc{example,
  title = {Example},
  year = {2024}
  % noteフィールドを削除
}
```

#### 3. VS Code LaTeX Workshop の設定

`.vscode/settings.json`で以下を確認：

```json
{
  "latex-workshop.latex.recipe.default": "lastUsed",
  "latex-workshop.latex.recipes": [
    {
      "name": "platex (with bibtex)",
      "tools": ["platex", "pbibtex", "platex", "platex", "dvipdfmx"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "pbibtex",
      "command": "pbibtex",
      "args": [
        "-kanji=utf8",
        "%DOC%" // ← "%DOCFILE%"ではなく"%DOC%"
      ]
    }
  ]
}
```

## 他の環境での再現性

### 必要な設定

1. **BibTeX スタイルファイル**: `ipsjunsrt.bst`が同じディレクトリにあること
2. **`references.bib`**: `note`フィールドを使わないこと
3. **コンパイル順序**: 上記の順序を守ること

### チェックリスト

- [ ] `references.bib`に`note`フィールドがない（または`\newblock`を含まない）
- [ ] `ipsjunsrt.bst`が同じディレクトリにある
- [ ] `pbibtex`コマンドが利用可能
- [ ] コンパイル順序が正しい（platex → pbibtex → platex → platex → dvipdfmx）

### トラブルシューティング

#### `.bbl`ファイルが生成されない

```powershell
# .auxファイルが存在するか確認
ls keyboard_assist.aux

# 存在しない場合は、まずLaTeXを実行
platex -kanji=utf8 keyboard_assist.tex

# その後、BibTeXを実行
pbibtex -kanji=utf8 keyboard_assist

# .bblファイルが生成されたか確認
ls keyboard_assist.bbl
```

#### 参考文献が表示されない

1. `.bbl`ファイルが存在するか確認
2. 論文内で`\cite{}`を使っているか確認
3. `references.bib`に該当するキーが存在するか確認

#### 文字化けが発生する

- `-kanji=utf8`オプションを付けて実行
- `pbibtex`にも`-kanji=utf8`を付ける

## 自動化スクリプト

### PowerShell 版（Windows）

```powershell
# compile.ps1
param(
    [switch]$KeepIntermediate,
    [switch]$Clean
)

$texFile = "keyboard_assist.tex"

if ($Clean) {
    Write-Host "中間ファイルを削除中..."
    Remove-Item -ErrorAction SilentlyContinue *.aux, *.log, *.dvi, *.bbl, *.blg, *.toc, *.out
    exit
}

Write-Host "LaTeXコンパイル開始..."

# ステップ1: LaTeX
Write-Host "[1/5] LaTeX実行中..."
& platex -kanji=utf8 -synctex=1 -interaction=nonstopmode $texFile

# ステップ2: BibTeX
Write-Host "[2/5] BibTeX実行中..."
& pbibtex -kanji=utf8 keyboard_assist

# ステップ3-4: LaTeX（2回）
Write-Host "[3/5] LaTeX実行中（1回目）..."
& platex -kanji=utf8 -synctex=1 -interaction=nonstopmode $texFile
Write-Host "[4/5] LaTeX実行中（2回目）..."
& platex -kanji=utf8 -synctex=1 -interaction=nonstopmode $texFile

# ステップ5: PDF生成
Write-Host "[5/5] PDF生成中..."
& dvipdfmx keyboard_assist.dvi

if (-not $KeepIntermediate) {
    Write-Host "中間ファイルを削除中..."
    Remove-Item -ErrorAction SilentlyContinue *.aux, *.log, *.dvi, *.bbl, *.blg, *.toc, *.out
}

Write-Host "完了！"
```

### Bash 版（Mac/Linux）

```bash
#!/bin/bash
# compile.sh

TEXFILE="keyboard_assist.tex"

echo "LaTeXコンパイル開始..."

# ステップ1: LaTeX
echo "[1/5] LaTeX実行中..."
platex -kanji=utf8 -synctex=1 -interaction=nonstopmode "$TEXFILE"

# ステップ2: BibTeX
echo "[2/5] BibTeX実行中..."
pbibtex -kanji=utf8 keyboard_assist

# ステップ3-4: LaTeX（2回）
echo "[3/5] LaTeX実行中（1回目）..."
platex -kanji=utf8 -synctex=1 -interaction=nonstopmode "$TEXFILE"
echo "[4/5] LaTeX実行中（2回目）..."
platex -kanji=utf8 -synctex=1 -interaction=nonstopmode "$TEXFILE"

# ステップ5: PDF生成
echo "[5/5] PDF生成中..."
dvipdfmx keyboard_assist.dvi

echo "完了！"
```
