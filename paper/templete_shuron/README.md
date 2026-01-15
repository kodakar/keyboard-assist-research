# 修論

# Usage.

## 自分の作業ブランチを作る

```
# その年のテンプレートがあるブランチへ
git checkout template/16fmi

# 自分ブランチを切る
git checkout -b 16fmi/hiro

# テンプレートファイルコピー
cp syuron_format.tex hiro_thesis.tex
```

## 今年のテンプレートを作る

```
# テンプレートブランチへ
git checkout master

# 今年向けに編集してコミット
git commit

# ブランチ作成
git checkout -b template/16fmi

```
