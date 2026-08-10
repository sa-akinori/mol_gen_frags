# Plan: split_repr_and_model_args

- **Date**: 2026-08-06
- **Status**: completed (rev.2 — `arc_name` を完全廃止する方針に修正。2026-08-06 に Step 1-6 実装・検証済み)

## Overview

`src/evaluation.py:35-44` の分岐が、**手法(表現)軸**と**モデル軸**という直交する 2 つの軸を
`--model_name` という 1 つの引数に混ぜている。`safe_gpt` は「safe 手法 × gpt モデル」の合成語であり、
軸として成立していない。

正しい軸は以下の 2 つで、ディレクトリ構成もそのとおりになっている (`results/{手法}/{モデル}/...`)。

| 手法(表現) | モデル | 実ディレクトリ |
|---|---|---|
| rffmg | t5chem | `results/rffmg/t5chem/`, `models/rffmg/t5chem/`, `data/rffmg/` |
| rffmg | gpt | `results/rffmg/gpt/` |
| safe | gpt | `results/safe/gpt/` |
| promptsmiles | gpt | `results/promptsmiles/gpt/` |
| fraggpt | gpt | `results/fraggpt/gpt/` |

`gen_rffmg.py` の `--model_name {t5chem, gpt}` は既にモデル軸として正しく、`evaluation.py` だけが
食い違っている。本計画で `evaluation.py` を 2 軸に分離し、内部変数 (`str_name` / `model_dir` /
`arc_name`) の命名も実態に合わせる。

### 現状の派生値と、2 軸化後の対応

| 現在の変数 | 意味 | 2 軸化後 |
|---|---|---|
| `str_name` | 手法のディレクトリ名 | `repr_name` (引数から直接) |
| `model_dir` | モデルのディレクトリ名 | `model_name` (引数から直接。派生不要) |
| `arc_name` | predictions.csv のレイアウト | **廃止** (`repr_name` をそのまま渡す) |
| `default_gen_method` | 既定デコード方式 | `'sampling' if repr_name == 'promptsmiles' else 'beam'` |

手法ごとに本当に違うのは **(1) predictions.csv のレイアウト (safe だけ独自)**、
**(2) 既定のデコード方式 (promptsmiles だけ sampling)**、**(3) 学習データの置き場所** の 3 点だけであり、
5 分岐は不要になる。

### `arc_name` を廃止できる根拠 (rev.2 で確認済み)

`arc_name` は `repr_name` から機械的に決まる冗長な中間変数であり、独立した情報を持っていない。

- 使用箇所は 2 つだけ。`loadTrainSmiles` では**関数内で一度も参照されておらず** (docstring にも
  「reader は与えられたパスの種類で選ぶ。arc_name は reader を選ばない」と明記されている)、
  実質的に `loadGenSmiles` の 2 分岐のためだけに存在する。
- その 2 分岐は「safe か、それ以外か」であり、`repr_name == 'safe'` と 1:1 で対応する。
  根拠は生成側の出力形式:

  | repr_name | predictions.csv の書き方 | index 列 | `fragment` 列 | 失敗行 |
  |---|---|---|---|---|
  | safe | `to_csv(path)` (`generation_safe_func.py:178`) | あり | 自前で持つ | `time_out` / `error` 文字列 |
  | rffmg / promptsmiles / fraggpt | `to_csv(path, index=False)` | なし | `test.source` から横結合 | 空文字 |

- `model_name` はレイアウトに一切影響しない (`rffmg × t5chem` と `rffmg × gpt` は同一レイアウト)。
- `figure.py` / `analyze_predictions.py` は同名のローカル変数 `arc_name` を持つだけで、
  `loadGenSmiles` / `loadTrainSmiles` を呼んでいない。呼び出し元は `evaluation.py` の 2 箇所のみ。

### 引数名の提案

`--frag_method` (フラグメント化法) と `--gen_method` (デコード方式) が既にあるため、
手法軸には **`--repr_name`** を提案する (structure.md の「表現 × モデル」に対応)。
`--method` は `--frag_method` / `--gen_method` と紛らわしいため避ける。
別名がよければ指示してください (`--str_name` / `--method` など)。

### 既定値の変更について

- `--repr_name`: 既定 `rffmg`
- `--model_name`: 既定 **`gpt`** に変更する (現在は `t5chem`)。`t5chem` は rffmg でしか存在しないのに対し、
  `gpt` は 4 手法すべてに存在するため。`rffmg × t5chem` を使うときは `--model_name t5chem` を明示する。

## Plan

### Step 1: 引数を 2 軸に分離

- **Target file**: `src/evaluation.py`
- **Changes**:
  - `--repr_name` を新設する (`type=str, default='rffmg', choices=['rffmg', 'safe', 'promptsmiles', 'fraggpt']`,
    help に「手法(表現)。`data/` `models/` `results/` の第 1 階層」と明記)。
  - `--model_name` の `choices` を `['t5chem', 'gpt']` に絞り、`default='gpt'` にする
    (help に「モデル。`results/{repr_name}/{model_name}/` の第 2 階層」と明記)。
  - 組み合わせの検証を入れる: `t5chem` は `rffmg` でしか学習されていないため、
    `model_name == 't5chem' and repr_name != 'rffmg'` のとき `parser.error()` で明示的に落とす。
- **Dependencies**: none

### Step 2: 派生値の導出を書き換え

- **Target file**: `src/evaluation.py`
- **Changes**:
  - 現 `evaluation.py:31-44` の 5 分岐を**まるごと削除**する。`str_name` / `model_dir` / `arc_name` /
    `default_gen_method` の 4 変数はすべて廃止し、`repr_name` / `model_name` をそのまま使う。
  - `gen_method = args.gen_method or ('sampling' if repr_name == 'promptsmiles' else 'beam')` にまとめる。
  - `sampling_seg` の条件を `str_name == 'rffmg'` から `repr_name == 'rffmg'` に変更する。
  - 現 `evaluation.py:31-34` のコメントを、2 軸であること・手法ごとに違うのは
    「predictions.csv のレイアウト」「既定のデコード方式」「学習データの置き場所」の 3 点だけであることを
    説明する内容に書き直す。
  - `loadTrainSmiles` / `loadGenSmiles` の呼び出しを `loadTrainSmiles(tr_file_name)` /
    `loadGenSmiles(repr_name, file_name, testInputfile)` に変更する (Step 4 と対応)。
- **Dependencies**: after Step 1

### Step 3: データパス分岐を手法軸で整理

- **Target file**: `src/evaluation.py`
- **Changes**:
  - 現 `evaluation.py:55-76` の分岐を `repr_name` ベースに書き換える。
    `promptsmiles` と `fraggpt` はディレクトリ名以外が完全に同一なので 1 分岐に統合し、
    `data/{repr_name}/{frag_method}/...` で組み立てる。
  - `outfd` を `results/{repr_name}/{model_name}/{model_ver}/{frag_method}/{sampling_seg}{gen_method}/{additional_path}` にする。
  - `frag_order` ブロック (現 `evaluation.py:108-113`) の `outfd` も同様に書き換える
    (`datafd` は既に `data/rffmg/...` 固定なので変更不要)。
  - `--sampling_num` の help を「`--model_name` が t5chem か gpt のとき」から
    「`--repr_name` が rffmg のとき」に修正する。
- **Dependencies**: after Step 2

### Step 4: `evaluation_func.py` から `arc_name` を廃止する

- **Target file**: `src/func/evaluation_func.py`
- **Changes**:
  - `loadGenSmiles` の第 1 引数 `arc_name` を `repr_name` に置き換え、分岐条件を
    `if arc_name == 'safe_gpt'` / `elif arc_name == 't5chem'` から
    `if repr_name == 'safe'` / `else` に変更する。`'safe_gpt'` / `'t5chem'` という
    レイアウト名の語彙をなくし、手法名だけで分岐させる。
    docstring も更新し、safe だけ生成側が index 付き・`fragment` 列入りで書き出すこと、
    他の 3 手法は `test.source` を横結合することを明記する。
  - `loadTrainSmiles` の `arc_name` 引数は関数内で一切使われていないため削除する
    (docstring の表も `arc_name` の行を削り、パスの種類でリーダを選ぶ説明だけ残す)。
- **Dependencies**: none (Step 1-3 と独立だが、`evaluation.py` の呼び出し側修正は Step 2 と同時に必要)

### Step 5: README の実行例を 2 軸に更新

- **Target file**: `README.md`, `README_ja.md`
- **Changes**: `README.md:247-251` / `README_ja.md:243-247` の実行例とコメントを新しい引数体系に更新する。
  `--model_name: t5chem / gpt (RFFMG-GPT) / safe_gpt / promptsmiles / fraggpt` という記述を
  `--repr_name` と `--model_name` の 2 行に分け、有効な組み合わせの表を添える。
- **Dependencies**: after Step 3

### Step 6: 検証

- **Target file**: なし (検証のみ)
- **Changes**:
  - `python -m py_compile` で変更ファイルの構文を確認する。
  - 有効な 5 組み合わせについて、`tr_file_name` / `testInputfile` / `outfd` が
    実ファイル (rffmg) または生成側スクリプトの出力先文字列 (safe / promptsmiles / fraggpt) と
    一致することを確認する。
  - 無効な組み合わせ (`--repr_name safe --model_name t5chem`) が `parser.error()` で落ちることを確認する。
- **Dependencies**: after Step 5

## 実装対象外 (要否を確認したい)

以下にも同じ軸の混同が残っているが、今回の指摘範囲 (`evaluation.py:35-44`) 外なので本計画には含めていない。
必要ならステップを追加する。

- **`src/figure.py:171-176`**: `arc_name = 't5chem' # ['t5chem', 'safe_gpt', 'gpt']` から
  `str_name` / `model_dir` を派生している (evaluation.py と同じ混同)。加えて変数 `model_name` に
  `'trained'` (= model_ver) が入っており命名がずれている。さらに結果パスに `{N}times_sampling` が無い。
- **`src/analyze_predictions.py:193-203`**: `results/{arc_name}/{model_name}/{str_name}/...` と
  **階層の順序が現在の出力レイアウトと違う** (正しくは `{手法}/{モデル}/{model_ver}/...`)。
  こちらは動作しない可能性が高い。
