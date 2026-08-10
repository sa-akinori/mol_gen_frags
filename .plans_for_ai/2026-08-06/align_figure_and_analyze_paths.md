# Plan: align_figure_and_analyze_paths

- **Date**: 2026-08-06
- **Status**: pending-approval

## Overview

`src/figure.py` と `src/analyze_predictions.py` に、`evaluation.py` で解消したのと同じ
「手法(表現)軸とモデル軸の混同」が残っている。加えて両者とも結果パスが現在の出力レイアウトと合っていない。

両ファイルを `evaluation.py` と同じ 2 軸の語彙 (`repr_name` / `model_name`) に揃え、
パス組み立てを現在のレイアウトに合わせる。

### 現状の問題

**`src/figure.py:167-180`**

| 行 | 問題 |
|---|---|
| 171-173 | `arc_name = 't5chem' # ['t5chem','safe_gpt','gpt']` から `str_name` / `model_dir` を派生 (evaluation.py と同じ混同) |
| 174 | 変数名 `model_name` に `'trained'` (= model_ver) が入っている。しかも `trained` は `from_scratch` の旧称 |
| 176 | `path_prefix` の階層順 `{手法}/{モデル}/{ver}` 自体は正しいが、rffmg の `{N}times_sampling` が無い |
| 177 | `slice_method` は `evaluation.py` の `frag_method` と同じもの。名前が揃っていない |
| 178 | `gen_method` のコメントが `['beam', 'random']`。実際の選択肢は `beam` / `sampling` |
| 193-371 | 同じ長い f-string が 20 箇所以上に重複。`additional_path` は `normal` 決め打ちで変数化されていない |
| 344,347,349 | `data/{str_name}/{slice_method}/normal/train.source` — rffmg の `{N}times_sampling` が無い |

**`src/analyze_predictions.py:193-197`**

| 行 | 問題 |
|---|---|
| 193-194 | `arc_name = 'safe_gpt'` から `str_name` を派生 (同じ混同) |
| 195 | 変数名 `model_name` に `'trained'` (= model_ver) が入っている |
| 201-230 | パスが `results/{arc_name}/{model_name}/{str_name}/{frag}/{gen}/...`。**手法が 2 回出てきており、階層順も現在のレイアウトと違う**。正しくは `results/{repr_name}/{model_name}/{model_ver}/{frag_method}/[{sampling}/]{gen_method}/{additional_path}/` |
| 201-230 | `if 0:` ブロックでは `{additional_path}` セグメント自体が欠けている (`if 1:` ブロックの 234/256/274 にはある) |
| 280-281 | `data/{str_name}/{slice_method}/...` — rffmg の `{N}times_sampling` が無い |

### 揃える先 (evaluation.py と同じ語彙)

```python
repr_name    = 'rffmg'       # ['rffmg', 'safe', 'promptsmiles', 'fraggpt']
model_name   = 't5chem'      # ['t5chem', 'gpt']
model_ver    = 'finetuning'  # ['pretrained', 'finetuning', 'from_scratch']
frag_method  = 'brics'       # ['brics', 'rc_cms']
gen_method   = 'beam'        # ['beam', 'sampling']
sampling_num = 5             # [5, 10] rffmg のみ
additional_path = 'normal'   # ['normal', 'dup_frags', 'frag_num', 'frag_order', 'attach_point_num']
sampling_seg = f'{sampling_num}times_sampling/' if repr_name == 'rffmg' else ''
```

### 設計判断

- **argparse 化はしない。** 両ファイルは `if 0:` / `if 1:` で実行ブロックを切り替える現行スタイルを維持し、
  設定ブロックの変数名とパス組み立てだけを直す。
- **長い f-string の重複は共通変数に集約する。** `result_dir` (results 側) と `path_prefix` (figures 側) を
  設定ブロックで 1 度だけ組み立て、各ブロックはそれを使う。20 箇所以上の重複が消え、
  今後レイアウトが変わったときの修正漏れもなくなる。

## Plan

### Step 1: `figure.py` の設定ブロックを 2 軸に書き換え

- **Target file**: `src/figure.py`
- **Changes**: `figure.py:170-179` を上記「揃える先」の変数群に置き換える。
  - `arc_name` / `str_name` / `model_dir` を廃止し、`repr_name` / `model_name` を直接指定する。
  - `model_name = 'trained'` を `model_ver = 'finetuning'` にリネームする
    (選択肢コメントから旧称 `trained` を外し、`pretrained` / `finetuning` / `from_scratch` にする)。
  - `slice_method` を `frag_method` にリネームする (`evaluation.py` と同名)。
  - `gen_method` のコメントを `['beam', 'sampling']` に直す。
  - `sampling_num` と `additional_path` を新設する。
- **Dependencies**: none

### Step 2: `figure.py` のパス組み立てを共通変数に集約

- **Target file**: `src/figure.py`
- **Changes**:
  - 設定ブロックで `sampling_seg` と、以下 2 つの共通プレフィックスを組み立てる。
    - `result_dir = f'{fd}/results/{repr_name}/{model_name}/{model_ver}/{frag_method}/{sampling_seg}{gen_method}/{additional_path}'`
    - `path_prefix = f'{repr_name}/{model_name}/{model_ver}/{frag_method}/{sampling_seg}{gen_method}'`
      (figures 側は末尾の条件セグメントがブロックごとに違うため、`additional_path` は含めない)
  - `figure.py:193-371` の各パスを、上の 2 変数を使った形に置き換える。
    `f'{fd}/results/{path_prefix}/{slice_method}/{gen_method}/normal/...'` のような重複記述をなくす。
  - `const_name` を使うブロック (315, 322) は `path_prefix` + `{const_name}` で組み立てる
    (`additional_path` とは別のループ変数なのでそのまま残す)。
- **Dependencies**: after Step 1

### Step 3: `figure.py` の `data/` 参照を修正

- **Target file**: `src/figure.py`
- **Changes**: `figure.py:344,347,349` の `data/{str_name}/{slice_method}/...` を
  `data/{repr_name}/{frag_method}/{sampling_seg}...` に直す
  (`train.source` は `data/rffmg/{frag}/{N}times_sampling/normal/train.source` に実在することを確認済み)。
- **Dependencies**: after Step 1

### Step 4: `analyze_predictions.py` の設定ブロックを 2 軸に書き換え

- **Target file**: `src/analyze_predictions.py`
- **Changes**: `analyze_predictions.py:193-197` を Step 1 と同じ変数群に置き換える
  (`arc_name` / `str_name` を廃止、`model_name` → `model_ver` のリネーム、`slice_method` → `frag_method`、
  `sampling_num` / `additional_path` の新設)。
- **Dependencies**: none

### Step 5: `analyze_predictions.py` のパス階層順を修正

- **Target file**: `src/analyze_predictions.py`
- **Changes**:
  - 設定ブロックで `result_dir` (Step 2 と同じ組み立て) と figures 用の `path_prefix` を作る。
  - `analyze_predictions.py:201-274` のパスを `result_dir` ベースに置き換える。
    これにより **手法が 2 回出る問題・階層順の誤り・`{additional_path}` の欠落**が同時に解消する。
  - `analyze_predictions.py:225-230,302-303` の `figures/` 側パスも `path_prefix` ベースに揃える。
  - `analyze_predictions.py:280-281` の `data/{str_name}/{slice_method}/...` を
    `data/{repr_name}/{frag_method}/{sampling_seg}...` に直す。
- **Dependencies**: after Step 4

### Step 6: 検証

- **Target file**: なし (検証のみ)
- **Changes**:
  - `python -m py_compile` で両ファイルの構文を確認する。
  - 設定ブロックだけを切り出して、`result_dir` が `evaluation.py` の `outfd` と
    文字列として一致することを 5 組み合わせすべてで確認する
    (`rffmg × {t5chem, gpt}`, `{safe, promptsmiles, fraggpt} × gpt`)。
  - rffmg については `result_dir` 配下の `predictions.csv` / `curated_data.tsv` が実在することを確認する。
- **Dependencies**: after Step 5

## Notes (実装対象外・報告のみ)

- **既存の `figures/` は再生成が必要。** 現在の `figures/physic_property/safe_gpt/trained/` などは
  `{arc_name}/{model_ver}/` の 2 段で、今回揃える 5 段のレイアウトとは別物。今回の変更で自動的には移動されない。
- **`figure.py:304` の `data/dummy/{frag}/{const_name}/target_frags.pkl` は存在しないディレクトリを参照している**
  (`if 0:` ブロック内)。軸の混同とは別問題なので今回は触らない。
- 両ファイルとも `if 0:` / `if 1:` で実行ブロックを切り替える設計のままにする。argparse 化はしない。
