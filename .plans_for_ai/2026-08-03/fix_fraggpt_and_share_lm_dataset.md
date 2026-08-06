# Plan: FragGPT の実行時エラー修正と無条件LM データセットの共通化

- **Date**: 2026-08-03
- **Status**: pending-approval

## Overview

FragGPT の学習・生成が**どちらも起動しない**状態にあり、あわせて `train_fraggpt.py` /
`train_promptsmiles.py` が HF 標準部品で置き換え可能な自作クラスを重複して持っている。
両方をまとめて解消する。

### 現在壊れている箇所（実行すれば即座に落ちる）

| # | 箇所 | 症状 |
|---|---|---|
| 1 | `fragment_for_fraggpt.py:163,167` | `augment_fusmiles` が **`KeyError: '1'`**。`findall` 側の `r'\[\d+\*\]'` に捕捉群が無く辞書キーが `'[1*]'` になる一方、置換側は `m.group(1)`（`'1'`）で引くため不一致。学習開始時に落ちる |
| 2 | `generation_fraggpt_func.py:14,17` | 存在しない `FRAGMENT_SEPARATOR` と `split_fragments` を import しており **`ImportError`**。生成が起動しない |

### 自作クラスの重複（HF 標準部品で置換可能）

実測で等価性を確認済み。`DataCollatorForLanguageModeling(tokenizer, mlm=False)` は自作 collator と
`input_ids` / `attention_mask` / `labels` の3つとも**完全一致**した
（`pad_token` が `<pad>` で `eos`(`</s>`) と別トークンのため、実 eos の label が誤って `-100` にならない）。

| クラス | 定義場所 | 置換先 |
|---|---|---|
| `FragGPTDataset` | `train_fraggpt.py` | HF `Dataset.map()` |
| `DataCollatorForCausalLM` | `train_fraggpt.py` | `DataCollatorForLanguageModeling(mlm=False)` |
| `PromptSMILESDataset` | `train_promptsmiles.py` | HF `Dataset.map()` |
| `DataCollatorForCausalLM` | `train_promptsmiles.py` | 同上 |

`train_gpt.py`（RFFMG）の `RFFMGDataset` と同名 collator は**対象外**。プロンプト部を `-100` で
マスクする条件付き学習であり、`DataCollatorForLanguageModeling` では代替できない。

### 副次的な利点

- `.map(num_proc=N)` でトークン化を並列化できる（現在は rc_cms 665万行を単一プロセスの Python ループで処理）
- 結果がディスクにキャッシュされ、2回目以降の実行ではトークン化をスキップできる
- `max_length` 超過の除外が `.filter()` 一行で書け、除外件数も `len()` の差で取れる

### max_length について（調査結果）

`entropy/gpt2_zinc_87m` は **`n_positions = 256`** のため、`max_length` を 256 より大きくできない
（位置埋め込みを拡張するか別モデルにするしかなく、いずれも RFFMG-GPT と同一モデルという
比較の前提を壊す）。したがって**超過行は除外する**方針とする（ユーザー判断）。

参考: 同一分子・同一 tokenizer での長さ実測（train 先頭20,000行）

| 表現 | brics 平均 / 最大 | rc_cms 平均 / 最大 |
|---|---|---|
| PromptSMILES（分子SMILES） | 34.2 / 105 | 35.4 / 110 |
| SAFE（`full_safe`） | 46.4 / 139 | 45.0 / 152 |
| FragGPT（`full_fragments`） | 71.8 / 210 | 66.8 / 205 |

FU-SMILES は切断結合1本を `[i*]` × 2箇所（各4トークン）で表すため、閉環数字で表す SAFE の
約1.5倍、分子SMILES の約2.1倍になる。この標本では 256 超過は 0 件だが、全体では裾に存在しうる。

## Plan

### Step 1: `augment_fusmiles` の KeyError を修正する

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**: 検証済みの実装に戻す。
  - モジュールレベルに `ATTACHMENT_LABEL = re.compile(r"\[(\d+)\*\]")` を定義する
    （`\]` まで含める。含めないと `[10*]` を `[1*]` と誤マッチする）。
    内包表記の中で `re.compile` を毎回呼ばない。
  - ラベルは**数値として**扱い、`1..n` に振り直す
    （現在の `sorted()` は文字列辞書順で `'[1*]' < '[10*]'`、かつ `labels.copy()` は
    既存ラベルの置換にとどまり、論文の「`1~n` の値に振り直す」と異なる）。
  - 置換は `ATTACHMENT_LABEL.sub` にコールバックを渡す**1パスの同時置換**にする
    （逐次 `str.replace` だと `1→2` の後に `2→1` を適用して壊れる）。
  - フラグメント順のシャッフルは現行のまま維持する。
  - シグネチャは現行の `(fusmiles: str, rng) -> str` を維持する（呼び出し側を変えないため）。
  - docstring の `Args:` が `fragments` のままなので `fusmiles` に直す。
- **Dependencies**: none

### Step 2: `FRAGMENT_SEPARATOR` と `split_fragments` を復活させる

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**: `generation_fraggpt_func.py` が import している2つを定義する。
  - `FRAGMENT_SEPARATOR = "."`（モジュールレベル定数。`DUMMY_ATOMIC_NUM` などの並びに置く）
  - `split_fragments(fusmiles: str) -> list[str]`:
    `FRAGMENT_SEPARATOR` で分割し、**空要素を除く**。
    空要素の除去はモデル出力（末尾 `.` や `..` を含みうる任意の文字列）を扱う生成側で必須。
  - `augment_fusmiles` 内の `fusmiles.split(".")` もこの2つを使う形に統一する。
- **Dependencies**: none（Step 1 と同一ファイルなので同時に実施してよい）

### Step 3: 未使用の `assemble_fragments` を削除する

- **Target file**: `src/func/fragment_for_fraggpt.py`
- **Changes**: `assemble_fragments`（102-115行）を削除する。
  コードからの呼び出しは無く、`assemble_fragments_with_reason` の戻り値の1要素目を
  返すだけの薄いラッパである。生成側は `assemble_fragments_with_reason` を使っている。
  削除後に残存参照がないことを grep で確認すること。
- **Dependencies**: none

### Step 4: `train_fraggpt.py` を HF 標準部品に置き換える

- **Target file**: `src/train_model/train_fraggpt.py`
- **Changes**:
  - `FragGPTDataset` と `DataCollatorForCausalLM` を削除する。
  - データ構築を HF `Dataset` のまま行う:
    1. `datasets.load_from_disk(...)` の train / validation split に対し `.map()` で
       `augment_fusmiles` の適用とトークン化を行い、`input_ids` 列（`<bos> … <eos>`）を作る。
       元の列は `remove_columns` で落とす。
    2. `.filter()` で `len(input_ids) <= max_length` の行だけを残す。
    3. **除外件数（filter 前後の行数の差）を train / validation それぞれ標準出力に記録する。**
       黙って捨てないこと。
  - collator は `DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)` を使う。
  - `--num_proc` 引数を追加し（default は `os.cpu_count() - 1` 相当の控えめな値）、
    `.map()` に渡す。並列化時も `rng` による augmentation が再現可能であること
    （`--seed` から決まること）を担保する方法をコード内に明記すること。
    プロセス間で `random.Random` インスタンスを共有できないため、
    **行インデックスからシードを導出する**（例: `random.Random(seed + idx)`）方式にする。
    `with_indices=True` を使えば `.map()` から行インデックスを受け取れる。
  - 未使用になる import（`from rdkit import Chem`、`torch`、`Dataset`、
    `PreTrainedTokenizerBase` 等）を整理する。
  - **学習ハイパラ（LR / epoch / batch / warmup / eval・save / EarlyStopping / seed）は変更しない。**
- **Dependencies**: after Step 1, Step 2

### Step 5: `train_promptsmiles.py` に同じ置き換えを適用する

- **Target file**: `src/train_model/train_promptsmiles.py`
- **Changes**: Step 4 と同一の方針で `PromptSMILESDataset` と `DataCollatorForCausalLM` を削除し、
  `.map()` / `.filter()` / `DataCollatorForLanguageModeling` に置き換える。
  - PromptSMILES は `smiles` 列を読み、`--randomize_smiles` が有効なときのみ
    ランダム根原子での書き直しを行う点が FragGPT と異なる。この分岐は維持する。
  - 除外件数のログ、`--num_proc`、シードの導出方法は Step 4 と揃える。
  - **学習ハイパラは変更しない。**
- **Dependencies**: after Step 4

### Step 6: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - 全変更ファイルの構文チェックと import 確認（`generation_fraggpt_func` が import できること）
  - `augment_fusmiles` が例外なく動き、**変更前の Mol 版と化学的に同一**であることを実データで確認
  - `split_fragments` が空要素を除去すること（`"a..b."` → `["a","b"]`）
  - 小さい HF DatasetDict で `train_fraggpt.py` / `train_promptsmiles.py` が数ステップ回ること
  - `.filter()` による除外件数がログに出ること
  - **同一 `--seed` で2回実行し、学習データが完全に一致すること**（並列化しても再現するか）
  - `train_gpt.py`（RFFMG）が影響を受けていないこと
- **Dependencies**: after Step 5

## スコープ外

- `train_gpt.py`（RFFMG）の `RFFMGDataset` と専用 collator（条件付き学習のため代替不可）
- `max_length` の値そのものの変更（`n_positions = 256` の制約により不可）
- 評価の立体比較の問題（`next_plan.md` の保留項目）
- `make_datasets.py`

## 注意

`data/fraggpt/` と `data/promptsmiles/` は生成済み。
本計画はコードのみの変更で、データの再生成は不要。
