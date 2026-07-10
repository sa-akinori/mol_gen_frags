# Plan: train_gpt.py の修正（引数反映・eval設定・max_length検証）

- **Date**: 2026-07-10
- **Status**: done

## Overview

`src/train_model/train_gpt.py` に対し、レビューで指摘した3点のバグ修正と、max_length の扱い変更を行う。

- 修正1: `--frag_method` が未使用で、`data_dir` 既定が `brics` 固定になっている問題を解消する。
- 修正2: `--mode` が `output_dir` 既定に反映されず、`from_scratch` でも `finetuning` フォルダに保存される問題を解消する。
- 修正3: `per_device_eval_batch_size` が未設定で、ヘルプ記述（train/eval共通）と実挙動（eval既定8）が食い違う問題を解消する。
- max_length: トークン長が `max_length` を超えた場合、無言の truncation をやめて `ValueError` で停止する。

## Plan

### Step 1: data_dir / output_dir 引数を削除し frag_method / mode から組み立てる（修正1・修正2）

- **Target file**: `src/train_model/train_gpt.py`
- **Changes**:
  - `parse_args` から `--data_dir` と `--output_dir` の `add_argument` を**削除**する（CLIオプションから外す）。
  - `__main__` 内で、`args` からローカル変数として
    `data_dir = Path(f"{BASEPATH}/data/rffmg/{args.frag_method}/normal")`、
    `output_dir = f"{BASEPATH}/models/rffmg/gpt/{args.mode}/{args.frag_method}"` を構築する。
  - 既存の `args.data_dir` / `args.output_dir` の参照箇所（`TrainingArguments(output_dir=...)`、
    `trainer.save_model(...)`、`tokenizer.save_pretrained(...)`、データセット読込）を
    ローカル変数 `data_dir` / `output_dir` に置き換える。
  - これにより `--frag_method rc_cms` / `--mode from_scratch` が保存先・読込先に必ず反映される。
- **Dependencies**: none

### Step 2: eval バッチサイズを train と揃える（修正3）

- **Target file**: `src/train_model/train_gpt.py`
- **Changes**:
  - `TrainingArguments` に `per_device_eval_batch_size=args.per_device_train_batch_size` を追加する。
  - `--per_device_train_batch_size` のヘルプ文言「Per-device train/eval batch size」と実挙動を一致させる。
- **Dependencies**: none

### Step 3: max_length 超過時に truncation ではなく raise する

- **Target file**: `src/train_model/train_gpt.py`
- **Changes**:
  - `RFFMGDataset.__init__` の
    `input_ids = ([bos_id] + prompt_ids + target_ids + [eos_id])[:max_length]` から `[:max_length]` を除去。
  - 完成した `input_ids` の長さが `max_length` を超える場合、どの例（zip の index）で
    実際の長さと `max_length` を含む `ValueError` を送出する。
  - `labels` 側も `[:max_length]` を除去（長さは `input_ids` と一致するため検証は input_ids 側で一括で行う）。
  - docstring に「max_length を超える系列があると `ValueError` を送出する」旨を追記する。
- **Dependencies**: none
