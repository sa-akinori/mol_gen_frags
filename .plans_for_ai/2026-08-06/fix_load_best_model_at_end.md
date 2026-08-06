# Plan: load_best_model_at_end を直書きに変更する

- **Date**: 2026-08-06
- **Status**: approved

## Overview

`train_fraggpt.py` / `train_promptsmiles.py` の `--load_best_model_at_end` は
`action='store_true'`（デフォルト `False`）だが、両スクリプトが使う
`EarlyStoppingCallback.on_train_begin` は次の assert を持つ（transformers 4.45.2 で実測）。

```python
assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
```

このため `.sh` を経由せず `.py` を直接起動すると、**1,714,298 行のトークナイズが終わった後に
AssertionError で落ちる**。`train_gpt.py`（RFFMG）は `load_best_model_at_end=True` を
直書きしており、この落とし穴がない。RFFMG に合わせて CLI フラグを廃止し直書きにする。

### 影響の確認

`train_promptsmiles.py` は既に実行済み（`models/promptsmiles/gpt/finetuning/brics/best_model` が存在）だが、
その実行は `.sh` 経由で `--load_best_model_at_end` を渡していたため、**変更後と挙動は同一**。
既存モデルの作り直しは不要。

### スコープ外（ユーザー判断で対応しないと決めた項目）

- `.sh` の `conda activate` 環境名（`env_fraggpt` → `fraggpt` 等）および
  `run_fraggpt.sh` のヘッダ復元。**手動で環境を activate してから実行するため問題なし**。
- `LogFile` の上書きモード。**同じ `mode` × `frag_method` の再実行で上書きされるのは問題なし**。
- `generation_fraggpt_func.py` の `except ValueError` 不整合（別タスク）。

## Plan

### Step 1: `train_fraggpt.py` から `--load_best_model_at_end` を削除して直書きにする

- **Target file**: `src/train_model/train_fraggpt.py`
- **Changes**:
  - `parse_args()` 内の
    `parser.add_argument("--load_best_model_at_end", action='store_true', help="Load best model at end (default: False)")`
    （102行）を削除する。
  - `TrainingArguments` の `load_best_model_at_end=args.load_best_model_at_end`（144行）を
    `load_best_model_at_end=True` にする。
  - 他の引数・ハイパラ・`TrainingArguments` の他のフィールドは一切変更しない。
- **Dependencies**: none

### Step 2: `train_promptsmiles.py` に同一の変更を行う

- **Target file**: `src/train_model/train_promptsmiles.py`
- **Changes**: Step 1 と完全に同じ。
  - `parse_args()` 内の `--load_best_model_at_end` の `add_argument`（109行）を削除する。
  - `TrainingArguments` の `load_best_model_at_end=args.load_best_model_at_end`（152行）を
    `load_best_model_at_end=True` にする。
  - SMILES ランダム化の挙動および他のハイパラは一切変更しない。
- **Dependencies**: none

### Step 3: 両 `.sh` から `--load_best_model_at_end` の行を削除する

- **Target file**: `src/train_model/run_fraggpt.sh`, `src/train_model/run_promptsmiles.sh`
- **Changes**:
  - 末尾の `--load_best_model_at_end` 行を削除する。
  - 削除により最終引数となる行（`--per_device_train_batch_size 32`）の行末の
    継続文字 `\` を外す。
  - **この削除は必須**。argparse から引数が消えるため、残すと
    `unrecognized arguments: --load_best_model_at_end` で起動に失敗する。
  - `.sh` の他の行（`FRAG_NAME` / `MODE` / `WANDB_*` / コメント）は一切変更しない。
- **Dependencies**: after Step 1, Step 2

### Step 4: 検証（メインエージェントが実施）

- **Target file**: 実行のみ（コード変更なし）
- **Changes**:
  - `.py` 2件の構文チェック、`.sh` 2件の `bash -n`
  - `load_best_model_at_end` を repo 内で grep し、残っているのが
    `TrainingArguments(... load_best_model_at_end=True ...)` の2箇所だけであることを確認
  - `python src/train_model/train_fraggpt.py --help` にフラグが出ないこと
  - `.py` を**フラグなしで直接起動**して `EarlyStoppingCallback` の assert を通過すること
    （極小データで数ステップ）
- **Dependencies**: after Step 3
