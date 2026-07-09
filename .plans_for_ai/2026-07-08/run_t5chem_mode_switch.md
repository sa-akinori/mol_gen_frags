# Plan: run_t5chem.sh に学習モード（from_scratch / finetune）切替を追加

- **Date**: 2026-07-08
- **Status**: pending-approval

## Overview

`run_safe.sh` と同様に、`src/train_model/run_t5chem.sh` に `MODE` 変数（`finetune` / `from_scratch`）を導入し、
bash の `if/else` でモードを切り替える。

t5chem（`t5chem/run_trainer.py`）のモード分岐は **`--pretrain` の有無**:
- `--pretrain models/t5chem/pretrained` あり → `T5ForConditionalGeneration.from_pretrained(...)` で**重みを継承**（finetune）。
- `--pretrain` なし → `--tokenizer` 種別の語彙で `T5Config(num_layers=4, num_heads=8, d_model=256, ...)` を構築し
  `T5ForConditionalGeneration(config)` で**ランダム初期化**（from_scratch）。

**確認済みの整合性**:
- pretrained t5chem の config は `num_layers=4, num_heads=8, d_model=256` で、from_scratch が構築する構造と**一致**。
- pretrained の `vocab.txt`（75行, `tokenizer: simple`）と同梱 `vocab/simple.txt` は**完全一致**。
- よって from_scratch は `--tokenizer simple` を指定すれば、**同一構造・同一語彙でのランダム初期化**になる。

| MODE | モデル引数 | `--output_dir` |
|------|-----------|----------------|
| `finetune` | `--pretrain models/t5chem/pretrained` | `models/t5chem/finetuning/rffmg/${FRAG_NAME}` |
| `from_scratch` | `--tokenizer simple` | `models/t5chem/from_scratch/rffmg/${FRAG_NAME}` |

学習率は両モードとも t5chem デフォルト（`--init_lr` 未指定＝5e-4）で、統一方針と整合（変更しない）。
`--task_type product`, `--num_epoch 50` も現状維持。

## Plan

### Step 1: run_t5chem.sh にモード切替を実装

- **Target file**: `src/train_model/run_t5chem.sh`
- **Changes**:
  - `FRAG_NAME` の直後に `MODE="finetune"  # "finetune" or "from_scratch"` を追加。
  - `if/elif/else` で分岐:
    - `finetune`: `MODEL_ARG="--pretrain models/t5chem/pretrained"`、`OUTPUT_DIR="models/t5chem/finetuning/rffmg/${FRAG_NAME}"`
    - `from_scratch`: `MODEL_ARG="--tokenizer simple"`、`OUTPUT_DIR="models/t5chem/from_scratch/rffmg/${FRAG_NAME}"`
    - それ以外: stderr にエラー出力して `exit 1`。
  - `t5chem train` 呼び出しを複数行化し、`${MODEL_ARG}` と `--output_dir ${OUTPUT_DIR}` を使用。
    他の引数（`--data_dir`, `--task_type product`, `--num_epoch 50`）は現状維持。
- **Dependencies**: none

### 実装後の run_t5chem.sh（想定内容）

```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

FRAG_NAME="rc_cms"          # "brics" or "rc_cms"
MODE="finetune"            # "finetune" or "from_scratch"

if [ "$MODE" = "finetune" ]; then
    MODEL_ARG="--pretrain models/t5chem/pretrained"
    OUTPUT_DIR="models/t5chem/finetuning/rffmg/${FRAG_NAME}"
elif [ "$MODE" = "from_scratch" ]; then
    MODEL_ARG="--tokenizer simple"
    OUTPUT_DIR="models/t5chem/from_scratch/rffmg/${FRAG_NAME}"
else
    echo "Unknown MODE: ${MODE} (use 'finetune' or 'from_scratch')" >&2
    exit 1
fi

t5chem train \
--data_dir data/rffmg/${FRAG_NAME}/normal \
--output_dir ${OUTPUT_DIR} \
${MODEL_ARG} \
--task_type product \
--num_epoch 50
```

## Notes / 確認事項

- 既定は `MODE="finetune"`。
- from_scratch は `--tokenizer simple`（pretrained と同一語彙）で明示指定。指定しなくてもデフォルト simple になるが、
  警告回避と明確化のため明示する。
- 命名: t5chem の finetune 出力は `finetuning/`（ユーザー指定）。safe 側の finetune 出力 `trained/` とは名称が異なる点は許容。
- 対象は `run_t5chem.sh` のみ。生成側スクリプトや oxygen リポジトリへの適用は含めない（要望あれば別途）。
