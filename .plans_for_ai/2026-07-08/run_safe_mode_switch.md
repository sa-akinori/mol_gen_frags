# Plan: run_safe.sh に学習モード（from_scratch / finetune）切替を追加

- **Date**: 2026-07-08
- **Status**: pending-approval

## Overview

`src/train_model/run_safe.sh` は現状 `--model_path` を渡さず、常に**ゼロ学習**になっている。
一方で出力先は `models/safe_gpt/trained/`（プロジェクト規約では trained=ファインチューニング）に固定されており、
ファインチューニングとゼロ学習を両立できない。

`MODE` 変数（`from_scratch` / `finetune`）を導入し、bash の `if/else` で以下を切り替える:

| MODE | `--model_path` | `--output_dir` |
|------|----------------|----------------|
| `finetune` | `models/safe_gpt/pretrained` を付与（重みを継承） | `models/safe_gpt/trained/safe/${FRAG_NAME}/` |
| `from_scratch` | 付与しない（ランダム初期化） | `models/safe_gpt/from_scratch/safe/${FRAG_NAME}/` |

`--config` / `--tokenizer` は両モードとも pretrained のものを使う（構造・語彙を揃え、比較を公平にするため）。
その他の学習ハイパーパラメータ（epochs/lr/warmup/batch 等）は現状維持。

## Plan

### Step 1: run_safe.sh にモード切替を実装

- **Target file**: `src/train_model/run_safe.sh`
- **Changes**:
  - 先頭の設定に `MODE="finetune"  # "finetune" or "from_scratch"` を追加（`FRAG_NAME` の直後）。
  - `PRETRAINED_DIR="models/safe_gpt/pretrained"` を定義。
  - `if/elif/else` で分岐:
    - `finetune`: `MODEL_PATH_ARG="--model_path ${PRETRAINED_DIR}"`、`OUTPUT_DIR="models/safe_gpt/trained/safe/${FRAG_NAME}/"`
    - `from_scratch`: `MODEL_PATH_ARG=""`、`OUTPUT_DIR="models/safe_gpt/from_scratch/safe/${FRAG_NAME}/"`
    - それ以外: エラーメッセージを stderr に出して `exit 1`。
  - `safe-train` 呼び出しで、`${MODEL_PATH_ARG}`（空なら無視される）と `--output_dir ${OUTPUT_DIR}` を使うよう変更。
    `--config`/`--tokenizer` は `${PRETRAINED_DIR}/config.json` / `${PRETRAINED_DIR}/tokenizer.json` に置換。
  - 他の引数・値は現状のまま。
- **Dependencies**: none

### 実装後の run_safe.sh（想定内容）

```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safe

FRAG_NAME="rc_cms"          # "brics" or "rc_cms"
MODE="finetune"            # "finetune" or "from_scratch"

PRETRAINED_DIR="models/safe_gpt/pretrained"

if [ "$MODE" = "finetune" ]; then
    MODEL_PATH_ARG="--model_path ${PRETRAINED_DIR}"
    OUTPUT_DIR="models/safe_gpt/trained/safe/${FRAG_NAME}/"
elif [ "$MODE" = "from_scratch" ]; then
    MODEL_PATH_ARG=""
    OUTPUT_DIR="models/safe_gpt/from_scratch/safe/${FRAG_NAME}/"
else
    echo "Unknown MODE: ${MODE} (use 'finetune' or 'from_scratch')" >&2
    exit 1
fi

safe-train \
${MODEL_PATH_ARG} \
--config ${PRETRAINED_DIR}/config.json \
--tokenizer ${PRETRAINED_DIR}/tokenizer.json \
--dataset data/safe/${FRAG_NAME}/normal \
--output_dir ${OUTPUT_DIR} \
--text_column full_safe \
--num_train_epochs 50 \
--learning_rate 1e-4 \
--warmup_steps 10000 \
--do_train \
--do_eval \
--eval_strategy steps \
--per_device_train_batch_size 32 \
--eval_steps 5000 \
--save_strategy steps \
--save_steps 5000 \
--save_total_limit 5 \
--load_best_model_at_end
```

## Notes / 確認事項

- `MODE="finetune"` を既定にする（元の意図＝ファインチューニングと解釈）。ゼロ学習時は手で `from_scratch` に変更。
- 空の `${MODEL_PATH_ARG}` は行継続と併用しても問題なく無視される（bash のトークン展開）。
- 本計画は `src/train_model/run_safe.sh` のみ変更。生成側（`gen_safe.py` / `gen_safe_denovo.py`）で
  `from_scratch/` モデルを読むためのパス変更は**含めない**（別途必要になれば追加提案）。
- 同一内容の `mol_gen_frags_oxygen/src/train_model/run_safe.sh` にも同じ修正が必要だが、
  本計画の対象は mol_gen_frags のみ（oxygen への適用は要望があれば別途）。
