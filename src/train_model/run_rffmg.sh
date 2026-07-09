#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# リポジトリルートに移動（どこから実行しても相対パスが解決できるようにする）
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

FRAG_NAME="brics"          # "brics" or "rc_cms"
MODE="finetuning"          # "finetuning" or "from_scratch"
MODEL_NAME="t5chem"

PRETRAINED_DIR="models/rffmg/${MODEL_NAME}/pretrained"

if [ "$MODE" = "finetuning" ]; then
    MODEL_ARG="--pretrain ${PRETRAINED_DIR}"
    OUTPUT_DIR="models/rffmg/${MODEL_NAME}/finetuning/${FRAG_NAME}"
elif [ "$MODE" = "from_scratch" ]; then
    MODEL_ARG="--tokenizer simple"
    OUTPUT_DIR="models/rffmg/${MODEL_NAME}/from_scratch/${FRAG_NAME}"
else
    echo "Unknown MODE: ${MODE} (use 'finetuning' or 'from_scratch')" >&2
    exit 1
fi

# wandb: ログの保存先を repr/model/mode/slice ごとに分ける（ローカルで識別するため）
export WANDB_MODE=offline
export WANDB_DIR="wandb/rffmg/${MODEL_NAME}/${MODE}/${FRAG_NAME}"
mkdir -p "${WANDB_DIR}"

t5chem train ${MODEL_ARG} --data_dir data/rffmg/${FRAG_NAME}/normal --output_dir ${OUTPUT_DIR} --task_type product --num_epoch 50
