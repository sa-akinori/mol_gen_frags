#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# リポジトリルートに移動（どこから実行しても相対パスが解決できるようにする）
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

FRAG_NAME="brics"          # "brics" or "rc_cms"
MODE="finetuning"          # "finetuning" or "from_scratch"
MODEL_NAME="t5chem"        # "t5chem" or "gpt"

if [ "$MODE" != "finetuning" ] && [ "$MODE" != "from_scratch" ]; then
    echo "Unknown MODE: ${MODE} (use 'finetuning' or 'from_scratch')" >&2
    exit 1
fi

OUTPUT_DIR="models/rffmg/${MODEL_NAME}/${MODE}/${FRAG_NAME}"

# wandb: ログの保存先を repr/model/mode/slice ごとに分ける（ローカルで識別するため）
export WANDB_MODE=offline
export WANDB_DIR="wandb/rffmg/${MODEL_NAME}/${MODE}/${FRAG_NAME}"
mkdir -p "${WANDB_DIR}"

if [ "$MODEL_NAME" = "t5chem" ]; then
    if [ "$MODE" = "finetuning" ]; then
        MODEL_ARG="--pretrain models/rffmg/${MODEL_NAME}/pretrained"
    else
        MODEL_ARG="--tokenizer simple"
    fi
    t5chem train ${MODEL_ARG} --data_dir data/rffmg/${FRAG_NAME}/normal --output_dir ${OUTPUT_DIR} --task_type product --num_epoch 50

elif [ "$MODEL_NAME" = "gpt" ]; then
    # GPT2 (entropy/gpt2_zinc_87m) を素の transformers で学習。finetuning/from_scratch は train_gpt.py が内部で処理。
    python src/train_model/train_gpt.py \
        --frag_method "${FRAG_NAME}" \
        --mode "${MODE}"

else
    echo "Unknown MODEL_NAME: ${MODEL_NAME} (use 't5chem' or 'gpt')" >&2
    exit 1
fi
