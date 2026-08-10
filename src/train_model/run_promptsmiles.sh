#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# リポジトリルートに移動（どこから実行しても相対パスが解決できるようにする）
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate promptsmiles

FRAG_NAME="brics"          # "brics" or "rc_cms"
MODE="finetuning"          # "finetuning" or "from_scratch"

# wandb: ログの保存先を model/mode/slice ごとに分ける（ローカルで識別するため）
export WANDB_MODE=offline
export WANDB_DIR="wandb/promptsmiles/gpt/${MODE}/${FRAG_NAME}"
mkdir -p "${WANDB_DIR}"

# PromptSMILES の prior は素の SMILES で学習した無条件言語モデル。
# データ水増しは行わない（1分子1系列）ので RFFMG・SAFE とデータ量は同条件。
# ただし推論時のプロンプトは非カノニカルなため、学習側も表記をランダム化する（--seed で再現可能）。
python src/train_model/train_promptsmiles.py \
    --frag_method "${FRAG_NAME}" \
    --mode "${MODE}" \
    --num_train_epochs 50 \
    --learning_rate 1e-4 \
    --warmup_steps 10000 \
    --eval_strategy steps \
    --eval_steps 5000 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 5 \
    --max_length 256 \
    --per_device_train_batch_size 32
