#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# リポジトリルートに移動（どこから実行しても相対パスが解決できるようにする）
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_promptsmiles

FRAG_NAME="brics"          # "brics" or "rc_cms"
MODE="finetuning"          # "finetuning" or "from_scratch"

if [ "$MODE" != "finetuning" ] && [ "$MODE" != "from_scratch" ]; then
    echo "Unknown MODE: ${MODE} (use 'finetuning' or 'from_scratch')" >&2
    exit 1
fi

# wandb: ログの保存先を model/mode/slice ごとに分ける（ローカルで識別するため）
export WANDB_MODE=offline
export WANDB_DIR="wandb/promptsmiles/gpt/${MODE}/${FRAG_NAME}"
mkdir -p "${WANDB_DIR}"

# PromptSMILES の prior は素の SMILES で学習した無条件言語モデル。
# データ水増しは行わない（1分子1系列）。表記のみランダム化する場合は --randomize_smiles を付ける。
python src/train_model/train_promptsmiles.py \
    --frag_method "${FRAG_NAME}" \
    --mode "${MODE}"
