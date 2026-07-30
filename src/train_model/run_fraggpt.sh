#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# リポジトリルートに移動（どこから実行しても相対パスが解決できるようにする）
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_fraggpt

FRAG_NAME="brics"          # "brics" or "rc_cms"
MODE="finetuning"          # "finetuning" or "from_scratch"

if [ "$MODE" != "finetuning" ] && [ "$MODE" != "from_scratch" ]; then
    echo "Unknown MODE: ${MODE} (use 'finetuning' or 'from_scratch')" >&2
    exit 1
fi

# wandb: ログの保存先を model/mode/slice ごとに分ける（ローカルで識別するため）
export WANDB_MODE=offline
export WANDB_DIR="wandb/fraggpt/gpt/${MODE}/${FRAG_NAME}"
mkdir -p "${WANDB_DIR}"

# FragGPT の prior は FU-SMILES（[i*] でペア付けした断片列）で学習した無条件言語モデル。
# 付番のランダム置換と断片順シャッフルはデフォルトで有効。無効にする場合は --no-augment を付ける。
python src/train_model/train_fraggpt.py \
    --frag_method "${FRAG_NAME}" \
    --mode "${MODE}"
