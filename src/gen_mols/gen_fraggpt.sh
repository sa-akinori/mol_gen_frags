#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# リポジトリルートに移動（どこから実行しても相対パスが解決できるようにする）
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
# 生成は datasets（SAFE test split の読み込み）に依存するため、env_fraggpt 以外で実行すると
# ModuleNotFoundError になる。環境の取り違えを防ぐためここで明示的に activate する。
conda activate env_fraggpt

FRAG_NAME="brics" # "brics" or "rc_cms"
MODEL_VER="finetuning" # "finetuning", "from_scratch"
GEN_METHOD="beam" # "sampling" (multinomial) or "beam" (beam search, as used by RFFMG and SAFE)

python src/gen_mols/gen_fraggpt.py --frag_method ${FRAG_NAME} --model_ver ${MODEL_VER} --gen_method ${GEN_METHOD} --n_samples 50 --max_length 256 --num_beams 50 --batch_size 24 --random_seed 42
