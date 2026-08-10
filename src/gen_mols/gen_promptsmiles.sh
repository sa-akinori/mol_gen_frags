#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate promptsmiles

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FRAG_NAME="brics" # "brics" or "rc_cms"
MODEL_VER="finetuning" # "finetuning", "from_scratch"
GEN_METHOD="beam" # "sampling" (multinomial) or "beam" (beam search, as used by RFFMG and SAFE)

python ${SCRIPT_DIR}/gen_promptsmiles.py --frag_method ${FRAG_NAME} --model_ver ${MODEL_VER} --gen_method ${GEN_METHOD} --n_samples 50 --max_length 256 --num_beams 50 --random_seed 42
