#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="gpt" # "t5chem" or "gpt"
FRAG_NAME="rc_cms" # "brics" or "rc_cms"
MODEL_VER="finetuning" # "finetuning", "from_scratch"
SAMPLING_NUM=5 # 5 or 10 (uses the data/rffmg/<frag>/<N>times_sampling slice)
ADDITIONAL_PATH="normal" # normal, dup_frags, frag_num, frag_order, attach_point_num

python ${SCRIPT_DIR}/gen_rffmg.py --model_name ${MODEL_NAME} --frag_method ${FRAG_NAME} --model_ver ${MODEL_VER} --sampling_num ${SAMPLING_NUM} --additional_path ${ADDITIONAL_PATH} --n_samples 50 --num_beams 50 --batch_size 8 --random_seed 42
