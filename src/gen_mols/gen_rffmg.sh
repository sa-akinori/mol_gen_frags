#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="t5chem" # "t5chem" or "gpt"
FRAG_NAME="brics" # "brics" or "rc_cms"
MODEL_VER="finetuning" # "finetuning", "from_scratch"
SAMPLING_NUM=10 # 5 or 10 (uses the data/rffmg/<frag>/<N>times_sampling slice)
ADDITIONAL_PATH="normal" # normal, dup_frags, frag_num, frag_order, attach_point_num

python ${SCRIPT_DIR}/gen_rffmg.py --model_name ${MODEL_NAME} --frag_method ${FRAG_NAME} --model_ver ${MODEL_VER} --sampling_num ${SAMPLING_NUM} --additional_path ${ADDITIONAL_PATH} --n_samples 50 --num_beams 50 --batch_size 24 --random_seed 42
