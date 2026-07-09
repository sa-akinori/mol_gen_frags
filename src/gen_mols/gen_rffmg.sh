#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="t5chem" # "t5chem" or "gpt"
FRAG_NAME="brics" # "brics" or "rc_cms"
MODEL_VER="finetuning" # "finetuning", "from_scratch", "pretrained"
ADDITIONAL_PATH="normal" # normal, dup_frags, frag_num, frag_order, attach_point_num

python ${SCRIPT_DIR}/gen_rffmg.py --model_name ${MODEL_NAME} --frag_method ${FRAG_NAME} --model_ver ${MODEL_VER} --additional_path ${ADDITIONAL_PATH} --n_samples 50 --num_beams 50 --batch_size 24 --random_seed 42
