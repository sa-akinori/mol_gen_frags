#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FRAG_NAME="brics" # "brics" or "rc_cms"
MODEL_VER="trained" # "trained" or "pretrained"

python ${SCRIPT_DIR}/gen_safe.py --frag_method ${FRAG_NAME} --model_ver ${MODEL_VER} --n_samples 50 --num_beams 50 --random_seed 42