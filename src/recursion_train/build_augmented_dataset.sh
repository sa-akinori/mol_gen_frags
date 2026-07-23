#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FRAG_METHOD="brics" # "brics" or "rc_cms"
MODEL_NAME="t5chem"
MODEL_VER="trained"
SCENARIOS="frag_num dup_frags attach_point_num"
N_SELECT=5
OUT_NAME="augmented"
SEED=0

python ${SCRIPT_DIR}/build_augmented_dataset.py --frag_method ${FRAG_METHOD} --model_name ${MODEL_NAME} --model_ver ${MODEL_VER} --scenarios ${SCENARIOS} --n_select ${N_SELECT} --out_name ${OUT_NAME} --seed ${SEED}
