#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

FRAG_NAME="rc_cms" # "brics" or "rc_cms"
t5chem train --data_dir data/rffmg/${FRAG_NAME}/normal --output_dir models/t5chem/trained/rffmg/${FRAG_NAME} --pretrain models/t5chem/pretrained --task_type product --num_epoch 50