#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safe

FRAG_NAME="rc_cms" # "brics" or "rc_cms"

safe-train \
--config models/safe_gpt/pretrained/config.json \
--tokenizer models/safe_gpt/pretrained/tokenizer.json \
--dataset data/safe/${FRAG_NAME}/normal \
--output_dir models/safe_gpt/trained/safe/${FRAG_NAME}/ \
--text_column full_safe \
--num_train_epochs 50 \
--learning_rate 1e-4 \
--warmup_steps 10000 \
--do_train \
--do_eval \
--eval_strategy steps \
--per_device_train_batch_size 32 \
--eval_steps 5000 \
--save_strategy steps \
--save_steps 5000 \
--save_total_limit 5 \
--load_best_model_at_end