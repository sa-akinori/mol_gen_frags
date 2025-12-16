#!/bin/bash
#PBS -q gpuq_RTX6000Ada
#PBS -N gen_mols
#PBS -l select=1:ngpus=1
#PBS -V

set -euo pipefail
cd "$PBS_O_WORKDIR"

# conda 初期化
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

# デバッグ用に確認
echo "Python: $(which python)"
echo "t5chem: $(which t5chem)"

# 実行
python ~/Research/mol_gen_frags/src/gen_mols/gen_t5chem.py
