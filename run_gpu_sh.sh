#!/bin/bash
#PBS -q gpuq_RTX6000Ada
#PBS -N train_safe
#PBS -l select=1:ngpus=1
#PBS -V
cd $PBS_O_WORKDIR
bash src/train_model/run_safe.sh