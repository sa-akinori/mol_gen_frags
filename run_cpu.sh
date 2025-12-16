#!/bin/bash
#PBS -q cpuq_80
#PBS -N mk_dataset
#PBS -l select=1:ncpus=20
cd $PBS_O_WORKDIR
~/miniconda3/envs/safe_copy/bin/python  ~/Research/mol_gen_frags_copy/src/make_datasets.py --frag_method brics