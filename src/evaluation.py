import os
import pandas as pd
import itertools
from func.evaluation_func import *
from func.utility import *
import numpy as np
from glob import glob
import argparse

if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, choices=['t5chem', 'safe_gpt'],
                        help='Model name (default: t5chem)')
    parser.add_argument('--model_ver', type=str, default='trained', choices=['trained', 'pretrained', 'from_scratch'],
                        help='Phase name (default: trained)')
    parser.add_argument('--frag_method', type=str, default='rc_cms', choices=['rc_cms', 'brics'],
                        help='Fragmentation method (default: rc_cms)')
    parser.add_argument('--additional_path', type=str, default='normal', choices=['normal', 'dup_frags', 'frag_num', 'frag_order', 'attach_point_num'],
                        help='Additional path (default: normal)')
    args = parser.parse_args()
    
    # Setting
    model_name  = args.model_name
    str_name    = 'rffmg' if model_name=='t5chem' else 'safe'
    model_ver   = args.model_ver
    frag_method = args.frag_method
    additional_path = args.additional_path
    cpu_num = os.cpu_count()
    
    # Load train dataset
    if model_name == 'safe_gpt':
        tr_file_name  = f'{BASEPATH}/data/safe/{frag_method}/normal'
        testInputfile = None
        additional_path = 'normal'
        
    elif model_name == 't5chem':
        tr_file_name  = f'{BASEPATH}/data/rffmg/{frag_method}/normal/train.target'
        testInputfile = f'{BASEPATH}/data/rffmg/{frag_method}/{additional_path}/test.source'
        
    trsmiles = loadTrainSmiles(model_name, tr_file_name)
    
    # Calculate some basic physic property for training smiles
    if not os.path.isfile(f'{BASEPATH}/results/train_physic_property.csv'):
        trPhysicprop = calcPhysicProp(list(trsmiles), n_jobs=cpu_num-1)
        trPhysicprop_df = pd.DataFrame(trPhysicprop)
        os.makedirs(f'{BASEPATH}/results', exist_ok=True)
        trPhysicprop_df.to_csv(f'{BASEPATH}/results/train_physic_property.csv')
        
    
    outfd = f'{BASEPATH}/results/{model_name}/{model_ver}/{str_name}/{frag_method}/beam/{additional_path}'
    
    ## Calculate physical property
    # Define file path
    file_name = f'{BASEPATH}/results/{model_name}/{model_ver}/{str_name}/{frag_method}/beam/{additional_path}/predictions.csv'
    
    # Evaluation gen mols
    genmols = loadGenSmiles(model_name, file_name, testInputfile)
    stats, genmols = sc3_check_genmol_results(outfd=outfd, genmols=genmols, trsmiles=trsmiles, skipCreateExcel=False, algorithm_name=frag_method, n_chunks=5)
    stats.to_csv(f'{outfd}/stats.csv')
    
    # Calculate some basic physic property for training smiles
    gensmiles = list(set([smi for _, row in genmols.iterrows() for smi in row['novel_smi']]))
    genPhysicprop = calcPhysicProp(list(gensmiles), n_jobs=cpu_num-1)
    genPhysicprop_df = pd.DataFrame(genPhysicprop)
    genPhysicprop_df.to_csv(f'{outfd}/physic_property.csv')

    # Evaluation metrics only for fragments used in frag_order (to compare the performance between unshuffled and shuffled fragment orders)
    if additional_path == 'frag_order':
        outfd  = f'{BASEPATH}/results/{model_name}/{model_ver}/{str_name}/{frag_method}/beam'
        datafd = f'{BASEPATH}/data/{str_name}/{frag_method}/'
        no_shuffle_df = pd.read_csv(f'{outfd}/normal/curated_data.tsv', sep='\t', index_col=0)
        random_get_id = pickle_load(f'{datafd}/frag_order/random_get_ids.pkl')
        no_shuffle_df = no_shuffle_df.loc[random_get_id]
        
        stats = dict()
        stats['avg_validity']         = no_shuffle_df['validratio'].mean() 
        stats['std_validity']         = no_shuffle_df['validratio'].std() 
        stats['avg_validity_onfrags'] = no_shuffle_df['validfragratio'].mean() # unique fragments should be used and count should be reflected.
        stats['std_validity_onfrags'] = no_shuffle_df['validfragratio'].std()
        stats['avg_validity_onfrags_exH'] = no_shuffle_df['validfragratio_exH'].mean() # unique fragments should be used and count should be reflected.
        stats['std_validity_onfrags_exH'] = no_shuffle_df['validfragratio_exH'].std()
        stats['avg_uniqueness']       = no_shuffle_df['uniqueratio'].mean()
        stats['std_uniqueness']       = no_shuffle_df['uniqueratio'].std()
        stats['avg_novelty']          = no_shuffle_df['novelratio'].mean()
        stats['std_novelty']          = no_shuffle_df['novelratio'].std()
        # 
        stats['avg_tanimoto_sim']     = no_shuffle_df[no_shuffle_df['nnovel'] != 0]['tanimoto_sim'].mean()
        stats['std_tanimoto_sim']     = no_shuffle_df[no_shuffle_df['nnovel'] != 0]['tanimoto_sim'].std()
        stats['avg_tanimoto_sim_onfrags'] = no_shuffle_df[no_shuffle_df['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].mean()
        stats['std_tanimoto_sim_onfrags'] = no_shuffle_df[no_shuffle_df['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].std()
        
        stats_df = pd.Series(stats)
        stats_df.to_csv(f'{outfd}/frag_order/no_shuffle_stats.csv')
        
    # if 0: # Calculate js-divergence between train and test
    #     # Setting
    #     compared_files = [pd.read_csv(file, index_col=0) for file in [
    #         f'{BASEPATH}/results/train_physic_property.csv',
    #         f'{BASEPATH}/results/t5chem/trained/rffmg/{frag_method}/beam/normal/physic_property.csv',
    #         f'{BASEPATH}/results/safe_gpt/pretrained/safe/{frag_method}/beam/normal/physic_property.csv',
    #         f'{BASEPATH}/results/safe_gpt/trained/safe/{frag_method}/beam/normal/physic_property.csv']
    #         ]
    #     properties = ['MW', 'TPSA', 'LogP', 'QED']
    #     bin_sizes  = [1, 1, 0.1, 0.01]
    #     file_names = ['train', 't5chem_trained_rffmg', 'safe_pretrained_safe', 'safe_trained_safe']
        
    #     for prop_name, bin_size in zip(properties, bin_sizes):
            
    #         # Calculate js-divergence
    #         js_div = calculate_js_divergence_for_properties(prop_dfs=compared_files, file_names=file_names, prop_name=prop_name, bin_size=bin_size)
    #         os.makedirs(f'{BASEPATH}/results/js_divergence/physic_properties/{frag_method}/beam/normal/', exist_ok=True)
    #         js_div.to_csv(f'{BASEPATH}/results/js_divergence/physic_properties/{frag_method}/beam/normal/{prop_name}.csv')
