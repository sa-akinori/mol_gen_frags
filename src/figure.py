import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from func.utility import pickle_load
from func.figure_func import *
from functools import partial
from joblib import Parallel, delayed
from PIL import Image
from glob import glob
import ast
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter

xlims = {
    'MW': [0, 1000],
    'TPSA': [0, 400],
    'LogP': [-15, 15],
    'QED': [0, 1]
    }
    
calc_MW = lambda smi: Descriptors.MolWt(Chem.MolFromSmiles(smi))

def prop_min_max(batch_data, pred_col, prop_df, property_names):
    """Process a single row and return results."""
    results = []
    for idx, row in batch_data:
        
        preds = [p for p in ast.literal_eval(row[pred_col]) if pd.notna(p) and p != '']
        props = prop_df.query('SMILES in @preds')
        frags_mw = calc_MW(row['fragment'])
        
        if not props.empty:
            result = {'idx': idx}
            
            for prop_name in property_names:
                
                if prop_name == 'MW':
                    base_value = frags_mw
                    
                else:
                    base_value = 0
                
                result[f'{prop_name}_min'] = props[prop_name].min() - base_value
                result[f'{prop_name}_max'] = props[prop_name].max() - base_value
            results.append(pd.DataFrame(result.values(), index=result.keys()).T)
    return pd.concat(results)

def attach_points_analyze(
    fragment:str
    )->tuple[int, int]:
    frags = fragment.split('.')
    return max(f.count('*') for f in frags), len(frags) - 1

def dup_frags_analyze_train(
    fragment:str
    )->tuple[int, int]:
    frags = fragment.split('.')
    return max(Counter(frags).values()), 0

class dup_frags_analyze:

    def __init__(self, target_frags:List[str]):
        self.target_frags = target_frags
        
    def __call__(self, fragment:str)->tuple[int, int]:
        fragments = fragment.split('.')
        app_frags = [frag for frag in fragments if frag in self.target_frags]
        if len(set(app_frags)) != 1:
            raise ValueError('Miss')
        return len(app_frags), len(fragments) - len(app_frags)
    

def frag_num_analyze(
    fragment:str
    )->tuple[int, int]:
    return len(fragment.split('.')), 0


if __name__ == "__main__":
    
    # Setting
    fd = os.path.dirname(os.path.dirname(__file__))
    arc_name     = 't5chem' # ['t5chem', 'safe_gpt']
    str_name     = 'rffmg' if arc_name=='t5chem' else 'safe'
    model_name   = 'trained' # ['pretrained', 'trained']
    slice_method = 'brics' # ['rc_cms', 'brics']
    gen_method   = 'beam' # ['beam', 'random']
    property_names = ['MW', 'TPSA', 'LogP', 'QED'] # ['MW', 'TPSA', 'LogP', 'QED']
    train_data = False

    if 0:
        for property_name in property_names:
            
            if train_data:
                
                file_path = f'{fd}/results/train_physic_property.csv'
                output_dir = f'{fd}/figures/physic_property/train/'
                os.makedirs(output_dir, exist_ok=True)
                
            else:
                
                file_path = f'{fd}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/physic_property.csv'
                output_dir = f'{fd}/figures/physic_property/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal'
                os.makedirs(output_dir, exist_ok=True)
            
            # Read file in chunks to avoid memory issues
            df = pd.read_csv(file_path, index_col=0)
            
            # Create individual plot
            stats = plot_single_dataset_pdf(data=df[property_name], x_label=property_name, y_label='Probability density', y_axis_st='float', density=True, output_path=f'{output_dir}/{property_name}')
        
    if 0:
        # Setting
        pred_path = f'{fd}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/predictions.csv'
        prop_path = f'{fd}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/physic_property.csv'
        n_samples = 5
        
        # Load dataset
        pred_df = pd.read_csv(pred_path)
        pred_df = pred_df.sample(frac=1, random_state=0)
        prop_df = pd.read_csv(prop_path, index_col=0)
        pred_cols = [col for col in pred_df.columns if col.startswith('prediction_')]
        
        # ---- Get {n_samples} valid samples ----
        valid_samples = []
        for idx, row in pred_df.iterrows():
            preds = [p for p in row[pred_cols].dropna() if str(p).strip()]
            new_prop_df = prop_df.query('SMILES in @preds')
            if not new_prop_df.empty:
                valid_samples.append((idx, preds))
            if len(valid_samples) >= n_samples:
                break

        # ---- Plot ----
        for idx, preds in valid_samples:
            new_prop_df = prop_df.query('SMILES in @preds')
            output_dir = f'{fd}/figures/physic_property/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/individual/{idx}'
            os.makedirs(output_dir, exist_ok=True)

            for prop in property_names:
                plot_single_dataset_pdf(data=new_prop_df[prop], x_label=prop, y_label='Number of compounds', density=False, output_path=f'{output_dir}/{prop}')
            
    
    # Extract min/max properties and create scatter plots
    if 0:
        # Setting
        prop_path = f'{fd}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/physic_property.csv'
        cur_path  = f'{fd}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/curated_data.tsv'
        output_dir = f'{fd}/figures/physic_property/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/minmax'
        
        # Load dataset
        prop_df = pd.read_csv(prop_path, index_col=0)
        cur_df  = pd.read_csv(cur_path, sep='\t', index_col=0)
        
        # Create partial function with fixed arguments
        rows   = list(cur_df.iterrows())
        n_jobs = os.cpu_count() - 1
        batch_size = len(rows) // n_jobs + (1 if len(rows) % n_jobs else 0)
        batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]
        
        # Process in parallel with joblib
        results = Parallel(n_jobs=n_jobs)(
        delayed(prop_min_max)(batch_data=batch, pred_col='valid_smis_on_frags', prop_df=prop_df, property_names=property_names)
        for batch in tqdm(batches, desc="Processing batches"))
        
        # Filter out None values
        results_df = pd.concat(results).reset_index(drop=True)
        
        for prop_name in property_names:
            x_col = f'{prop_name}_min'
            y_col = f'{prop_name}_max'
            output_dir  = f'{fd}/figures/physic_property/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/minmax'
            os.makedirs(output_dir, exist_ok=True)
            create_scatter_plot(df=results_df, x_col=x_col, y_col=y_col, output_path=f'{output_dir}/{prop_name}.png', add_diagonal=False)
    
    if 0: # For test dataset
        # Setting
        const_name = 'frag_num' # ['attach_point_num', 'frag_num']
        
        if const_name == 'attach_point_num':
            analyze_func = attach_points_analyze
            col_name = f'max_{const_name}'
            hue_name = 'add_frags_num'
        
        elif const_name == 'frag_num':
            analyze_func = frag_num_analyze
            col_name = f'frag_num'
            hue_name = None
        
        # Load data sets
        curated_df = pd.read_csv(f'{fd}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/curated_data.tsv', sep='\t', index_col=0)
        curated_df[[col_name, 'add_frags_num']] = curated_df['fragment'].apply(lambda x: pd.Series(analyze_func(x)))
        
        # Create individual box plots for each metric
        metrics = [('validratio', 'Valid ratio'), ('uniqueratio', 'Unique ratio'), ('novelratio', 'Novel ratio'), ('validfragratio', 'Validfrag ratio')]
        for metric, metric_name in metrics:
            x_lim, y_lim = [min(curated_df[col_name]) - 0.5, max(curated_df[col_name]) + 0.5], [-0.05, 1.05]
            save_path = f'{fd}/figures/constraints/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/{metric}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            create_boxplot(df=curated_df, x_col=col_name, y_col=metric, x_name=col_name, y_name=metric_name, x_lim=x_lim, y_lim=y_lim, hue=hue_name, save_path=save_path)
            
    if 1:
        # For constrained data set
        # Setting
        const_name = 'attach_point_num' # ['attach_point_num', 'dup_frags', 'frag_num']
        
        if 'attach_point_num' in const_name:
            analyze_func = attach_points_analyze
            col_name = f'max_{const_name}'
            hue_name = 'add_frags_num'
            
        elif 'dup_frags' in const_name:
            target_frags = pickle_load(f'{fd}/data/dummy/{slice_method}/{const_name}/target_frags.pkl')
            analyze_func = dup_frags_analyze(target_frags)
            col_name = f'dup_frags'
            hue_name = 'add_frags_num' 
        
        elif 'frag_num' in const_name:
            analyze_func = frag_num_analyze
            col_name = f'frag_num'
            hue_name = None
        
        # Load data sets
        curated_df = pd.read_csv(f'{fd}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/{const_name}/curated_data.tsv', sep='\t', index_col=0)
        curated_df[[col_name, 'add_frags_num']] = curated_df['fragment'].apply(lambda x: pd.Series(analyze_func(x)))
        
        # Create individual box plots for each metric
        metrics = [('validratio', 'Valid ratio'), ('uniqueratio', 'Unique ratio'), ('novelratio', 'Novel ratio'), ('validfragratio', 'Validfrag ratio')]
        for metric, metric_name in metrics:
            x_lim, y_lim = [min(curated_df[col_name]) - 0.5, max(curated_df[col_name]) + 0.5], [-0.05, 1.05]
            save_path = f'{fd}/figures/constraints/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/{const_name}/{metric}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            create_boxplot(df=curated_df, x_col=col_name, y_col=metric, x_name=col_name, y_name=metric_name, x_lim=x_lim, y_lim=y_lim, hue=hue_name, save_path=save_path)
            
    if 0:
        # Distribution of train about 'const_name'
        # Setting
        const_name = 'new_frag_num' # ['attach_point_num', 'dup_frags', 'frag_num', 'new_attach_point_num', 'new_dup_frags', 'new_frag_num']
        
        if 'attach_point_num' in const_name:
            analyze_func = attach_points_analyze
            col_name = f'max_{const_name}'
            
        elif 'dup_frags' in const_name:
            analyze_func = dup_frags_analyze_train
            col_name = f'dup_frags'
        
        elif 'frag_num' in const_name:
            analyze_func = frag_num_analyze
            col_name = f'frag_num'
        
        # Load data sets
        train_df = pd.read_csv(f'{fd}/data/{str_name}/{slice_method}/normal/train.source', sep='\t', names=['smiles'])
        train_df[[col_name, 'add_frags_num']] = train_df['smiles'].apply(lambda x: pd.Series(analyze_func(x)))
        const_count = pd.DataFrame(train_df[col_name].value_counts())
        const_count.to_csv(f'{fd}/data/{str_name}/{slice_method}/{const_name}/count.csv')
        x_lim, y_lim = [min(train_df[col_name]) - 0.5, max(train_df[col_name]) + 0.5], [-0.05, train_df.shape[0]+10]
        save_path = f'{fd}/figures/constraints/train/{str_name}/{slice_method}/{const_name}/train.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_single_dataset_pdf(data=train_df[col_name], x_label=col_name, y_label='Number of compounds', y_axis_st='float', density=False, output_path=save_path)
        
        
    if 0:
        # validratio, uniqueratio, validfragratio, novelratio, SAscores, tanimoto_sim
        curated_df = pd.read_csv(f'{fd}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/curated_data.tsv', sep='\t', index_col=0)
        
        # Calculate fragment statistics for all data
        curated_df['n_fragments'] = curated_df['fragment'].apply(lambda x: len(x.split('.')))
        curated_df['n_wildcards'] = curated_df['fragment'].apply(lambda x: x.count('*'))
        curated_df['n_dup_frags'] = curated_df['fragment'].apply(dup_frags_analyze_train)
        
        for y_col in ['validratio', 'uniqueratio', 'validfragratio', 'novelratio', 'tanimoto_sim']:
            
            # Plot different combinations
            os.makedirs(f'{fd}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/n_fragments', exist_ok=True)
            os.makedirs(f'{fd}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/n_wildcards', exist_ok=True)
            os.makedirs(f'{fd}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/n_dup_frags', exist_ok=True)
            create_scatter_plot(df=curated_df, x_col='n_fragments', y_col=y_col, output_path=f'{fd}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/n_fragments/{y_col}.png', show_corr=False, add_diagonal=False)
            create_scatter_plot(df=curated_df, x_col='n_wildcards', y_col=y_col, output_path=f'{fd}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/n_wildcards/{y_col}.png', show_corr=False, add_diagonal=False)
            create_scatter_plot(df=curated_df, x_col='n_wildcards', y_col=y_col, output_path=f'{fd}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/n_dup_frags/{y_col}.png', show_corr=False, add_diagonal=False)