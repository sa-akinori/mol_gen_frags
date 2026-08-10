import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from func.figure_func import *
from func.utility import *
import seaborn as sns

def extract_minmax_properties(predictions_path, properties_path, output_dir):
    """
    Process all rows in predictions.csv to extract min/max properties for each row's predictions.
    """
    # Load predictions
    print("Loading predictions...")
    pred_df = pd.read_csv(predictions_path)
    
    # Load properties (in chunks for memory efficiency)
    print("Loading properties...")
    prop_chunks = []
    for chunk in pd.read_csv(properties_path, chunksize=100000):
        prop_chunks.append(chunk)
    prop_df = pd.concat(prop_chunks, ignore_index=True)
    
    # Properties to analyze
    properties = ['MW', 'TPSA', 'LogP', 'QED']
    
    # Store results
    results = []
    pred_cols = [col for col in pred_df.columns if col.startswith('prediction_')]
    
    print(f"Processing {len(pred_df)} rows...")
    for idx, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
        # Extract predictions from this row
        predictions = row[pred_cols].values
        predictions = [p for p in predictions if pd.notna(p) and p != '']
        
        if not predictions:
            continue
        
        # Find matching properties
        matched_props = prop_df[prop_df['SMILES'].isin(predictions)]
        
        if matched_props.empty:
            continue
        
        # Calculate min/max for each property
        row_result = {'row_index': idx, 'target': row.get('target', ''), 'rank': row.get('rank', np.nan)}
        
        for prop in properties:
            if prop in matched_props.columns:
                row_result[f'{prop}_min'] = matched_props[prop].min()
                row_result[f'{prop}_max'] = matched_props[prop].max()
                row_result[f'{prop}_mean'] = matched_props[prop].mean()
                row_result[f'{prop}_std'] = matched_props[prop].std()
        
        results.append(row_result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f'{output_dir}/minmax_properties.csv', index=False)
    print(f"Results saved to {output_dir}/minmax_properties.csv")
    
    return results_df

def create_scatter_plots(results_df, output_dir):
    """
    Create scatter plots for min vs max values of each property.
    """
    properties = ['MW', 'TPSA', 'LogP', 'QED']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, prop in enumerate(properties):
        ax = axes[i]
        
        min_col = f'{prop}_min'
        max_col = f'{prop}_max'
        
        if min_col in results_df.columns and max_col in results_df.columns:
            # Remove rows with NaN values
            data = results_df[[min_col, max_col]].dropna()
            
            # Create scatter plot
            ax.scatter(data[min_col], data[max_col], alpha=0.5, s=10)
            
            # Add diagonal line
            min_val = data[min_col].min()
            max_val = data[max_col].max()
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.3, label='y=x')
            
            # Labels and title
            ax.set_xlabel(f'{prop} Min')
            ax.set_ylabel(f'{prop} Max')
            ax.set_title(f'{prop}: Min vs Max across predictions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            correlation = data[min_col].corr(data[max_col])
            ax.text(0.05, 0.95, f'Corr: {correlation:.3f}', 
                   transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/minmax_scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Scatter plots saved to {output_dir}/minmax_scatter_plots.png")

def create_distribution_plots(results_df, output_dir):
    """
    Create distribution plots for the range (max - min) of each property.
    """
    properties = ['MW', 'TPSA', 'LogP', 'QED']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, prop in enumerate(properties):
        ax = axes[i]
        
        min_col = f'{prop}_min'
        max_col = f'{prop}_max'
        
        if min_col in results_df.columns and max_col in results_df.columns:
            # Calculate range
            data = results_df[[min_col, max_col]].dropna()
            range_values = data[max_col] - data[min_col]
            
            # Create histogram
            ax.hist(range_values, bins=50, alpha=0.7, edgecolor='black')
            
            # Labels and title
            ax.set_xlabel(f'{prop} Range (Max - Min)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{prop}: Distribution of prediction ranges')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_range = range_values.mean()
            median_range = range_values.median()
            ax.axvline(mean_range, color='red', linestyle='--', label=f'Mean: {mean_range:.2f}')
            ax.axvline(median_range, color='green', linestyle='--', label=f'Median: {median_range:.2f}')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/range_distribution_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Distribution plots saved to {output_dir}/range_distribution_plots.png")

def plot_fragment_validity(
    df:pd.DataFrame,
    x_axis:str,
    y_axis:str,
    save_path:str):
    """
    Plot fragment statistics vs validity ratio
    
    Args:
        df: DataFrame with fragment data
        x_axis: 'n_fragments' or 'n_wildcards' or 'fragment_size'
        save_path: Path to save figure (optional)
    """
    corr = df[x_axis].corr(df[y_axis])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_axis], df[y_axis], alpha=0.5, s=10)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{x_axis} vs {y_axis}, Correlation: {corr:.3f}')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    # Settings
    # The representation and the model are two independent axes, matching the layout
    # written by evaluation.py: results/{repr_name}/{model_name}/{model_ver}/...
    repr_name    = 'safe' # ['rffmg', 'safe', 'promptsmiles', 'fraggpt']
    model_name   = 'gpt' # ['t5chem', 'gpt']
    model_ver    = 'finetuning' # ['pretrained', 'finetuning', 'from_scratch']
    frag_method  = 'brics' # ['rc_cms', 'brics']
    gen_method   = 'beam' # ['beam', 'sampling']
    sampling_num = 5 # [5, 10]
    additional_path = 'normal' # ['normal', 'dup_frags', 'frag_num', 'frag_order', 'attach_point_num']
    # RFFMG keeps its data and results under a {N}times_sampling segment (the number of
    # fragmentation patterns per molecule); the other representations have no such segment.
    sampling_seg = f'{sampling_num}times_sampling/' if repr_name == 'rffmg' else ''
    result_dir   = f'{BASEPATH}/results/{repr_name}/{model_name}/{model_ver}/{frag_method}/{sampling_seg}{gen_method}/{additional_path}'
    # The figures path keeps the condition segment out, because it differs between blocks.
    path_prefix  = f'{repr_name}/{model_name}/{model_ver}/{frag_method}/{sampling_seg}{gen_method}'

    if 0:
        # Paths
        predictions_path = f'{result_dir}/predictions.csv'
        properties_path = f'{result_dir}/physic_property.csv'
        output_dir = f'{result_dir}/analysis'
        
        # Extract min/max properties for all rows
        results_df = extract_minmax_properties(predictions_path, properties_path, output_dir)
        
        # Create visualizations
        create_scatter_plots(results_df, output_dir)
        create_distribution_plots(results_df, output_dir)
        
    if 0:
        # validratio, uniqueratio, validfragratio, novelratio, SAscores, tanimoto_sim
        curated_df = pd.read_csv(f'{result_dir}/curated_data.tsv', sep='\t', index_col=0)

        # Calculate fragment statistics for all data
        curated_df['mean_SAscores'] = curated_df['SAscores'].apply(lambda x: sum(ast.literal_eval(x))/len(ast.literal_eval(x)) if len(ast.literal_eval(x)) else 0)
        curated_df['n_fragments']   = curated_df['fragment'].apply(lambda x: len(x.split('.')))
        curated_df['n_wildcards']   = curated_df['fragment'].apply(lambda x: x.count('*'))
        curated_df['fragment_size'] = curated_df['fragment'].apply(lambda x: len(x.replace('.', '').replace('*', '')))
        
        fig_dir = f'{BASEPATH}/figures/frag_feat_vs_prop/{path_prefix}/{additional_path}'
        for y_axis in ['validratio', 'uniqueratio', 'validfragratio', 'novelratio', 'mean_SAscores', 'tanimoto_sim']:

            # Plot different combinations
            os.makedirs(f'{fig_dir}/n_fragments', exist_ok=True)
            os.makedirs(f'{fig_dir}/n_wildcards', exist_ok=True)
            os.makedirs(f'{fig_dir}/fragment_size', exist_ok=True)
            plot_fragment_validity(df=curated_df, x_axis='n_fragments', y_axis=y_axis, save_path=f'{fig_dir}/n_fragments/{y_axis}.png')
            plot_fragment_validity(df=curated_df, x_axis='n_wildcards', y_axis=y_axis, save_path=f'{fig_dir}/n_wildcards/{y_axis}.png')
            plot_fragment_validity(df=curated_df, x_axis='fragment_size', y_axis=y_axis, save_path=f'{fig_dir}/fragment_size/{y_axis}.png')
            
    if 0:
        const_name = 'dup_frags' # ['attach_point_num', 'dup_frags', 'frag_num']
        
        # Verification of why the generation accuracy is poor with respect to the number of fragments (Is the input fragment larger than the training data because it is randomly selected from unique fragments?)
        data_dir   = f'{BASEPATH}/data/{repr_name}/{frag_method}/{sampling_seg}'
        train_df   = pd.read_csv(f'{data_dir}normal/train.source', sep='\t', header=None, names=['smiles'])
        curated_df = pd.read_csv(f'{data_dir}{const_name}/test.source', sep='\t', header=None, names=['smiles'])
        train_df['smi_length'] = train_df['smiles'].apply(len)
        train_df['n_frags']    = train_df['smiles'].apply(lambda smi: len(smi.split('.')))
        curated_df['smi_length'] = curated_df['smiles'].apply(len)
        curated_df['n_frags']    = curated_df['smiles'].apply(lambda smi: len(smi.split('.')))

        train_df['dataset'] = 'train'
        curated_df['dataset'] = 'curated'

        # Concatenate data
        df = pd.concat([train_df, curated_df])

        # Plot style settings
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='n_frags', y='smi_length', hue='dataset', width=0.6)

        plt.xlabel('Number of fragments (n_frags)', fontsize=12)
        plt.ylabel('SMILES length (smi_length)', fontsize=12)
        plt.title('Distribution of SMILES length by number of fragments', fontsize=14)
        plt.legend(title='Dataset', loc='upper left')
        plt.tight_layout()
        smiles_length_dir = f'{BASEPATH}/figures/smiles_length/{repr_name}/{frag_method}/{sampling_seg}'
        os.makedirs(smiles_length_dir, exist_ok=True)
        plt.savefig(f'{smiles_length_dir}{const_name}.png')
        