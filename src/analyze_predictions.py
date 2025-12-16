import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
from func.figure_func import *
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

def spec_cond_frags(
    frags:str
    ):
    if len(frags.split('.')) == 1:
        return True
    
    elif len(frags.split('.')) > 2:
        return False
    
    else:
        frags = frags.split('.')
        return all([True if frag.count('*')==1 else False for frag in frags])
        
if __name__ == "__main__":
    # Settings
    arc_name = 't5chem'
    str_name = 'safe' if arc_name=='safe_gpt' else 'rffmg' # safe, rffmg
    model_name   = 'trained' # pretrained, trained
    slice_method = 'brics' # brics, rc_cms
    gen_method   = 'beam'
    
    if 0:
        # Paths
        predictions_path = f'{BASEPATH}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/predictions.csv'
        properties_path = f'{BASEPATH}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/physic_property.csv'
        output_dir = f'{BASEPATH}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/analysis'
        
        # Extract min/max properties for all rows
        results_df = extract_minmax_properties(predictions_path, properties_path, output_dir)
        
        # Create visualizations
        create_scatter_plots(results_df, output_dir)
        create_distribution_plots(results_df, output_dir)
        
    if 0:
        # validratio, uniqueratio, validfragratio, novelratio, SAscores, tanimoto_sim
        curated_df = pd.read_csv(f'{BASEPATH}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/curated_data.tsv', sep='\t', index_col=0)
        
        # Calculate fragment statistics for all data
        curated_df['mean_SAscores'] = curated_df['SAscores'].apply(lambda x: sum(ast.literal_eval(x))/len(ast.literal_eval(x)) if len(ast.literal_eval(x)) else 0)
        curated_df['n_fragments'] = curated_df['fragment'].apply(lambda x: len(x.split('.')))
        curated_df['n_wildcards'] = curated_df['fragment'].apply(lambda x: x.count('*'))
        curated_df['fragment_size'] = curated_df['fragment'].apply(lambda x: len(x.replace('.', '').replace('*', '')))
        
        for y_axis in ['validratio', 'uniqueratio', 'validfragratio', 'novelratio', 'mean_SAscores', 'tanimoto_sim']:
            
            # Plot different combinations
            os.makedirs(f'{BASEPATH}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/n_fragments', exist_ok=True)
            os.makedirs(f'{BASEPATH}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/n_wildcards', exist_ok=True)
            os.makedirs(f'{BASEPATH}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/fragment_size', exist_ok=True)
            plot_fragment_validity(df=curated_df, x_axis='n_fragments', y_axis=y_axis, save_path=f'{BASEPATH}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/n_fragments/{y_axis}.png')
            plot_fragment_validity(df=curated_df, x_axis='n_wildcards', y_axis=y_axis, save_path=f'{BASEPATH}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/n_wildcards/{y_axis}.png')
            plot_fragment_validity(df=curated_df, x_axis='fragment_size', y_axis=y_axis, save_path=f'{BASEPATH}/figures/frag_feat_vs_prop/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/fragment_size/{y_axis}.png')
            
    if 0:
        # Difference in generation accuracy between cases that satisfy the condition and those that do not (Condition: Input is one fragment with multiple attachment points or two fragments with one attachment point each)
        curated_df = pd.read_csv(f'{BASEPATH}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/curated_data.tsv', sep='\t', index_col=0)
        spec_cond_bool  = [spec_cond_frags(f) for f in curated_df['fragment']]
        spec_cond_df    = curated_df[spec_cond_bool].reset_index(drop=True)
        no_spec_cond_df = curated_df[~np.array(spec_cond_bool)].reset_index(drop=True)
        print(curated_df.shape, spec_cond_df.shape, no_spec_cond_df.shape)
        
        # Summary of generation accuracy when the condition is satisfied
        spec_stats = dict()
        spec_stats['avg_validity']         = spec_cond_df['validratio'].mean() 
        spec_stats['std_validity']         = spec_cond_df['validratio'].std() 
        spec_stats['avg_validity_onfrags'] = spec_cond_df['validfragratio'].mean() # unique fragments should be used and count should be reflected.
        spec_stats['std_validity_onfrags'] = spec_cond_df['validfragratio'].std()
        spec_stats['avg_uniqueness']       = spec_cond_df['uniqueratio'].mean()
        spec_stats['std_uniqueness']       = spec_cond_df['uniqueratio'].std()
        spec_stats['avg_novelty']          = spec_cond_df['novelratio'].mean()
        spec_stats['std_novelty']          = spec_cond_df['novelratio'].std()
        # 
        spec_stats['avg_tanimoto_sim']     = spec_cond_df[spec_cond_df['nnovel'] != 0]['tanimoto_sim'].mean()
        spec_stats['std_tanimoto_sim']     = spec_cond_df[spec_cond_df['nnovel'] != 0]['tanimoto_sim'].std()
        # no_spec_stats['avg_tanimoto_sim_onfrags'] = spec_cond_df[spec_cond_df['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].mean()
        # no_spec_stats['std_tanimoto_sim_onfrags'] = spec_cond_df[spec_cond_df['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].std()
        spec_stats_df = pd.Series(spec_stats)
        spec_stats_df.to_csv(f'{BASEPATH}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/spec_stats.csv')
        
        # Summary of generation accuracy when the condition is not satisfied
        no_spec_stats = dict()
        no_spec_stats['avg_validity']         = no_spec_cond_df['validratio'].mean() 
        no_spec_stats['std_validity']         = no_spec_cond_df['validratio'].std() 
        no_spec_stats['avg_validity_onfrags'] = no_spec_cond_df['validfragratio'].mean() # unique fragments should be used and count should be reflected.
        no_spec_stats['std_validity_onfrags'] = no_spec_cond_df['validfragratio'].std()
        no_spec_stats['avg_uniqueness']       = no_spec_cond_df['uniqueratio'].mean()
        no_spec_stats['std_uniqueness']       = no_spec_cond_df['uniqueratio'].std()
        no_spec_stats['avg_novelty']          = no_spec_cond_df['novelratio'].mean()
        no_spec_stats['std_novelty']          = no_spec_cond_df['novelratio'].std()
        # 
        no_spec_stats['avg_tanimoto_sim']     = no_spec_cond_df[no_spec_cond_df['nnovel'] != 0]['tanimoto_sim'].mean()
        no_spec_stats['std_tanimoto_sim']     = no_spec_cond_df[no_spec_cond_df['nnovel'] != 0]['tanimoto_sim'].std()
        # no_spec_stats['avg_tanimoto_sim_onfrags'] = spec_cond_df[spec_cond_df['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].mean()
        # no_spec_stats['std_tanimoto_sim_onfrags'] = spec_cond_df[spec_cond_df['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].std()
        no_spec_stats_df = pd.Series(no_spec_stats)
        no_spec_stats_df.to_csv(f'{BASEPATH}/results/{arc_name}/{model_name}/{str_name}/{slice_method}/{gen_method}/normal/no_spec_stats.csv')
        
    if 1:
        const_name = 'sim_tr_dup_frags' # ['attach_point_num', 'dup_frags', 'frag_num', 'new_attach_point_num', 'new_dup_frags', 'new_frag_num', 'sim_tr_attach_point_num', 'sim_tr_dup_frags', 'sim_tr_frag_num']
        
        # Verification of why the generation accuracy is poor with respect to the number of fragments (Is the input fragment larger than the training data because it is randomly selected from unique fragments?)
        train_df   = pd.read_csv(f'{BASEPATH}/data/{str_name}/{slice_method}/normal/train.source', sep='\t', header=None, names=['smiles'])
        curated_df = pd.read_csv(f'{BASEPATH}/data/{str_name}/{slice_method}/{const_name}/test.source', sep='\t', header=None, names=['smiles'])
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
        os.makedirs(f'{BASEPATH}/figures/smiles_length/{str_name}/{slice_method}', exist_ok=True)
        plt.savefig(f'{BASEPATH}/figures/smiles_length/{str_name}/{slice_method}/{const_name}.png')
        