import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__)).replace('/mol_gen_frags_copy/src', '')

if 0:
    # Check the content of curated dataset
    origin = pd.read_csv(f'{base_dir}/mol_gen_frags/data/curated/passed_filters_rdkit_canonical_smiles.tsv', sep='\t')
    copy   = pd.read_csv(f'{base_dir}/mol_gen_frags_copy/data/curated/passed_filters_rdkit_canonical_smiles.tsv', sep='\t')
    origin_chembl_id = set(origin['chembl_id'])
    copy_chembl_id   = set(copy['chembl_id'])
    if origin_chembl_id == copy_chembl_id:
        print('The content of the dataset is the same')
    else:
        print('The content of the dataset is different')

if 0:
    # Check the content of rffmg full dataset
    for method in ['brics', 'rc_cms']:
        origin = pd.read_csv(f'{base_dir}/mol_gen_frags/data/dummy/{method}/full_dataset.csv', index_col=0)
        copy   = pd.read_csv(f'{base_dir}/mol_gen_frags_copy/data/rffmg/{method}/full_dataset.csv', index_col=0)
        origin_f_frags = set(origin['full_fragments'])
        origin_p_frags = set(origin['pass_fragments'])
        copy_f_frags   = set(copy['full_fragments'])
        copy_p_frags   = set(copy['pass_fragments'])
        if (origin.shape==copy.shape) and (origin_f_frags == copy_f_frags) and (origin_p_frags == copy_p_frags):
            print(f'{method}, rffmg, The content of the dataset is the same')
        else:
            print(f'{method}, rffmg, The content of the dataset is different')

if 0:
    # Check the content of safe full dataset
    for method in ['brics', 'rc_cms']:
        origin = pd.read_csv(f'{base_dir}/mol_gen_frags/data/safe/{method}/safe_smiles.csv', index_col=0)
        copy   = pd.read_csv(f'{base_dir}/mol_gen_frags_copy/data/safe/{method}/safe_smiles.csv', index_col=0)
        origin_f_safe = set(origin['full_safe'])
        origin_p_safe = set(origin['pass_safe'])
        copy_f_safe   = set(copy['full_safe'])
        copy_p_safe   = set(copy['pass_safe'])
        if (origin.shape==copy.shape) and (origin_f_safe == copy_f_safe) and (origin_p_safe == copy_p_safe):
            print(f'{method}, safe, The content of the dataset is the same')
        else:
            print(f'{method}, safe, The content of the dataset is different')


if 0:
    # unique_frags
    for method in ['brics', 'rc_cms']:

        origin = pd.read_csv(f'{base_dir}/mol_gen_frags/data/rffmg/{method}/unique_frags.csv', index_col=0)
        copy   = pd.read_csv(f'{base_dir}/mol_gen_frags_copy/data/rffmg/{method}/unique_frags.csv', index_col=0)
        
        if origin.shape==copy.shape:
            print(f'{method}, attach_point_num, frag_num_{frag_num}, The content of the dataset is the same')
        else:
            print(f'{method}, attach_point_num, frag_num_{frag_num}, The content of the dataset is different')
if 1:
    # Check the content of rffmg datasets
    for method in ['brics', 'rc_cms']:
        
        for aug_name in ['attach_point_num']: # ['normal', 'robustness', 'attach_point_num', 'dup_frags', 'frag_num']:
            
            for name in ['train', 'val', 'test']:
                
                if not os.path.exists(f'{base_dir}/mol_gen_frags/data/rffmg/{method}/{aug_name}/{name}.source') and not os.path.exists(f'{base_dir}/mol_gen_frags_copy/data/rffmg/{method}/{aug_name}/{name}.source'):
                    continue

                origin_source = pd.read_csv(f'{base_dir}/mol_gen_frags/data/rffmg/{method}/{aug_name}/{name}.source', sep='\t', names=['smiles'])
                origin_target = pd.read_csv(f'{base_dir}/mol_gen_frags/data/rffmg/{method}/{aug_name}/{name}.target', sep='\t', names=['smiles'])
                copy_source = pd.read_csv(f'{base_dir}/mol_gen_frags_copy/data/rffmg/{method}/{aug_name}/{name}.source', sep='\t', names=['smiles'])
                copy_target = pd.read_csv(f'{base_dir}/mol_gen_frags_copy/data/rffmg/{method}/{aug_name}/{name}.target', sep='\t', names=['smiles'])
                
                if (origin_source['smiles'].to_list()==copy_source['smiles'].to_list()) and (origin_target['smiles'].to_list()==copy_target['smiles'].to_list()):
                    print(f'{method}, {aug_name}, {name}, The content and order of the dataset is the same')

                elif (sorted(origin_source['smiles'].to_list())==sorted(copy_source['smiles'].to_list())) and (sorted(origin_target['smiles'].to_list())==sorted(copy_target['smiles'].to_list())):
                    print(f'{method}, {aug_name}, {name}, The content of the dataset is the same but the order is different')

                else:
                    print(f'{method}, {aug_name}, {name}, The content and order of the dataset is different')

        



                

        


