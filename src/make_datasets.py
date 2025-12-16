import os
import pickle
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from rdkit import Chem
import itertools
from tqdm import tqdm
from collections import Counter
from func.fragmentation import GetNHA, RemoveAtomIsotope
from func.utility import pickle_save, canonical_smiles
from concurrent.futures import ProcessPoolExecutor
import argparse

def save_file(
    target,
    save_file
    ):
    with open(save_file, "w", newline="\n", encoding="utf-8") as f:
        f.write(target)
        
def load_file(
    file_name:str
    ):
    with open(file_name) as f:
        data = [s.rstrip() for s in f.readlines()]
        
    return data
    
def process_reaction(reaction):
    reaction = reaction.split(">>")
    return (canonical_smiles(reaction[0]), canonical_smiles(reaction[1]))

def setrffmgAtoms(
    smi:str
    ):
    mol = Chem.MolFromSmiles(smi)
    mol = RemoveAtomIsotope(mol)
    new_smi = Chem.MolToSmiles(mol)
    
    return new_smi

unique_f_num = lambda frags: len(set(frags.split('.')))
countAtttachPoint = lambda smiles: smiles.count('*')

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--frag_method', type=str, choices=['brics', 'rc_cms'], required=True, help='fragmentation method')
    args = parser.parse_args()
    
    # Setting
    fd = os.path.dirname(__file__).replace('/src', '') + '/data'
    frag_method = args.frag_method
    
    # Create random split dataset
    rffmg_frags = pd.read_csv(f'{fd}/rffmg/{frag_method}/full_dataset.csv', index_col=0)
    rffmg_frags['smiles'] = rffmg_frags['sentence'].apply(lambda s : s.split('>>')[-1])
    safe_frags = pd.read_csv(f'{fd}/safe/{frag_method}/safe_smiles.csv', index_col=0)
    
    unique_smiles = list(rffmg_frags['smiles'].unique())
    random.seed(0)
    random.shuffle(unique_smiles)
    tr_ratio, val_ratio, te_ratio = 0.95, 0.025, 0.025
    tr_smiles, val_smiles, te_smiles = unique_smiles[:int(len(unique_smiles)*tr_ratio)], unique_smiles[int(len(unique_smiles)*tr_ratio):int(len(unique_smiles)*(tr_ratio + val_ratio))], unique_smiles[int(len(unique_smiles)*(tr_ratio + val_ratio)):]
    
    # rffmg
    rffmg_tr  = rffmg_frags.query('smiles in @tr_smiles').reset_index(drop=True)
    rffmg_val = rffmg_frags.query('smiles in @val_smiles').reset_index(drop=True)
    rffmg_te  = rffmg_frags.query('smiles in @te_smiles').reset_index(drop=True)
    
    for data, name in zip([rffmg_tr, rffmg_val, rffmg_te], ["train", "val", "test"]):
        reactions = [reaction.split('>>') for reaction in data['sentence']]
        source = "\n".join([reaction[0] for reaction in reactions]) + "\n"
        target = "\n".join([reaction[1] for reaction in reactions]) + "\n"
        os.makedirs(f'{fd}/rffmg/{frag_method}/normal/', exist_ok=True)
        save_file(source, f'{fd}/rffmg/{frag_method}/normal/{name}.source')
        save_file(target, f'{fd}/rffmg/{frag_method}/normal/{name}.target')
    
    # rffmg, for debug
    for data, name in zip([rffmg_tr.head(10000), rffmg_val.head(10000), rffmg_te.head(10000)], ["train", "val", "test"]):
        add_path  = '' if name != 'test' else 'normal'
        reactions = [reaction.split('>>') for reaction in data['sentence']]
        source = "\n".join([reaction[0] for reaction in reactions]) + "\n"
        target = "\n".join([reaction[1] for reaction in reactions]) + "\n"
        os.makedirs(f'{fd}/rffmg/{frag_method}/normal/debug', exist_ok=True)
        save_file(source, f'{fd}/rffmg/{frag_method}/normal/debug/{name}.source')
        save_file(target, f'{fd}/rffmg/{frag_method}/normal/debug/{name}.target')
    
    # safe
    safe_tr  = safe_frags.query('smiles in @tr_smiles').drop_duplicates(subset='full_safe').reset_index(drop=True)
    safe_val = safe_frags.query('smiles in @val_smiles').drop_duplicates(subset='full_safe').reset_index(drop=True)
    safe_te  = safe_frags.query('smiles in @te_smiles').reset_index(drop=True)
    
    train_safe_ds = Dataset.from_dict({col:safe_tr[col].tolist() for col in safe_tr.columns})
    valid_safe_ds = Dataset.from_dict({col:safe_val[col].tolist() for col in safe_val.columns})
    test_safe_ds  = Dataset.from_dict({col:safe_te[col].tolist() for col in safe_te.columns})
    
    os_dataset = DatasetDict({
        "train": train_safe_ds,
        "validation": valid_safe_ds,
        "test": test_safe_ds,
    })
    
    os.makedirs(f'{fd}/safe/{frag_method}/normal/', exist_ok=True)
    os_dataset.save_to_disk(f'{fd}/safe/{frag_method}/normal')
    
    # safe, for debug
    debug_train_safe_ds = Dataset.from_dict({col:safe_tr.head(10000)[col].tolist() for col in safe_tr.columns})
    debug_valid_safe_ds = Dataset.from_dict({col:safe_val.head(10000)[col].tolist() for col in safe_val.columns})
    debug_test_safe_ds  = Dataset.from_dict({col:safe_te.head(10000)[col].tolist() for col in safe_te.columns})
    
    debug_os_dataset = DatasetDict({
        "train": debug_train_safe_ds,
        "validation": debug_valid_safe_ds,
        "test": debug_test_safe_ds,
    })
    os.makedirs(f'{fd}/safe/{frag_method}/normal/debug', exist_ok=True)
    debug_os_dataset.save_to_disk(f'{fd}/safe/{frag_method}/normal/debug')

    # Creation of datasets to evaluate the limitations of molecular generation using RFFMG
    rffmg_frags = pd.read_csv(f'{fd}/rffmg/{frag_method}/full_dataset.csv', index_col=0)
    all_frags   = [frag for sentence in rffmg_frags['sentence'] for frag in sentence.split('>>')[0].split('.') if frag != 'O=c1/c=c\\c(=O)-n2-c3ccccc3-n-1-c1ccccc1-2']
    frags_df    = pd.DataFrame(Counter(all_frags).items(), columns=['fragment', 'count'])
    frags_df['frag_NHA'] = frags_df['fragment'].apply(lambda smi: GetNHA(Chem.MolFromSmiles(smi)))
    frags_df['frag_NAP'] = frags_df['fragment'].apply(countAtttachPoint)
    os.makedirs(f'{fd}/rffmg/{frag_method}', exist_ok=True)
    frags_df.to_csv(f'{fd}/rffmg/{frag_method}/unique_frags.csv')

    # Dataset for robustness to the order of fragments in fragment sets.
    rffmg_source_set = load_file(f'{fd}/rffmg/{frag_method}/normal/test.source')
    rffmg_target_set = load_file(f'{fd}/rffmg/{frag_method}/normal/test.target')
    random_get_id = [i for i, frag in enumerate(rffmg_source_set) if unique_f_num(frag) >= 3]
    random.seed(0)
    random_get_id = random.sample(random_get_id, 10000)
    
    reactions  = [[".".join(new_frag), target] for i, (frag, target) in enumerate(zip(rffmg_source_set, rffmg_target_set)) if i in random_get_id for new_frag in random.sample(list(itertools.permutations(frag.split('.'))), 5)]
    
    ext_source = [frag for i, frag in enumerate(rffmg_source_set) if i in random_get_id]
    new_source = "\n".join([reaction[0] for reaction in reactions]) + "\n"
    new_target = "\n".join([reaction[1] for reaction in reactions]) + "\n"
    os.makedirs(f'{fd}/rffmg/{frag_method}/frag_order/', exist_ok=True)
    pickle_save(f'{fd}/rffmg/{frag_method}/frag_order/random_get_ids.pkl', random_get_id)
    pickle_save(f'{fd}/rffmg/{frag_method}/frag_order/extracted_source.pkl', ext_source)
    save_file(new_source, f'{fd}/rffmg/{frag_method}/frag_order/test.source')
    save_file(new_target, f'{fd}/rffmg/{frag_method}/frag_order/test.target')
    
    # Dataset for robustness to the number of fragments in fragment sets.
    # Load unique fragments.
    unique_frags_df = pd.read_csv(f'{fd}/rffmg/{frag_method}/unique_frags.csv', index_col=0)
    cand_frags_set  = [frag for frag, count in zip(unique_frags_df['fragment'], unique_frags_df['count']) for _ in range(count)]
    
    # Load training dataset.
    train_source_set = load_file(f'{fd}/rffmg/{frag_method}/normal/train.source')
    train_source_set = [canonical_smiles(source) for source in tqdm(train_source_set) if source != 'O=c1/c=c\\c(=O)-n2-c3ccccc3-n-1-c1ccccc1-2']
    
    # Generate new reactions
    val_frag_sets = list()
    for frag_num in range(1, 11):
        
        random.seed(frag_num)
        prev_frag_sets = list()
        
        while len(prev_frag_sets) < 1000:
            frag_set     = '.'.join(random.sample(cand_frags_set, frag_num))
            can_frag_set = canonical_smiles(frag_set)
            
            if can_frag_set not in train_source_set and can_frag_set not in prev_frag_sets:
                prev_frag_sets.append(can_frag_set)
                val_frag_sets.append(frag_set)
    
    new_source = "\n".join(val_frag_sets) + "\n"
    new_target = "\n".join(['' for _ in val_frag_sets]) + "\n" # In T5Chem, some kind of target file is required.
    os.makedirs(f'{fd}/rffmg/{frag_method}/frag_num/', exist_ok=True)
    save_file(new_source, f'{fd}/rffmg/{frag_method}/frag_num/test.source')
    save_file(new_target, f'{fd}/rffmg/{frag_method}/frag_num/test.target')
    
    # Dataset for robustness to the number of duplicated fragments in fragment sets
    # Load unique fragments
    unique_frags_df = pd.read_csv(f'{fd}/rffmg/{frag_method}/unique_frags.csv', index_col=0)
    unique_frags  = list(unique_frags_df['fragment'])
    frags_NHA     = list(unique_frags_df['frag_NHA'])
    
    # Load train dataset
    train_source_set = load_file(f'{fd}/rffmg/{frag_method}/normal/train.source')
    train_source_set = [canonical_smiles(source) for source in tqdm(train_source_set) if source != 'O=c1/c=c\\c(=O)-n2-c3ccccc3-n-1-c1ccccc1-2']
    
    # Get fragments for duplicated operations
    target_frag_set  = list()
    for heavy_num in range(5, 21):
        target_frags = [unique_frag for unique_frag, frag_NHA in zip(unique_frags, frags_NHA) if frag_NHA == heavy_num]
        random.seed(heavy_num)
        target_frags = random.sample(target_frags, min(len(target_frags), 30))
        target_frag_set.extend(target_frags)
        
    # Get fragments for additing to target_frag
    cand_frags_df  = unique_frags_df.query('fragment not in @target_frag_set').reset_index(drop=True)
    cand_frags_set = [frag for frag, count in zip(cand_frags_df['fragment'], cand_frags_df['count']) for _ in range(count)]
    
    val_frag_sets, prev_frag_sets = [['.'.join([target_frags] * dup_num)for dup_num in range(2, 6) for target_frags in target_frag_set]], list()
    for frag_num in range(1, 4):
        random.seed(frag_num)
        
        for dup_num in range(2, 6):
            comb_frag_sets = list()
            
            # Get 20 random combinations that are not in the training data
            while len(comb_frag_sets) < 20:
                add_frags    = random.sample(cand_frags_set, frag_num)
                add_frags_df = pd.DataFrame(Counter(add_frags).items(), columns=['fragment', 'count'])
                
                if add_frags_df.query('count > @dup_num').shape[0]:
                    add_frags_df['count'] = add_frags_df['count'].clip(upper=dup_num)
                    
                    while (new_frag_num := frag_num - sum(add_frags_df['count'])) != 0:
                        add_frags    = [frag for frag, count in zip(add_frags_df['fragment'], add_frags_df['count']) for _ in range(count)]
                        new_add_frag = random.sample(cand_frags_set, new_frag_num)
                        add_frags.extend(new_add_frag)
                        add_frags_df = pd.DataFrame(Counter(add_frags).items(), columns=['fragment', 'count'])
                        add_frags_df['count'] = add_frags_df['count'].clip(upper=dup_num)
                        
                comb_frag_set     = ['.'.join([target_frags] * dup_num + add_frags) for target_frags in target_frag_set]
                can_comb_frag_set = [canonical_smiles(frag_set) for frag_set in comb_frag_set]
                can_add_frags     = canonical_smiles('.'.join(add_frags))
                
                if not set(can_comb_frag_set) & set(train_source_set) and can_add_frags not in prev_frag_sets:
                    comb_frag_sets.append(comb_frag_set)
                    prev_frag_sets.append(can_add_frags)
                    
            val_frag_sets.extend(comb_frag_sets)
        
    val_frag_sets = list(itertools.chain.from_iterable(val_frag_sets))
    new_source = "\n".join(val_frag_sets) + "\n"
    new_target = "\n".join(['' for _ in val_frag_sets]) + "\n" # In T5Chem, some kind of target file is required.
    os.makedirs(f'{fd}/rffmg/{frag_method}/dup_frags/', exist_ok=True)
    pickle_save(f'{fd}/rffmg/{frag_method}/dup_frags/target_frags.pkl', target_frag_set)
    save_file(new_source, f'{fd}/rffmg/{frag_method}/dup_frags/test.source')
    save_file(new_target, f'{fd}/rffmg/{frag_method}/dup_frags/test.target')
    
    # Dataset for robustness to the maximum number of attachment points in fragment sets
    # Load unique fragments
    unique_frags_df = pd.read_csv(f'{fd}/rffmg/{frag_method}/unique_frags.csv', index_col=0)
    
    # Load training data set
    train_source_set = load_file(f'{fd}/rffmg/{frag_method}/normal/train.source')
    train_source_set = [canonical_smiles(source) for source in tqdm(train_source_set) if source != 'O=c1/c=c\\c(=O)-n2-c3ccccc3-n-1-c1ccccc1-2']
    
    # Generate new reactions
    val_frag_sets = list()
    for max_att_point in range(2, 6): 
        
        cond_frags_df  = unique_frags_df[unique_frags_df['frag_NAP'] < max_att_point].reset_index(drop=True)
        cond_frags_set = [frag for frag, count in zip(cond_frags_df['fragment'], cond_frags_df['count']) for _ in range(count)]
        target_frags   = list(unique_frags_df[unique_frags_df['frag_NAP'] == max_att_point].sample(n=100, replace=True, random_state=max_att_point)['fragment'])
        val_frag_sets.append(target_frags)
        
        for frag_num in range(1, 4):
            
            frag_sets, add_frag_sets = list(), list()
            random.seed(frag_num)
            while len(frag_sets) < 100:
                add_frags     = random.sample(cond_frags_set, frag_num)
                can_add_frags = canonical_smiles('.'.join(add_frags))
                frag_set     = ['.'.join(random.sample(add_frags + [target_frag], len(add_frags + [target_frag]))) for target_frag in target_frags]
                can_frag_set = [canonical_smiles(rxn) for rxn in frag_set]
                
                if not set(can_frag_set) & set(train_source_set) and can_add_frags not in add_frag_sets:
                    frag_sets.append(frag_set)
                    add_frag_sets.append(can_add_frags)
                    
            val_frag_sets.extend(frag_sets) 
        
    val_frag_sets = list(itertools.chain.from_iterable(val_frag_sets))
    new_source = "\n".join(val_frag_sets) + "\n"
    new_target = "\n".join(['' for _ in val_frag_sets]) + "\n" # In T5Chem, some kind of target file is required.
    os.makedirs(f'{fd}/rffmg/{frag_method}/attach_point_num/', exist_ok=True)
    save_file(new_source, f'{fd}/rffmg/{frag_method}/attach_point_num/test.source')
    save_file(new_target, f'{fd}/rffmg/{frag_method}/attach_point_num/test.target')
    