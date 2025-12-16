import os
import re
import safe
import pandas as pd
from tqdm import tqdm
from func.fragment_for_safe import convert2safe
from func.utility import BASEPATH
from rdkit import Chem
from multiprocessing import Pool
import multiprocessing as mp
import argparse

# SAFE decoder cannot reproduce stereochemistry; therefore, this limitation needs to be tolerated.
canonical_no_iso = lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False) 

from rdkit import Chem

def convert_dummy_atoms_rdkit(
    smiles:str
    )->str:
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Check all atoms
    for atom in mol.GetAtoms():
        # Dummy atom (*)
        if atom.GetAtomicNum() == 0:
            # Clear atom map number
            atom.SetIsotope(0)
    
    return Chem.MolToSmiles(mol)
    
def process_row(
    row:pd.Series
    )->list:
    
    smiles, f_frags, p_frags = row['smiles'], row['full_fragments'], row['pass_fragments']
    safe_f_frags = convert2safe(f_frags, smiles)
    safe_p_frags = convert2safe(p_frags, smiles)
    decode_smi = safe.decode(safe_f_frags)
    
    # Verify if the decoded SMILES from SAFE can be reproduced.
    if canonical_no_iso(smiles) != canonical_no_iso(decode_smi):
        raise ValueError(f"{smiles} : The safe fragments doesn't match the original smiles.")
    
    return [smiles, safe_f_frags, safe_p_frags, convert_dummy_atoms_rdkit(f_frags), convert_dummy_atoms_rdkit(p_frags)]
                    
if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--frag_method', type=str, default='rc_cms', choices=['brics', 'rc_cms'],
                        help='fragmentation method')

    args = parser.parse_args()
    
    # Setting
    fd = f'{BASEPATH}/data'
    frag_method = args.frag_method
    target_df = pd.read_csv(f'{fd}/rffmg/{frag_method}/full_dataset.csv', index_col=0)
    
    # Main
    n_cores = mp.cpu_count() - 1
    rows_list = [row for _, row in tqdm(target_df.iterrows())]
    with Pool(processes=n_cores) as pool:

        safe_frags = list(tqdm(pool.imap(process_row, rows_list), total=len(rows_list), desc="Processing molecules"))
        
    safe_df = pd.DataFrame(safe_frags, columns=['smiles', 'full_safe', 'pass_safe', 'full_fragments', 'pass_fragments'])
    os.makedirs(f'{fd}/safe/{frag_method}', exist_ok=True)
    safe_df.to_csv(f'{fd}/safe/{frag_method}/safe_smiles.csv')
        