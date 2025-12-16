import argparse
from collections import Counter
from typing import List, Tuple, Dict, Iterable, Optional
import itertools
from rdkit import Chem

def flatten(seq):
    for x in seq:
        if isinstance(x, (list, tuple)):
            yield from flatten(x)
        else:
            yield x

def dummy_delete(smarts: str):
    
    smarts_H = smarts.replace("*", "[H]")
    mol = Chem.MolFromSmiles(smarts_H)
    
    return Chem.MolToSmiles(mol)

def find_one_assignment(bmol: str, smi_frags: str):
    
    smi_frags = dummy_delete(smi_frags)
    parts_all = [p.strip() for p in smi_frags.split('.') if p.strip()]
    need = Counter(parts_all)
    
    frag_combos = dict()
    for frag_str, cnt in need.items():
        mol = Chem.MolFromSmarts(frag_str)
        matches = list(bmol.GetSubstructMatches(mol))
        if len(matches) < cnt:
            return False
        combos = list(itertools.combinations(matches, cnt))
        frag_combos[frag_str] = combos

    match_combs = list(itertools.product(*list(frag_combos.values())))
    match_combs = [list(flatten(match_comb)) for match_comb in match_combs]
    duplicate   = [True if len(match_comb) == len(set(match_comb)) else False for match_comb in match_combs]
    
    if sum(duplicate):
        return True
    
    else:
        return False

def main(smiles:str, frags:str):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("ERROR: Failed to parse SMILES.")
        raise SystemExit(1)

    result = find_one_assignment(mol, frags)
    ok = result is not None

if __name__ == "__main__":
    smiles = 'C#CCc1cn(c2cc3c(O)nc(C)nc3cc2)nn1'
    # frags  = '*CC#C.*c1ccc(*)cc1.*c1ccc2nc(C)nc(O)c2c1'
    frags  = dummy_delete('CC#C.c1ccccc1.c1ccc2nc(C)nc(O)c2c1')
    main(smiles, frags)
