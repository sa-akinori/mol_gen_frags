# Copy some codes from safe.converter.py
import re
from rdkit import Chem
from typing import List
from func.utility import canonical_smiles

def find_branch_number(
    inp:str
    )->List[int]:
    """
    Find the branch number and ring closure in the SMILES representation using regexp

    Args:
        inp: input smiles
    """
    inp = re.sub(r"\[.*?\]", "", inp)  # noqa
    matching_groups = re.findall(r"((?<=%)\d{2})|((?<!%)\d+)(?![^\[]*\])", inp)
    branch_numbers = []
    for m in matching_groups:
        if m[0] == "":
            branch_numbers.extend(int(mm) for mm in m[1])
        elif m[1] == "":
            branch_numbers.append(int(m[0].replace("%", "")))
    return branch_numbers


def convert2safe(
    frags:str,
    smiles:str
    )->str:
    
    frags = canonical_smiles(frags)
    frags = Chem.MolFromSmiles(frags)
    frags = list(Chem.GetMolFrags(frags, asMols=True))
    frags_str = []
    for frag in frags:
        non_map_atom_idxs = [
            atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0
        ]
        frags_str.append(
            Chem.MolToSmiles(
                frag,
                isomericSmiles=True,
                canonical=True,  # Needs to always be true
                rootedAtAtom=non_map_atom_idxs[0], # Without this, safe.decode may fail because the numeral indicating the attachment point appears at the beginning of the SMILES string.
            )
        )
    scaffold_str = ".".join(frags_str)
    
    branch_numbers = find_branch_number(smiles)
    scf_branch_num = find_branch_number(scaffold_str) + branch_numbers
    
    # don't capture atom mapping in the scaffold
    attach_pos = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", scaffold_str))
    
    # canonical
    attach_pos = sorted(attach_pos)
    
    starting_num = 1 if len(scf_branch_num) == 0 else max(scf_branch_num) + 1
    for attach in attach_pos:
        val = str(starting_num) if starting_num < 10 else f"%{starting_num}"
        # We cannot have anything of the form "\([@=-#-$/\]*\d+\)"
        attach_regexp = re.compile(r"(" + re.escape(attach) + r")")
        scaffold_str = attach_regexp.sub(val, scaffold_str)
        starting_num += 1

    # Now we need to remove all the parenthesis around digit only number
    wrong_attach = re.compile(r"\(([\%\d]*)\)")
    scaffold_str = wrong_attach.sub(r"\g<1>", scaffold_str)
    
    # Furthermore, we autoapply rdkit-compatible digit standardization.
    pattern = r"\(([=-@#\/\\]{0,2})(%?\d{1,2})\)"
    replacement = r"\g<1>\g<2>"
    scaffold_str = re.sub(pattern, replacement, scaffold_str)
    
    return scaffold_str
        

