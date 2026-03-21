import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from typing import List
import datasets
from func.visualization import WriteDataFrameSmilesToXls
from joblib import Parallel, delayed
from tqdm import tqdm
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import Descriptors, Crippen, QED
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from collections import Counter, defaultdict
import itertools
from glob import glob
from typing import Set, Dict

def loadTrainSmiles(
    arc_name:str,
    file_name:str,
    )->Set[str]:
    
    if arc_name == 'safe_gpt':
        row_datasets = datasets.load_from_disk(file_name)
        train_dataset = row_datasets['train']
        trsmiles = train_dataset['smiles']
        
    elif arc_name == 't5chem':
        trsmiles = pd.read_csv(file_name, header=None).squeeze().tolist()
        
    trsmiles = Parallel(n_jobs=-1)(delayed(Smi2CanSmi)(smi) for smi in tqdm(trsmiles, desc='trsmiles, convert canonical'))
    return set(trsmiles)

def loadGenSmiles(
    arc_name:str,
    file_name:str,
    testInputfile:str=None,
    )->pd.DataFrame:
    
    if arc_name == 'safe_gpt':
        genmols = pd.read_csv(file_name, index_col=0)
        target_pattern = r"time_out|error"
        cols_to_check  = [col for col in genmols.columns if 'prediction' in col]
        genmols = genmols[~genmols[cols_to_check].apply(lambda row: row.str.contains(target_pattern, na=False).all(), axis=1)]
        
    elif arc_name == 't5chem':
        genmols = pd.read_csv(file_name)
        inmols  = pd.read_csv(testInputfile, sep='>', header=None, names=['fragment']).iloc[:,[0]]
        genmols = pd.concat([inmols, genmols], axis=1)   
    
    return genmols
    
def Smi2Mol(smi:str)->Chem.Mol:
    mol = None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print('Fail to generate ROMol.')
        return None
    return mol

def isValidSmiles(smi:str):
    a = Smi2Mol(smi)
    if a is not None:
        return Chem.MolToSmiles(a)
    return False

def Smi2CanSmi(smi:str):
    mol = Smi2Mol(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None

def getSAScore(mol:Chem.Mol):
    """
    Calculate the SAscore (Synthetic Accessibility Score) for a given molecule.
    
    Args:
        mol (Chem.Mol): molecule
        
    Returns:
        float: SAscore value. Returns None if SMILES is invalid
    """
    if mol is not None:
        try:
            return sascorer.calculateScore(mol)
        except:
            print(f'Failed to calculate SAscore for: {Chem.MolToSmiles(mol)}')
            return None
    else:
        return None

def calculateTopKAccuracy(
    ranks:List[int],
    k_values:List[int]=[1, 3, 5, 10, 30, 50]
    )->Dict[str, float]:
    """
    Calculate top-k accuracy (mean and std) from rank values.

    Args:
        ranks (List[int]): List of rank values (0 means target not found, 1-50 means position)
        k_values (List[int]): List of k values to calculate accuracy for

    Returns:
        dict: Dictionary with top-k avg and std values
    """
    ranks = np.array(ranks)
    accuracies = {}

    for k in k_values:
        hits = ((ranks >= 1) & (ranks <= k)).astype(int)
        accuracies[f'top_{k}_avg'] = hits.mean()
        accuracies[f'top_{k}_std'] = hits.std()

    return accuracies
    
def getSmiContainAllFrags(
    smis:List[str], 
    smi_frags:List[str],
    algorithm_name:str
    )->List[str]:
    return [smi for smi in smis if molContainAllFrags(Smi2Mol(smi), smi_frags, algorithm_name)]

def getSmiContainAllFrags_exH(
    smis:List[str],
    smi_frags:List[str],
    algorithm_name:str
    )->List[str]:
    return [smi for smi in smis if molContainAllFrags_exH(Smi2Mol(smi), smi_frags, algorithm_name)]

def flatten(seq):

    for x in seq:
        if isinstance(x, (list, tuple)):
            yield from flatten(x)
        else:
            yield x

def dummy_delete(smarts:str)->str:
    """
    Delete dummy atom from SMARTS string.
    
    Args:
        smarts (str): SMARTS string
        
    Returns:
        str: SMARTS string with dummy atom deleted
    """
    smarts_H = smarts.replace("*", "[H]")
    mol = Chem.MolFromSmiles(smarts_H)
    
    return Chem.MolToSmiles(mol)

def anchors_and_core(
    frag_smarts:str,
    dummy2H:bool=True
    ):
    """
    Extracts anchors and core from a fragment.
    dummy atom == *
    anchor atom is the atom connected to dummy.
    """
    frag = Chem.MolFromSmarts(frag_smarts)
    if frag is None:
        raise ValueError(f"Invalid fragment: {frag_smarts}")
    
    for i, a in enumerate(frag.GetAtoms()):
        a.SetAtomMapNum(i+1)
    
    anchor_mapnums = set() # anchor atom is the atom connected to dummy.
    dummy_idxs = list()
    dummy_maps = list()
    
    # Get dummy atoms and anchor atoms
    for f_atom in frag.GetAtoms():
        if f_atom.GetAtomicNum() == 0:  # dummy(*)
            dummy_idxs.append(f_atom.GetIdx())
            dummy_maps.append(f_atom.GetAtomMapNum())
            for nb in f_atom.GetNeighbors():
                anchor_mapnums.add(nb.GetAtomMapNum())
    
    m = Chem.RWMol(frag)
    for idx in dummy_idxs:
        a = m.GetAtomWithIdx(idx)
        if a.GetDegree() != 1:
            raise ValueError(f"Dummy atom (idx {idx}) has degree {a.GetDegree()} (expected 1).")
        
        if dummy2H:
            # Replace [#0] -> [H] (bond is maintained)
            h = Chem.Atom(1) # Hydrogen
            m.ReplaceAtom(idx, h)
            
    core = m.GetMol()
    core = Chem.RemoveHs(core, sanitize=False) if dummy2H else core
    core_mapnums = [c_atom.GetAtomMapNum() for c_atom in core.GetAtoms() if c_atom.GetAtomicNum()] # Exclude attachment point when dummy2H == False
    frag_mapnums = [atom.GetAtomMapNum() for atom in frag.GetAtoms()]
    return core, anchor_mapnums, core_mapnums, dummy_maps, frag_mapnums

def match_is_legal(
    bmol:Chem.Mol,
    match_tuple:tuple,
    core_mapnums:list,
    anchor_mapnums:set,
    algorithm_name:str
    )->bool:
    """ 
    If there is a non-anchor to external bond, it's NG.
    """
    for i, prod_idx in enumerate(match_tuple):
        atom = bmol.GetAtomWithIdx(prod_idx)
        
        if algorithm_name == 'rc_cms':
            if atom.IsInRing():
                continue
        
        mapnum = core_mapnums[i]
        nbs    = [nb for nb in atom.GetNeighbors() if nb.GetIdx() not in match_tuple]
        
        if not nbs:
            continue
        
        else:
            if mapnum not in anchor_mapnums:
                return False
            
            else:
                continue
            
    return True
    
def molContainAllFrags(
    bmol:Chem.Mol,
    smi_frags:str,
    algorithm_name:str
    )->bool:
    """
    Checks if all fragments exist without overlap, and if each embedding
    does not bond externally from non-anchor atoms.
    """
    parts_all = [p.strip() for p in smi_frags.split('.') if p.strip()]
    need = Counter(parts_all)

    frag_combs = dict()
    for frag_str, cnt in need.items():
        core, anchor_mapnums, core_mapnums, _, _ = anchors_and_core(frag_str)
        # 1) Enumerate all matches
        raw_matches = list(bmol.GetSubstructMatches(core, useChirality=True, uniquify=False))
        if not raw_matches:
            return False
        
        # 2) Here, legality filter (external bond check) is implemented
        matches = [m for m in raw_matches if match_is_legal(bmol, m, core_mapnums, anchor_mapnums, algorithm_name)]
        matches = list(set([tuple(sorted(m)) for m in matches]))
        if len(matches) < cnt:
            return False  # Not enough matches
        
        combos = list(itertools.combinations(matches, cnt))
        frag_combs[frag_str] = combos

    match_combs = list(itertools.product(*list(frag_combs.values())))
    match_combs = [list(flatten(match_comb)) for match_comb in match_combs]
    
    duplicate   = [True if len(match_comb) == len(set(match_comb)) else False for match_comb in match_combs]

    if sum(duplicate):
        return True
    
    else:
        return False
    
def molContainAllFrags_exH(
    bmol:Chem.Mol,
    smi_frags:str,
    algorithm_name:str
    )->bool:
    """
    Checks if all fragments exist without overlap, and if each embedding
    does not bond externally from non-anchor atoms.
    """
    parts_all = [p.strip() for p in smi_frags.split('.') if p.strip()]
    need = Counter(parts_all)

    frag_combs = dict()
    for frag_str, cnt in need.items():
        core, anchor_mapnums, core_mapnums, dummy_mapnums, frag_mapnums = anchors_and_core(frag_str, dummy2H=False)
        
        # Enumerate all matches
        raw_matches = list(bmol.GetSubstructMatches(core, useChirality=True, uniquify=False))
        if not raw_matches:
            return False
        
        # Collect the indices of the generated molecule that should be ignored
        new_raw_matches = []
        for m in raw_matches:
            # Extract generated molecule indices corresponding to dummy(*) positions on the fragment
            dummy_idxs = [m[i] for i, mapnum in enumerate(frag_mapnums) if mapnum not in dummy_mapnums]
            new_raw_matches.append(dummy_idxs)
            
        # Implement legality filter (external bond check)
        matches = [m for m in new_raw_matches if match_is_legal(bmol, m, core_mapnums, anchor_mapnums, algorithm_name)]
        matches = list(set([tuple(sorted(m)) for m in matches]))
        if len(matches) < cnt:
            return False  # Not enough matches
            
        combos = list(itertools.combinations(matches, cnt))
        frag_combs[frag_str] = combos

    match_combs = list(itertools.product(*list(frag_combs.values())))
    match_combs = [list(flatten(match_comb)) for match_comb in match_combs]
    
    duplicate   = [True if len(match_comb) == len(set(match_comb)) else False for match_comb in match_combs]

    if sum(duplicate):
        return True
    
    else:
        return False
    

def calculate_avg_tanimoto_similarity(fingerprints: List)->float:
    """
    Calculate average Tanimoto similarity between all pairs of fingerprints efficiently.
    
    Args:
        fingerprints (List): List of RDKit fingerprint objects
        
    Returns:
        float: Average Tanimoto similarity, or 0.0 if empty or single fingerprint
    """
    n = len(fingerprints)
    
    if n < 2:
        return 0.0
    
    total_sum = 0.0
    
    # Calculate similarity for each fingerprint with all others
    for i in range(n - 1):
        # Calculate similarity for all fingerprints after i
        similarities = BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:])
        total_sum += sum(similarities)
    
    # Calculate average similarity
    num_pairs = n * (n - 1) // 2
    return total_sum / num_pairs

def calculate_avg_tanimoto_for_smiles_list(mols_list: List[Chem.Mol]) -> float:
    """
    Calculate average Tanimoto similarity for a list of SMILES strings.
    
    Args:
        smiles_list (List[str]): List of SMILES strings
        
    Returns:
        float: Average Tanimoto similarity
    """
    if not len(mols_list):
        return 0.0
    
    # Convert SMILES to fingerprints using parallel processing for efficiency
    def calc_morgan_fp(mol):
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        return morgan_gen.GetFingerprint(mol)
    
    fingerprints = Parallel(n_jobs=-1)(
        delayed(calc_morgan_fp)(mol) for mol in mols_list
    )
    
    return calculate_avg_tanimoto_similarity(fingerprints)

def calculate_heavy_atom_growth(
    generated_smis: List[str],
    fragment_mol: Chem.Mol
    ) -> List[int]:
    """
    Calculate the difference in heavy atom count between generated molecule and input fragments.
    
    Args:
        generated_smi (str): Generated molecule SMILES
        fragment_str (str): Input fragment string
        
    Returns:
        int: Heavy atom growth (generated - fragments)
    """
    count_heavy_atoms = lambda mol: mol.GetNumHeavyAtoms()
    
    frag_heavy_atoms = count_heavy_atoms(fragment_mol)
    gen_heavy_atoms = Parallel(n_jobs=-1)(
        delayed(count_heavy_atoms)(Chem.MolFromSmiles(smi)) for smi in tqdm(generated_smis))
    
    diff_heavy_atoms = [gen_heavy - frag_heavy_atoms for gen_heavy in gen_heavy_atoms]
    
    return diff_heavy_atoms

def calculateRank(target_can, valid_smis):
    """
    Find the rank of the target molecule among valid canonical SMILES.

    Args:
        target_can: Target molecule canonical SMILES
        valid_smis: List of canonical SMILES strings

    Returns:
        int: Rank of first occurrence (1-indexed), or 0 if not found
    """
    if target_can is None:
        return 0
    for i, smi in enumerate(valid_smis):
        if smi == target_can:
            return i + 1
    return 0

def evaluation_func(genmoldf, catsmiCol, trsmiles, nmaxgen, algorithm_name):

    # Create valid smiles
    genmoldf['valid_smis']   = genmoldf[catsmiCol].apply(lambda x: [Smi2CanSmi(s) for s in x if Smi2Mol(s) is not None])
    genmoldf['nvalid']       = genmoldf['valid_smis'].apply(len)
    genmoldf['validratio']   = genmoldf['nvalid']/nmaxgen

    # Calculate rank (rediscovery)
    if 'target' in genmoldf.columns and 'rank' not in genmoldf.columns:
        genmoldf['rank'] = genmoldf.apply(
            lambda row: calculateRank(Smi2CanSmi(row['target']), row['valid_smis']), axis=1)

    # Create unique smiles
    genmoldf['unique_smis']  = genmoldf['valid_smis'].apply(lambda x:list(set(x)))
    genmoldf['nunique']      = genmoldf['unique_smis'].apply(len)
    genmoldf['uniqueratio']  = genmoldf['nunique']/genmoldf['nvalid']
    genmoldf.loc[~np.isfinite(genmoldf['uniqueratio']),'uniqueratio'] = 0 # 0 division -> 0
    
    # substructure based validity must be checked
    genmoldf['valid_smis_on_frags']  = genmoldf.apply(lambda row: getSmiContainAllFrags(row['unique_smis'],row['fragment'], algorithm_name), axis=1)
    genmoldf['valid_mols_on_frags']  = genmoldf['valid_smis_on_frags'].apply(lambda x: [Smi2Mol(s) for s in x])
    genmoldf['nvalid_onfrags']       = genmoldf['valid_mols_on_frags'].apply(len)
    genmoldf['validfragratio']       = genmoldf['nvalid_onfrags']/genmoldf['nunique']
    genmoldf['valid_smis_on_frags_exH'] = genmoldf.apply(lambda row: getSmiContainAllFrags(row['unique_smis'],row['fragment'], algorithm_name), axis=1)
    genmoldf['valid_mols_on_frags_exH'] = genmoldf['valid_smis_on_frags_exH'].apply(lambda x: [Smi2Mol(s) for s in x])
    genmoldf['nvalid_onfrags_exH']       = genmoldf['valid_mols_on_frags_exH'].apply(len)
    genmoldf['validfragratio_exH']       = genmoldf['nvalid_onfrags_exH']/genmoldf['nunique']
    genmoldf.loc[~np.isfinite(genmoldf['validfragratio']),'validfragratio'] = 0 # 0 division
    genmoldf.loc[~np.isfinite(genmoldf['validfragratio_exH']),'validfragratio_exH'] = 0 # 0 division
    
    # Create novel smiles
    genmoldf['novel_smi']  = genmoldf['unique_smis'].apply(lambda x: set(x) - trsmiles)
    genmoldf['nnovel']     = genmoldf['novel_smi'].apply(len)
    genmoldf['novelratio'] = genmoldf['nnovel']/genmoldf['nunique']
    genmoldf.loc[~np.isfinite(genmoldf['novelratio']),'novelratio'] = 0 # 0 division
    
    # Calculate SAscore
    genmoldf['novel_mols'] = genmoldf['novel_smi'].apply(lambda x: [Smi2Mol(s) for s in x])
    genmoldf['SAscores']   = genmoldf['novel_mols'].apply(lambda x: [getSAScore(m) for m in x])
    
    # Calculate Tanimoto_similarity
    genmoldf['tanimoto_sim'] = genmoldf['novel_mols'].apply(calculate_avg_tanimoto_for_smiles_list)
    genmoldf['tanimoto_sim_onfrags'] = genmoldf['valid_mols_on_frags'].apply(calculate_avg_tanimoto_for_smiles_list)
    
    # Calculate difference of heavy_atoms_num
    genmoldf["diff_heavy_atoms_num"] = genmoldf.apply(lambda row: calculate_heavy_atom_growth(row['valid_smis_on_frags'], Smi2Mol(row['fragment'])), axis=1)
    
    return genmoldf
    
def sc3_check_genmol_results(
    outfd:str,
    genmols:pd.DataFrame,
    trsmiles:List[str],
    algorithm_name:str,
    skipCreateExcel=False,
    n_chunks=5
    )->pd.DataFrame:
    
    # options for visualization (excel file)
    nanalogsForVis	= 30  # number of gen_mols for visualization purpose ()
    nmolsForVis	 	= 100
    rseed			= 42  
    nmaxgen			= 50 # 50 smiles are generated at most 
    rng     = np.random.RandomState(rseed)
    
    os.makedirs(outfd, exist_ok=True)
    
    # random selection for making visualization table 
    if not skipCreateExcel:
        cols        = ['fragment', 'target'] + [f'prediction_{i}' for i in np.sort(rng.choice(np.arange(1,nmaxgen+1),nanalogsForVis, replace=False))]
        selCatmols  = genmols[cols].dropna(axis=0) # row-wise drop
        selCatmols  = selCatmols.sample(min(nmolsForVis, selCatmols.shape[0]), random_state=rseed)
        outfname    = f'{outfd}/gen_samples_visualize.tsv'
        selCatmols.to_csv(outfname, sep='\t')
        WriteDataFrameSmilesToXls(selCatmols, cols, out_filename=outfname.replace('.tsv', '.xlsx'))

    # checking the statistics (validity and correctness (containing the same scaffold)
    genmols     = genmols.where(genmols.notnull(), '') # convert Null to ''
    predcols    = [f'prediction_{i}' for i in range(1,nmaxgen+1)]
    catsmiCol   = 'all_smiles'
    
    genmols[catsmiCol] = genmols[predcols].apply(lambda x: (' '.join(x)).strip().split(), axis=1)
    genmols.drop(columns=predcols, inplace=True)
    
    # Split dataframe into chunks, process sequentially, and save each to disk to reduce memory usage
    chunk_size = len(genmols) // n_chunks if len(genmols) >= n_chunks else len(genmols)
    genmoldfs = [genmols.iloc[i:i+chunk_size] for i in range(0, len(genmols), chunk_size)]
    del genmols

    chunk_dir = f'{outfd}/chunks'
    os.makedirs(chunk_dir, exist_ok=True)

    for chunk_id, genmoldf in enumerate(tqdm(genmoldfs, desc='Processing chunks')):
        sub_chunk_size = os.cpu_count()-1
        sub_chunks = [genmoldf.iloc[j:j+sub_chunk_size] for j in range(0, len(genmoldf), sub_chunk_size)]
        sub_results = Parallel(n_jobs=sub_chunk_size)(
            delayed(evaluation_func)(sub, catsmiCol, trsmiles, nmaxgen, algorithm_name)
            for sub in sub_chunks)
        result = pd.concat(sub_results, ignore_index=True)
        result.to_csv(f'{chunk_dir}/chunk_{chunk_id}.tsv', sep='\t')
        del result
    del genmoldfs

    # Load all chunks from disk and combine
    chunk_files = sorted(glob(f'{chunk_dir}/chunk_*.tsv'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    genmols = pd.concat([pd.read_csv(f, sep='\t', index_col=0) for f in chunk_files], ignore_index=True)
    
    # Calculate top-k accuracy
    if 'rank' in genmols.columns:
        topKacc = calculateTopKAccuracy(ranks=list(genmols['rank']), k_values=[1, 3, 5, 10, 30, 50])
        topKacc_df = pd.DataFrame.from_dict(topKacc, orient='index').T
        topKacc_df.to_csv(f'{outfd}/top_K_acc.tsv', sep='\t')
        
    stats = dict()
    stats['avg_validity']         = genmols['validratio'].mean() 
    stats['std_validity']         = genmols['validratio'].std() 
    stats['avg_validity_onfrags'] = genmols['validfragratio'].mean() # unique fragments should be used and count should be reflected.
    stats['std_validity_onfrags'] = genmols['validfragratio'].std()
    stats['avg_validity_onfrags_exH'] = genmols['validfragratio_exH'].mean() # unique fragments should be used and count should be reflected.
    stats['std_validity_onfrags_exH'] = genmols['validfragratio_exH'].std()
    stats['avg_uniqueness']       = genmols['uniqueratio'].mean()
    stats['std_uniqueness']       = genmols['uniqueratio'].std()
    stats['avg_novelty']          = genmols['novelratio'].mean()
    stats['std_novelty']          = genmols['novelratio'].std()
    # 
    stats['avg_tanimoto_sim']     = genmols[genmols['nnovel'] != 0]['tanimoto_sim'].mean()
    stats['std_tanimoto_sim']     = genmols[genmols['nnovel'] != 0]['tanimoto_sim'].std()
    stats['avg_tanimoto_sim_onfrags'] = genmols[genmols['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].mean()
    stats['std_tanimoto_sim_onfrags'] = genmols[genmols['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].std()
    
    stats_df = pd.Series(stats)
    
    genmols.to_csv(f'{outfd}/curated_data.tsv', sep='\t')
    
    return stats_df, genmols

def calculate_prop_single_molecule(smiles:str)->dict:
    
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return {
            'SMILES': smiles,
            'MW': None,
            'TPSA': None,
            'LogP': None,
            'QED': None
        }
        
    else:
        return {
            'SMILES': smiles,
            'MW': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'LogP': Crippen.MolLogP(mol),
            'QED': QED.qed(mol)
        }

def calcPhysicProp(
    smiles_list:list,
    n_jobs:int
    )->list:
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_prop_single_molecule)(smiles) for smiles in tqdm(smiles_list, desc="Calculating descriptors")
    )
    return results

def calculate_js_divergence_for_properties(
    prop_dfs:list,
    file_names:list,
    prop_name:str,
    bin_size:float=None
    )->pd.DataFrame:
    """
    Calculate Jensen-Shannon divergence between physicochemical property distributions of multiple dataframes.
    Creates a 4x4 comparison matrix when 4 dataframes are provided.
    
    Args:
        prop_dfs (list): List of pandas DataFrames containing physicochemical properties
        file_names (list): List of file names corresponding to the dataframes
        prop_name (str): Property name to compare (e.g., 'MW', 'TPSA', 'LogP', 'QED')
        bin_size (float): Size of each bin.
    
    Returns:
        pd.DataFrame: 4x4 matrix of JS divergences between all pairs of dataframes
    """
    
    print(f"\nCalculating JS divergence for {prop_name}...")
    
    # Get data ranges for consistent binning
    all_values = []
    for df in prop_dfs:
        if prop_name in df.columns:
            valid_values = df[prop_name].dropna()
            all_values.extend(valid_values.tolist())
    
    if not all_values:
        print(f"No valid data found for property {prop_name}")
        return pd.DataFrame()
        
    # Define common bins for all distributions
    min_val, max_val = min(all_values), max(all_values)
    
    # Use specified bin size
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    
    # Calculate histograms for each dataframe
    histograms = {}
    for file_name, df in zip(file_names, prop_dfs):
        if prop_name in df.columns:
            valid_values = df[prop_name].dropna()
            hist, _ = np.histogram(valid_values, bins=bins, density=True)
            # Add small epsilon to avoid zero probabilities
            hist = hist + 1e-10
            # Normalize to make it a probability distribution
            hist = hist / np.sum(hist)
            histograms[file_name] = hist
    
    # Create 4x4 matrix for Jensen-Shannon divergence
    n_files = len(file_names)
    js_matrix = np.zeros((n_files, n_files))
    
    for i in range(n_files):
        for j in range(n_files):
            if i == j:
                js_matrix[i, j] = 0.0  # Divergence with itself is 0
            else:
                file1, file2 = file_names[i], file_names[j]
                if file1 in histograms and file2 in histograms:
                    # Jensen-Shannon divergence using scipy
                    js_div = jensenshannon(histograms[file1], histograms[file2])
                    js_matrix[i, j] = js_div
                    
                    print(f"{file1} vs {file2}: JS divergence={js_div:.4f}")
    
    # Convert to DataFrame with proper labels
    js_df = pd.DataFrame(js_matrix, index=file_names, columns=file_names)
    
    return js_df

def save_js_divergence_results(results, output_file):
    """
    Save Jensen-Shannon divergence results to CSV file.
    
    Args:
        results (dict): Results from calculate_js_divergence_for_properties
        output_file (str): Output CSV file path
    """
    rows = []
    for prop, pairs in results.items():
        for pair_name, js_values in pairs.items():
            row = {
                'property': prop,
                'comparison': pair_name,
                'js_divergence': js_values['js_divergence'],
                'js_distance': js_values['js_distance']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Jensen-Shannon divergence results saved to {output_file}")
    
if __name__=='__main__':
    
    algorithm_name  = 'rc_cms'
    
    ## part1
    # compounds = ['C12=CC=CC=C1C=CC=C2', 'C1(C2=CC=CC=C2)=CC=CC=C1', 'FC1=CC(C2=C(Br)C=CC=C2)=CC=C1', 'FC1=CC(C2=C(Br)C=CC=C2)=CC(C)=C1']
    # frag_sets = ['[*]C1=CC=CC=C1.[*]C1=CC=CC=C1', '[*]C1=C([*])C=CC=C1.[*]C1=C([*])C=CC=C1', '[*]C1=CC([*])=CC=C1.[*]C2=C([*])C=CC=C2']
    
    # part2
    compounds = ['CCC(C1=CC=CC=C1)=O', 'CCC(C1=CC=CC(C)=C1)=O', 'OC(CC1=CC=CC=C1)=O', 'OC(CC1=CC(C)=CC(C)=C1)=O']
    frag_sets = ['[*]C1=CC=CC([*])=C1.[*]C(O)=O']
    
    for compound in compounds:
        
        for frag_set in frag_sets:
            
            compound = Smi2CanSmi(compound)
            frag_set = Smi2CanSmi(frag_set)
            b = molContainAllFrags(bmol=Smi2Mol(compound), smi_frags=frag_set, algorithm_name=algorithm_name)
            c = molContainAllFrags_exH(bmol=Smi2Mol(compound), smi_frags=frag_set, algorithm_name=algorithm_name)
            print(compound, frag_set, b, c)
            
