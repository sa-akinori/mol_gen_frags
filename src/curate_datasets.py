import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from func.utility import LogFile, BASEPATH

def SmilesToCanSmiles(smi: str):
    # Note, RDKit successfully translate a blank sentence to mol object
    if (smi is None) or (smi == ''):
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        return None
    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol) # canonical smiles are generated

def GetNHA(mol: Chem.Mol):
	# dummy atom (0) is not counted as heavy atom
	return rdMolDescriptors.CalcNumHeavyAtoms(mol)

def CalcMW(mol: Chem.Mol):
	return rdMolDescriptors._CalcMolWt(mol)

def sc0_prepare_chembldataset(fd, debug=True):
    """
    Curate comopunds in the chembl31 cpds in our group
    """
    outfd       = f'{fd}/curated'
    chemblFile  = f'{fd}/all_curated_cpds_chembl31.tsv'
    mols        = pd.read_csv(chemblFile, sep='\t', index_col=0)
    os.makedirs(outfd, exist_ok=True)

    if debug:
        mols = mols.sample(10000, random_state=0)
    logfile 	= LogFile(f'{outfd}/compound_curation.txt')
    logfile.write(f'loaded molecules: {len(mols)}')
    smi_col     = 'washed_openeye_smiles'

    # curateion by RDKit
    rdsmi_col = 'rdkit_washed_smiles'
    mols[rdsmi_col] = mols[smi_col].apply(SmilesToCanSmiles) # by rdkit 
    ngmols = mols[rdsmi_col].isna().sum()
    logfile.write(f'cannnot handeld mols in rdkit: {ngmols}')
    okmols      = mols[mols[rdsmi_col].notna()]
    failmols    = mols[mols[rdsmi_col].isna()]
    logfile.write(f'passed mols: {len(okmols)}')
    okmols = okmols.drop_duplicates(subset=rdsmi_col)
    logfile.write(f'unique mols: {len(okmols)}')
    failmols.to_csv(f'{outfd}/fail_covert_rdkit_{len(failmols)}.tsv', sep='\t')

    # molecular size filter # 99 percentile in the number of heavy atoms 
    nhaCol  = 'num_heavy_atoms'
    mwCol   = 'molecular_weight'
    roCol   = 'romol'
    okmols[roCol]   = okmols[rdsmi_col].apply(Chem.MolFromSmiles)
    
    # Filter out molecules that failed conversion (None)
    failed_indices = okmols[roCol].isna()
    n_none_mols = failed_indices.sum()
    
    if n_none_mols > 0:
        logfile.write(f'MolFromSmiles failed for {n_none_mols} molecules.')
        
        # Save failed molecules
        failed_mols = okmols[failed_indices]
        failed_mols.to_csv(f'{outfd}/fail_molfromsmiles_{n_none_mols}.tsv', sep='\t')
        
        # Keep only successful molecules
        okmols = okmols[~failed_indices]

    okmols[nhaCol]  = okmols[roCol].apply(GetNHA)
    okmols[mwCol]   = okmols[roCol].apply(CalcMW)
    
    maxha, minha = okmols[nhaCol].max(), okmols[nhaCol].min()
    logfile.write(f'maximum heavy atom number is {maxha}')
    logfile.write(f'minimum heavy atom number is {minha}')

    # eliminate the 95 percentile in NHA and 95 percentile MW 
    percentile = 0.95
    thres_nha_lower = 5
    thres_nha  = okmols[nhaCol].quantile(percentile)
    passNha    = (okmols[nhaCol]<thres_nha) & (okmols[nhaCol]>thres_nha_lower)
    okmolsNha  = okmols[passNha]
    failNha    = okmols[~passNha]
    logfile.write(f'the nha percentile {percentile}')
    logfile.write(f'the nha upper threshold {thres_nha}')
    logfile.write(f'the nha lower threshold {thres_nha_lower}')
    logfile.write(f'passed molecules: {len(okmolsNha)}')
    failNha.to_csv(f'{outfd}/fail_nha_{percentile}_filter_{len(failNha)}.tsv', sep='\t')

    
    # structural aleart is elimiated (only Glaxo and PAINS due to not to remove too many mols)
    filternames     = ['Glaxo', 'PAINS']
    passmols        = okmolsNha[~okmolsNha[filternames].any(axis=1)]
    failmolsFilter  = okmolsNha[okmolsNha[filternames].any(axis=1)]
    logfile.write(f'passd {filternames}: {len(passmols)}')
    
    maxmw, minmw = passmols[mwCol].max(), passmols[mwCol].min()
    logfile.write(f'maximum mw passing the filter is {maxmw}')
    logfile.write(f'minimum mw passing the filter is {minmw}')

    # remove extremely large molecule: MW > 1000 just in case (maybe no molecules are hit here)
    mw_thres = 1000
    mwRange       = passmols[mwCol] < mw_thres

    passmolsFinal = passmols[mwRange]
    failPassmols  = passmols[~mwRange]
    logfile.write(f'mw final thres {mw_thres} passd : {len(passmolsFinal)}')
    # tokenize molecules
    outputcol= ['chembl_id', rdsmi_col, nhaCol, mwCol] + filternames 
    passmolsFinal[outputcol].to_csv(f'{outfd}/passed_filters_rdkit_canonical_smiles.tsv', sep='\t')
    failmolsFilter[outputcol].to_csv(f'{outfd}/failed_filters_rdkit_canonical_smiles_{len(failmolsFilter)}.tsv', sep='\t')
    failPassmols[outputcol].to_csv(f'{outfd}/failed_final_mwthres{mw_thres}_{len(failPassmols)}.tsv', sep='\t')

if __name__ == '__main__':

    sc0_prepare_chembldataset(f'{BASEPATH}/data', debug=False)
        
    






