import sys
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from func.fragmentation import MultiThresdSmilesToStences, Smi2SentenceOpt, Smi2Sentences
from func.utility import *
import argparse

def sc1_make_sentences_for_training(
    fd: str,
    smilesFilePath: str,
    fragmentMethod: str,
    nSamplingTrialsPerFragset: int = 5,
    debug: bool = True
    ) -> pd.DataFrame:
    """Generate fragment-to-molecule training sentences from curated SMILES.

    Args:
        fd: Base data directory used to build the log output path.
        smilesFilePath: Path to the TSV file containing curated SMILES.
        fragmentMethod: Fragmentation method, either 'brics' or 'rc_cms'.
        nSamplingTrialsPerFragset: Number of sampling trials per fragment set.
        debug: If True, subsample 10000 molecules for a quick debug run.

    Returns:
        DataFrame with columns:
            - 'sentence': Fragment-to-molecule training sentence.
            - 'full_fragments': All fragments produced for the molecule.
            - 'pass_fragments': Fragments that passed the sampling selection.
    """
    outfd   = f'{fd}/t5chem'
    os.makedirs(outfd, exist_ok=True)
    mols    = pd.read_csv(smilesFilePath, sep='\t', index_col=0)
    smiName = 'rdkit_washed_smiles'
    logfp   = LogFile(f'{outfd}/sentences_logs.txt')
    logfp.write(f'Loaded smiles: {len(mols)}')
    
    if debug:
        mols = mols.sample(10000)
        logfp.write('debug mode')

    smiles 	= mols[smiName]
    # smiles length restrictions
    if fragmentMethod=='rc_cms':
        trimRonRing = True
    
    elif fragmentMethod=='brics':
        trimRonRing = False
        
    opt = Smi2SentenceOpt(
                        fragmentMethod=fragmentMethod,
                        fragmentRatio=0.6,
                        removeDummyAtoms=False,
                        smallCfilder=True,
                        trimRonRing=trimRonRing,
                        bigRingThres=7,
                        randomizeSmi=False,
                        nSamplingTrialsPerFragset=nSamplingTrialsPerFragset,
                        nFragmentPatterns=5,
                        uppMolSizeToFragSize=1.75,
                        uniqunize=False)
    
    logfp.write('Parameters for extracting sentences')
    logfp.write(f'{opt}')
    rseed1 = 42
    rseed2 = 1045
    njobs  = -1
    backend= 'multiprocessing'
    logfp.write(f'random seed 1: {rseed1} ,random seed 2: {rseed2}, njobs: {njobs}, backend: {backend}')

    retList, fragsNPList, fragsNPSelList = MultiThresdSmilesToStences(smiles.tolist(),
                                                    rseed1=rseed1,
                                                    rseed2=rseed2,
                                                    opt=opt,
                                                    njobs=njobs,
                                                    batch_num=15, # To ensure reproducibility. Match it to the number of CPU cores on our(the authors') execution machine.
                                                    backend=backend,
                                                    writeFile=False,
                                                    fileName = f'{outfd}/from{len(smiles)}'
                                                    )
    
    frags_df = pd.DataFrame([retList, fragsNPList, fragsNPSelList], index=['sentence', 'full_fragments', 'pass_fragments']).T
    return frags_df

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--frag_method', type=str, default='rc_cms', choices=['brics', 'rc_cms'], help='fragmentation method')
    parser.add_argument('--sampling_num', type=int, default=5, help='number of sampling trials per fragment set (data/rffmg/<frag>/<N>times_sampling)')
    args = parser.parse_args()

    # Setting
    frag_method = args.frag_method
    fd = f'{BASEPATH}/data'
    smilesPath  = f'{fd}/curated/passed_filters_rdkit_canonical_smiles.tsv'

    # Main
    frags_df = sc1_make_sentences_for_training(fd, smilesPath, frag_method, nSamplingTrialsPerFragset=args.sampling_num, debug=False)
    frags_df['smiles'] = frags_df['sentence'].apply(lambda s : s.split('>>')[-1])
    out_dir = f'{fd}/rffmg/{frag_method}/{args.sampling_num}times_sampling'
    os.makedirs(out_dir, exist_ok=True)
    frags_df.to_csv(f'{out_dir}/full_dataset.csv')



