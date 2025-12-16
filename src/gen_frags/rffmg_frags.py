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
    debug=True
    ):
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
                        nSamplingTrialsPerFragset=5,
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
    parser.add_argument('--frag_method', type=str, default='rc_cms', choices=['brics', 'rc_cms'], 
                        help='fragmentation method')
    args = parser.parse_args()

    # Setting
    frag_method = args.frag_method
    fd = f'{BASEPATH}/data'
    smilesPath  = f'{fd}/curated/passed_filters_rdkit_canonical_smiles.tsv'

    # Main
    frags_df = sc1_make_sentences_for_training(fd, smilesPath, frag_method, debug=False)
    frags_df['smiles'] = frags_df['sentence'].apply(lambda s : s.split('>>')[-1])
    os.makedirs(f'{fd}/rffmg/{frag_method}', exist_ok=True)
    frags_df.to_csv(f'{fd}/rffmg/{frag_method}/full_dataset.csv')



