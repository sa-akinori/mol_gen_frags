import logging
import os
from pathlib import Path
from typing import Optional, Union
from joblib import cpu_count, delayed, Parallel
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, BRICS
from copy import deepcopy
import itertools
from func.utility import canonical_smiles

def GetNHA(mol: Chem.Mol):
	# dummy atom (0) is not counted as heavy atom
	return rdMolDescriptors.CalcNumHeavyAtoms(mol)

# constant 
HALOGEN_ATOMNUMBERS = [9, 17, 35, 53] # F, Cl, Br, I 

def BRICSFragmentize(
    mol: Chem.Mol,
    returnSmiles: bool
    ):
    brics_atoms = [brics[0] for brics in BRICS.FindBRICSBonds(mol) if brics]
    
    if not brics_atoms:
        return None
    
    brics_bonds = [mol.GetBondBetweenAtoms(atoms_idx[0], atoms_idx[1]).GetIdx() for atoms_idx in brics_atoms]
    fmol = Chem.FragmentOnBonds(mol, brics_bonds, dummyLabels=[(i + 1, i + 1) for i in range(len(brics_bonds))])
    Chem.SanitizeMol(fmol)
    
    return MolOrSmiles(fmol, returnSmiles)
    
def SatisfyBondConditions(
    bond: Chem.Bond,
    mol: Chem.Mol,
    bigRingThres: int,
    ):
    okRingBond  =  (IsBondDifferentRingCarbons(bond) or IsBondRingCarbonAndNonRing(bond)) \
                    and (bond.GetBondType() == Chem.BondType.SINGLE)
    # Sp3 and (non-ring bonds or macrocylic bonds) and (non-ethyl splitting)
    okSP3bond = IsBondSP3carbons(bond) and \
        (IsBondInRingGtThres(bond, mol, bigRingThres) or (not bond.IsInRing())) and \
        (not DoesBondSplitMakeMethyl(bond, mol)) # except for metyl split
    return okRingBond | okSP3bond
    

def RandomFragmentize(
    mol: Chem.Mol, 
    returnSmiles: bool=False, 
    bigRingThres: int=7, # bonds in big rings are the subject of dissection
    rseed: int=42,
    ratio: float=1.0,
    removeDummy: bool=False
    ):
    """
    Eligible bonds are based on not in the functional groups detected by 
        1. SP3 - SP3 carbons. Hetero -SP3 carbon is not selected. Methyl -SP3 carbon is prohibited 
        2. Ring1 atom - Ring2 atom 
        3. Ring atom - Non ring atom (may)

    input:
        ratio: governs the number of cut bonds from the eligible bond pool
    """
    mol = Chem.RemoveAllHs(mol) # to specify
    fragBondIdxs = list()
    for bond in mol.GetBonds():
        if SatisfyBondConditions(bond, mol, bigRingThres):
            fragBondIdxs.append(bond.GetIdx())
    nebonds = len(fragBondIdxs)
    fmol    = None # will be updated if found
    if nebonds > 0:
        rng     = np.random.RandomState(rseed)
        ratio   = np.clip(ratio, 0.01, 1) # 0 might cause 
        nsel    = np.ceil(nebonds * ratio).astype(int)
        rng.shuffle(fragBondIdxs)
        useFragBondIdxs=fragBondIdxs[:nsel]
        fmol = Chem.FragmentOnBonds(mol, useFragBondIdxs, dummyLabels=[(i + 1, i + 1) for i in range(len(fragBondIdxs))])
        Chem.SanitizeMol(fmol)
    return MolOrSmiles(fmol, returnSmiles)

def IsBondSP3carbons(bond: Chem.Bond):
    batom, eatom    = bond.GetBeginAtom(), bond.GetEndAtom()
    isbsp3      = (batom.GetHybridization() == Chem.HybridizationType.SP3)
    isesp3      = (eatom.GetHybridization() == Chem.HybridizationType.SP3)
    sp3bonds    = isbsp3 and isesp3
    batomNum, eatomNum = batom.GetAtomicNum(), eatom.GetAtomicNum()
    bothCs      = (batomNum == 6) and (eatomNum == 6)
    return sp3bonds and bothCs

def IsBondInRingGtThres(bond: Chem.Bond, mol: Chem.Mol, ringthres: int=7):
    return mol.GetRingInfo().MinBondRingSize(bond.GetIdx()) > ringthres

def IsBondRingCarbonAndNonRing(bond: Chem.Bond):
    batom, eatom = bond.GetBeginAtom(), bond.GetEndAtom()
    if batom.IsInRing():
        return (batom.GetAtomicNum() ==6) and (not eatom.IsInRing())
    else:
        return (eatom.GetAtomicNum() ==6) and (eatom.IsInRing())

def IsBondDifferentRingCarbons(bond: Chem.Bond):
    batom, eatom = bond.GetBeginAtom(), bond.GetEndAtom()
    return (not bond.IsInRing()) and batom.IsInRing() and eatom.IsInRing() and (batom.GetAtomicNum() == 6) and (eatom.GetAtomicNum() == 6)

def MolOrSmiles(mol, returnAsSmi):
    if mol is None: return None # mol is None 
    if returnAsSmi:
        return Chem.MolToSmiles(mol)
    else:
        return mol

def IsSmallAlkylGroup(fragMol: Chem.Mol, nthres=3):
    countNonR = 0 # frag size is measured without R 
    for atom in fragMol.GetAtoms():
        aNum = atom.GetAtomicNum()
        if not (aNum in [0, 1, 6]): # R: 0
            return False
        if aNum == 0: 
            countNonR+=1
    for bond in fragMol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False
    if countNonR < nthres:
        return True
    return False

def DoesBondSplitMakeMethyl(bond: Chem.Bond, mol: Chem.Mol):
    batom, eatom = bond.GetBeginAtom(), bond.GetEndAtom()
    for a1, a2 in zip([batom, eatom], [eatom, batom]):
        for neiAtom in a1.GetNeighbors():
            if neiAtom.GetIdx() == a2.GetIdx():
                continue
            natom=mol.GetAtomWithIdx(neiAtom.GetIdx())
            if (natom.GetTotalNumHs() == 3) and (natom.GetAtomicNum() == 6):    # if methyl then return True. Actually, GetAtomicNum check is not necessary...
                return True
    return False

def CombineMultipleMols(mols: list[Chem.Mol]):
    cmol = mols[0]
    for mol in mols[1:]:
        cmol=Chem.CombineMols(cmol, mol)
    return cmol

def IsRingSystem(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [0,1]:
            continue
        if not atom.IsInRing():
            return False
    return True

def RemoveAtomIsotope(mol: Chem.Mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol

# Post process of the fragmentation
def PostProcessSelectFrags(frags: Chem.Mol, 
                smallCarbonFilter: bool=True, 
                trimRgroupOnRing: bool=True,
                uniquenize: bool=False, 
                returnAsSmi: bool=False):
    """
    smallCarbonFilter: if True, remove fragments with small carbon fragments due to unlikely setting these fragments as input
    """
    tFrags      = Chem.GetMolFrags(frags, asMols=True)
    passFrags   = list()
    passFragsNP = list()
    smiOpt      = Chem.SmilesParserParams()
    smiOpt.removeHs = False # specify not remove H.
    for fmol in tFrags:
        fmol_copy = deepcopy(fmol)
        if smallCarbonFilter and (IsSmallAlkylGroup(fmol)):
            continue
        if trimRgroupOnRing:
            for atom in fmol.GetAtoms():
                if atom.GetAtomicNum() == 0: # free atom (attachment point)
                    attachAtom = list(atom.GetNeighbors())[0]
                    if attachAtom.IsInRing():
                        atom.SetAtomicNum(1) # conver to hydrogen
            fmol = Chem.RemoveAllHs(fmol) # to remove isotoped frags
        # remove atomic numbres 
        fmol = RemoveAtomIsotope(fmol)
        passFrags.append(fmol)
        passFragsNP.append(fmol_copy)
    
    if len(passFrags) == 0: # empty after discarding alkyl groups
        return None, None
    if uniquenize:
        
        smiFrags, pass_idx = np.unique([Chem.MolToSmiles(m) for m in passFrags], return_index=True)[0].tolist(), np.unique([Chem.MolToSmiles(m) for m in passFrags], return_index=True)[1].tolist()
        passFrags = [Chem.MolFromSmiles(s) for s in smiFrags]
        passFragsNP = [passFragsNP[i] for i in pass_idx]
        
    return '.'.join([Chem.MolToSmiles(s) for s in passFrags]), '.'.join([Chem.MolToSmiles(s) for s in passFragsNP])


# API for handling sentence of molecules (randomization is also included)
@dataclass
class Smi2SentenceOpt():
    fragmentMethod: str         = 'rc_cms'
    fragmentRatio: float        = 0.6 # 60% of fragmentable points are selected for dissection
    removeDummyAtoms: bool      = False
    smallCfilder: bool          = True
    trimRonRing: bool           = True
    bigRingThres: int           = 11 #  > thres-sized ring becomes a subject of dissection
    randomizeSmi: bool          = False # randomize smieles
    nSamplingTrialsPerFragset: int     = 1 # number of trials per fragment set.
    nFragmentPatterns: int      = 5 # number of fragment patterns
    uppMolSizeToFragSize: float = 2.0 # upper ratio of mol to fragments
    uniqunize: bool = True # uniquenize the fragments

class Smi2Sentences():
    """
    SMILES -> Sentences
    1: generate random fragments with a predefined probability
    2: randomly sampling a set of fragment with a probability
    """
    def __init__(self, opt: Smi2SentenceOpt):
        self.opt  = deepcopy(opt)
    def __call__(self, 
                smiles: str, 
                rseed1: int=42, # for selection of bonds to be cut.
                rseed2: int=1052   # for selecting fragments to be included
                ) -> list:
        try:
            cmol = Chem.MolFromSmiles(smiles)
        except:
            print('Exception! SMILES cannot be converted', smiles)
            cmol = None
        if cmol is None:
            print('Without exception. SMILES cannot be converted', smiles)
            return list() # return list with 0
        # retFrags    = set() # set
        retFrags    = list()
        retFragsNoPro = list()
        retFragsNPSel = list()
        cpdSmi      = Chem.MolToSmiles(cmol)
        rng2        = np.random.RandomState(rseed2)
        opt         = self.opt
        nha         = GetNHA(cmol)
        for i in range(opt.nFragmentPatterns):
            seed1       = rseed1+i*10
            # fragSmi = self._runFragmentaion(cmol, seed1) # Note! seed value is updated
            fragSmi, fragNoPro, passfragNP = self._runFragmentaion(cmol, seed1)
            if fragSmi is None: 
                logging.info(f'Fragment set was not provided: {cpdSmi}, random seed: {seed1}')
                continue
            lFragSmi    = np.array(fragSmi.split('.'))
            lFragNoPro  = np.array(passfragNP.split('.'))
            for j in range(opt.nSamplingTrialsPerFragset):# random selection from the fragments
                inclMask    = [r >= 0.5 for r in rng2.random(len(lFragSmi))] # threshold 0.5
                fragSelect  = '.'.join(lFragSmi[inclMask].tolist())
                fragNoProSelect  = '.'.join(lFragNoPro[inclMask].tolist())
                if fragSelect == '': # no fragments are selected
                    logging.info(f'No fragment set was sampled: {cpdSmi}, fragset: {lFragSmi}')
                    continue
                if GetNHA(Chem.MolFromSmiles(fragSelect))*opt.uppMolSizeToFragSize >= nha:
                    frag_pair = f"{canonical_smiles(fragSelect)}>>{canonical_smiles(cpdSmi)}"
                    if frag_pair not in retFrags:
                        retFrags.append(frag_pair)
                        retFragsNoPro.append(canonical_smiles(fragNoPro))
                        retFragsNPSel.append(canonical_smiles(fragNoProSelect))
        
        return retFrags, retFragsNoPro, retFragsNPSel
    
    def _runFragmentaion(self, mol: Chem.Mol, rseed: int=42):
        opt = self.opt
        if 'rc_cms' in opt.fragmentMethod:
            frags = RandomFragmentize(mol,
                                    returnSmiles=False, 
                                    bigRingThres=opt.bigRingThres, 
                                    rseed=rseed,
                                    ratio=opt.fragmentRatio,
                                    removeDummy=opt.removeDummyAtoms)
        
    
        elif opt.fragmentMethod == 'brics':
            frags = BRICSFragmentize(mol, returnSmiles=False)
            
        if frags is None: # not found eligible fragments
            return None, None, None

        passfrags, passfragsNP = PostProcessSelectFrags(frags, 
                                smallCarbonFilter=opt.smallCfilder,
                                trimRgroupOnRing=opt.trimRonRing,
                                uniquenize=opt.uniqunize,
                                returnAsSmi=True)
        # return passfrags
        return passfrags, Chem.MolToSmiles(frags), passfragsNP
    

# API for handling pandas series as input
def MultiThresdSmilesToStences(SmilesList: list, 
                                rseed1: int=42,
                                rseed2: int=1052,
                                opt: Optional[Smi2SentenceOpt]=None,
                                njobs: int =-1,
                                batch_num: int = 15, # To ensure reproducibility. Match it to the number of CPU cores on our(the authors') execution machine.
                                backend: str='multiprocessing', # for mac
                                writeFile: bool=False,
                                fileName: Optional[Union[Path, str]]='sentence',
                                ):
    if opt is None:
        opt = Smi2SentenceOpt() # default setting
    njobs         = cpu_count() -1 if njobs < 1 else njobs
    batchSmis     = np.array_split(SmilesList, batch_num)
    sentenceList  = Parallel(n_jobs=njobs, backend=backend)(delayed(WorkerMakeMultiSetences)(dset, rseed1+idx, rseed2+idx, writeFile, fileName, opt, idx) for idx, dset in enumerate(batchSmis))
    if not writeFile:
        retList        = [s for x, _, _ in sentenceList for s in x]# flatten list
        fragsNPList    = [s for _, x, _ in sentenceList for s in x]
        fragsNPSelList = [s for _, _, x in sentenceList for s in x]
        return retList, fragsNPList, fragsNPSelList
    else:
        # cat all the files 
        nWorkerFiles = len(batchSmis)
        with open(f'{fileName}-sentences.txt', 'w') as fp:
            for idx in range(nWorkerFiles): # batch size == file numbers
                with open(f'{fileName}-worker{idx}.txt') as inFp:
                    for line in inFp:
                        fp.write(line)
        
        # remove workder files
        logging.info('clean up worker file.')
        for idx in range(len(batchSmis)):
            os.remove(f'{fileName}-worker{idx}.txt')

def WorkerMakeMultiSetences(smilesList, 
                            rseed1, 
                            rseed2, 
                            writeFile, 
                            fileName, 
                            opt, 
                            workIdx):
    retSet          = list()
    fragsNPSet      = list()
    fragsNPSelSet   = list()
    smi2sentence    = Smi2Sentences(opt)
    nsamples        = len(smilesList)
    if writeFile:
        with open(f'{fileName}-worker{workIdx}.txt', 'w', buffering=1) as fp:
            for idx, smiles in enumerate(smilesList):
                if workIdx == 0:
                    logging.info(f'processing {idx}/{nsamples}')
                oneSet, fragsNP, fragsNPSel = smi2sentence(smiles, rseed1, rseed2)
                if len(oneSet) ==0:
                    continue
                fp.write('\n'.join(oneSet) + '\n')
    else:
        for idx, smiles in enumerate(smilesList):
            if workIdx == 0:
                logging.info(f'processing {idx}/{nsamples}')
            oneSet, fragsNP, fragsNPSel = smi2sentence(smiles, rseed1, rseed2)
            retSet.extend(oneSet) # empty list can be returned if cannot be fragmented
            fragsNPSet.extend(fragsNP)
            fragsNPSelSet.extend(fragsNPSel)
        return retSet, fragsNPSet, fragsNPSelSet
