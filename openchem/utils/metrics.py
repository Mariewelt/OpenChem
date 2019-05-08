from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
from openchem.utils.sa_score import sascorer


def qed(smiles, return_mean=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    clean_idx = [m is not None for m in mols]
    clean_idx = list(np.where(clean_idx)[0])
    clean_mols = [mols[i] for i in clean_idx]
    if len(clean_mols) > 0:
        score = [QED.qed(mol) for mol in clean_mols]
    else:
        score = -1.0
    if return_mean:
        return np.mean(score)
    else:
        return score


def sa_score(smiles, return_mean=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    clean_idx = [m is not None for m in mols]
    clean_idx = list(np.where(clean_idx)[0])
    clean_mols = [mols[i] for i in clean_idx]
    if len(clean_mols) > 0:
        score = [sascorer.calculateScore(m) for m in clean_mols]
    else:
        score = -1.0
    if return_mean:
        return np.mean(score)
    else:
        return score
