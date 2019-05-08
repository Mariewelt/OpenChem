from rdkit import Chem
from rdkit.Chem import QED
import numpy as np


def qed(smiles, return_mean=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    clean_idx = [m is not None for m in mols]
    clean_idx = list(np.where(clean_idx)[0])
    clean_mols = [mols[i] for i in clean_idx]
    qed = [QED.qed(mol) for mol in clean_mols]
    if return_mean:
        return np.mean(qed)
    else:
        qed
