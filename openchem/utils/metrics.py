from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
import networkx as nx
from openchem.utils.sa_score import sascorer
from rdkit.Chem import Descriptors


def reward_penalized_log_p(smiles, return_mean=True):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    mols = [Chem.MolFromSmiles(sm) for sm in smiles]
    log_p = np.array([Chem.Descriptors.MolLogP(mol) for mol in mols])
    SA = -np.array(sa_score(smiles, return_mean=False))

    # cycle score
    cycle_score = []
    for mol in mols:
        cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_score.append(-cycle_length)

    cycle_score = np.array(cycle_score)

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    score = list(normalized_log_p + normalized_SA + normalized_cycle)
    if return_mean:
        return np.mean(score)
    else:
        return score


def logP_pen(smiles, return_mean=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    logp_pen = []
    for mol in mols:
        cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))

        tmp = sum([len(c) > 6 for c in cycle_list])

        logp_pen.append(Descriptors.MolLogP(mol) - sascorer.calculateScore(mol) - tmp)

    if return_mean:
        return np.mean(logp_pen)
    else:
        return logp_pen


def logP(smiles, return_mean=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    clean_idx = [m is not None for m in mols]
    clean_idx = list(np.where(clean_idx)[0])
    clean_mols = [mols[i] for i in clean_idx]
    if len(clean_mols) > 0:
        score = [Chem.Crippen.MolLogP(mol) for mol in clean_mols]
    else:
        score = -10.0
    if return_mean:
        return np.mean(score)
    else:
        return score


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
