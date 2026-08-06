"""
utils.py

Shared utility functions used across the QSAR pipeline scripts.
Centralized here to avoid duplicating logic between data collection,
preprocessing, featurization, and modeling steps.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def is_valid_smiles(smi: str) -> bool:
    """
    Check whether a SMILES string can be parsed by RDKit.

    Parameters
    ----------
    smi : str
        SMILES string to validate.

    Returns
    -------
    bool
        True if RDKit can parse the SMILES into a valid molecule.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except Exception:
        return False


def smiles_to_morgan(smi: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """
    Convert a SMILES string into a Morgan (ECFP) fingerprint.

    Parameters
    ----------
    smi : str
        SMILES string of the molecule.
    radius : int, default=2
        Morgan fingerprint radius (radius=2 corresponds to ECFP4).
    n_bits : int, default=2048
        Length of the resulting binary fingerprint vector.

    Returns
    -------
    np.ndarray or None
        Binary fingerprint array of shape (n_bits,), or None if the
        SMILES could not be parsed.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def ic50_to_pic50(ic50_nm: float) -> float:
    """
    Convert IC50 (in nanomolar) to pIC50.

    Formula: pIC50 = -log10(IC50_M), where IC50_M = IC50_nM * 1e-9

    Parameters
    ----------
    ic50_nm : float
        IC50 value in nanomolar (nM). Must be > 0.

    Returns
    -------
    float
        pIC50 value.
    """
    if ic50_nm <= 0:
        raise ValueError("IC50 must be a positive value.")
    return -np.log10(ic50_nm * 1e-9)