"""
featurization.py

Generates Morgan (ECFP4) fingerprints for all molecules in the curated
dataset, producing a feature matrix ready for model training.

Equivalent to notebooks/03_fingerprints.ipynb, extracted as a reusable script.

Usage (standalone):
    python -m src.featurization
"""

import pandas as pd
import numpy as np

from src.utils import smiles_to_morgan

INPUT_PATH = "data/processed/btk_curated.csv"
OUTPUT_PATH = "data/processed/btk_fps.csv"

RADIUS = 2     # ECFP4: radius 2 = 2 bonds away from each atom
N_BITS = 2048  # fingerprint vector length


def generate_fingerprint_matrix(smiles_series, radius=RADIUS, n_bits=N_BITS):
    """
    Convert a Series of SMILES strings into a fingerprint matrix.

    Parameters
    ----------
    smiles_series : pd.Series
        SMILES strings to convert.
    radius : int
        Morgan fingerprint radius.
    n_bits : int
        Length of each fingerprint vector.

    Returns
    -------
    X : np.ndarray
        Fingerprint matrix of shape (n_valid_molecules, n_bits).
    valid_mask : pd.Series
        Boolean mask indicating which rows of the input were successfully
        converted (use this to align labels/IDs with X).
    """
    fps = smiles_series.apply(lambda smi: smiles_to_morgan(smi, radius, n_bits))
    valid_mask = fps.notna()

    failed = (~valid_mask).sum()
    if failed > 0:
        print(f"Warning: {failed} SMILES failed fingerprint conversion and were dropped.")

    X = np.vstack(fps[valid_mask].values)
    return X, valid_mask


def build_fingerprint_dataframe(df, X, valid_mask, n_bits=N_BITS):
    """Combine the fingerprint matrix with molecule IDs and labels into one DataFrame."""
    bit_cols = [f"bit_{i}" for i in range(n_bits)]
    df_fps = pd.DataFrame(X, columns=bit_cols)

    df_valid = df[valid_mask].reset_index(drop=True)
    df_fps.insert(0, "molecule_chembl_id", df_valid["molecule_chembl_id"].values)
    df_fps.insert(1, "pIC50", df_valid["pIC50"].values)
    df_fps.insert(2, "activity", df_valid["activity"].values)

    return df_fps


def main():
    import os
    if os.path.exists(OUTPUT_PATH):
        print(f"Fingerprint dataset already exists at {OUTPUT_PATH} — skipping generation.")
        import pandas as pd
        df_existing = pd.read_csv(OUTPUT_PATH)
        print(f"Existing shape: {df_existing.shape}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded curated dataset: {df.shape[0]} molecules")

    X, valid_mask = generate_fingerprint_matrix(df["canonical_smiles"])
    print(f"Fingerprint matrix shape: {X.shape}")

    df_fps = build_fingerprint_dataframe(df, X, valid_mask)

    df_fps.to_csv(OUTPUT_PATH, index=False)
    print(f"Fingerprint dataset saved to {OUTPUT_PATH}")
    print(f"Final shape: {df_fps.shape}")


if __name__ == "__main__":
    main()