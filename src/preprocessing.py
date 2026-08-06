"""
preprocessing.py

Curates raw ChEMBL bioactivity data: filters exact IC50 values, removes
missing/invalid entries, converts IC50 -> pIC50, deduplicates molecules
tested in multiple assays, and assigns binary activity labels.

Equivalent to notebooks/02_preprocessing.ipynb, extracted as a reusable script.

Usage (standalone):
    python src/preprocessing.py
"""

import pandas as pd
import numpy as np

from src.utils import is_valid_smiles, ic50_to_pic50

INPUT_PATH = "data/raw/btk_raw.csv"
OUTPUT_PATH = "data/processed/btk_curated.csv"
PIC50_THRESHOLD = 6.0  # pIC50 >= 6 -> active (IC50 <= 1 micromolar)


def filter_exact_nm_values(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only records with an exact IC50 relation ('=') reported in nM."""
    filtered = df[
        (df["standard_relation"] == "=") & (df["standard_units"] == "nM")
    ].copy()
    return filtered


def drop_missing_essentials(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows missing a SMILES string or an IC50 value."""
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    df = df[df["canonical_smiles"].str.strip() != ""]
    return df


def convert_to_pic50(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the standard_value (IC50 in nM) column into a pIC50 column."""
    df = df.copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"] > 0]
    df["pIC50"] = df["standard_value"].apply(ic50_to_pic50)
    return df


def deduplicate_by_molecule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple assay measurements per molecule into a single row,
    using the median pIC50 (robust to outlier assays).
    """
    dedup = (
        df.groupby("molecule_chembl_id", as_index=False)
        .agg(
            canonical_smiles=("canonical_smiles", "first"),
            pIC50=("pIC50", "median"),
            n_assays=("assay_chembl_id", "nunique"),
        )
    )
    return dedup


def validate_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose SMILES cannot be parsed by RDKit."""
    df = df.copy()
    df["valid_smiles"] = df["canonical_smiles"].apply(is_valid_smiles)
    valid = df[df["valid_smiles"]].drop(columns="valid_smiles").reset_index(drop=True)
    return valid


def assign_activity_labels(df: pd.DataFrame, threshold: float = PIC50_THRESHOLD) -> pd.DataFrame:
    """Assign a binary activity label: 1 = active (pIC50 >= threshold), 0 = inactive."""
    df = df.copy()
    df["activity"] = (df["pIC50"] >= threshold).astype(int)
    return df


def curate(df_raw: pd.DataFrame, threshold: float = PIC50_THRESHOLD) -> pd.DataFrame:
    """Run the full curation pipeline on a raw ChEMBL bioactivity DataFrame."""
    df = filter_exact_nm_values(df_raw)
    print(f"After relation/unit filter: {len(df)} records")

    df = drop_missing_essentials(df)
    print(f"After removing missing SMILES/IC50: {len(df)} records")

    df = convert_to_pic50(df)
    print(f"After pIC50 conversion: {len(df)} records")

    df = deduplicate_by_molecule(df)
    print(f"After deduplication: {len(df)} unique molecules")

    df = validate_smiles(df)
    print(f"After SMILES validation: {len(df)} valid molecules")

    df = assign_activity_labels(df, threshold)
    counts = df["activity"].value_counts()
    print(f"Activity labels -> Active: {counts.get(1, 0)}, Inactive: {counts.get(0, 0)}")

    return df[["molecule_chembl_id", "canonical_smiles", "pIC50", "activity", "n_assays"]]


def main():
    df_raw = pd.read_csv(INPUT_PATH)
    print(f"Loaded raw data: {df_raw.shape[0]} records")

    df_curated = curate(df_raw)

    df_curated.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCurated dataset saved to {OUTPUT_PATH}")
    print(f"Final shape: {df_curated.shape}")


if __name__ == "__main__":
    main()