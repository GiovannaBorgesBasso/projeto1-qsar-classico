"""
data_collection.py

Downloads raw IC50 bioactivity data for BTK (CHEMBL2842) from the ChEMBL API.
Equivalent to notebooks/01_data_collection.ipynb, extracted as a reusable script.

Usage (standalone):
    python src/data_collection.py
"""

import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGET_ID = "CHEMBL5251"  # Human BTK (Bruton's Tyrosine Kinase)
OUTPUT_PATH = "data/raw/btk_raw.csv"

ACTIVITY_FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_value",
    "standard_units",
    "standard_relation",
    "pchembl_value",
    "assay_chembl_id",
    "target_chembl_id",
    "document_chembl_id",
]


def fetch_bioactivity_data(target_id: str = TARGET_ID) -> pd.DataFrame:
    """
    Query the ChEMBL API for all IC50 bioactivity records against a target.

    Parameters
    ----------
    target_id : str
        ChEMBL target ID (e.g. 'CHEMBL2842' for human BTK).

    Returns
    -------
    pd.DataFrame
        Raw bioactivity records, one row per measurement.
    """
    activity = new_client.activity

    print(f"Querying ChEMBL for IC50 data on target {target_id}...")
    results = activity.filter(
        target_chembl_id=target_id,
        standard_type="IC50",
    ).only(*ACTIVITY_FIELDS)

    df = pd.DataFrame.from_records(results)
    print(f"Done. {len(df)} records retrieved.")
    return df


def save_raw_data(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    """Save the raw, unmodified bioactivity DataFrame to disk."""
    df.to_csv(output_path, index=False)
    print(f"Raw data saved to {output_path}")
    print(f"Shape: {df.shape}")


def main():
    df = fetch_bioactivity_data(TARGET_ID)
    save_raw_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()