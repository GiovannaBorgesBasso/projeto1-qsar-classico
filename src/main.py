"""
main.py

Orchestrates the full QSAR pipeline end-to-end:
ChEMBL data collection -> preprocessing -> featurization -> model training.

Usage:
    python -m src.main

    # Skip data collection if data/raw/btk_raw.csv already exists:
    python -m src.main --skip-download
"""

import argparse
import time

from src import data_collection, preprocessing, featurization, train_classifier, train_regressor


def run_pipeline(skip_download: bool = False):
    start = time.time()

    if not skip_download:
        print("\n" + "=" * 60)
        print("STEP 1/5 — Data Collection")
        print("=" * 60)
        data_collection.main()
    else:
        print("\nSkipping data collection (--skip-download). "
              "Using existing data/raw/btk_raw.csv")

    print("\n" + "=" * 60)
    print("STEP 2/5 — Preprocessing")
    print("=" * 60)
    preprocessing.main()

    print("\n" + "=" * 60)
    print("STEP 3/5 — Featurization")
    print("=" * 60)
    featurization.main()

    print("\n" + "=" * 60)
    print("STEP 4/5 — Classifier Training")
    print("=" * 60)
    train_classifier.main()

    print("\n" + "=" * 60)
    print("STEP 5/5 — Regressor Training")
    print("=" * 60)
    train_regressor.main()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed/60:.1f} minutes.")
    print("Trained models saved in models/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run the full BTK QSAR pipeline.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip ChEMBL data collection and reuse data/raw/btk_raw.csv",
    )
    args = parser.parse_args()
    run_pipeline(skip_download=args.skip_download)


if __name__ == "__main__":
    main()