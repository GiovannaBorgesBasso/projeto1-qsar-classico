# projeto1-qsar-classico

> Classical QSAR pipeline: ChEMBL bioactivity data → RDKit molecular fingerprints → scikit-learn (Random Forest, SVM) → cross-validated evaluation, applied to BTK inhibitors.

Part of a personal cheminformatics/ML portfolio developed in parallel with research at **LabMol (UFG)**, aimed at building applied AI skills for drug discovery.

---

## Overview

This project builds a complete classical QSAR pipeline from raw bioactivity
data to trained, evaluated, and tuned machine learning models — applied to
**BTK (Bruton's Tyrosine Kinase)**, a non-receptor tyrosine kinase central
to B-cell receptor (BCR) signaling.

BTK inhibition is an established therapeutic strategy for **B-cell
malignancies** (the primary approved indication) and is under active
clinical investigation for **autoimmune diseases** such as Systemic Lupus
Erythematosus (SLE). First-generation BTK
inhibitors (ibrutinib, acalabrutinib) show prohibitive toxicity profiles
for chronic autoimmune use; next-generation selective reversible inhibitors
(e.g. remibrutinib, approved for chronic spontaneous urticaria) represent
the current frontier for autoimmune indications.

The project covers both **classification** (active/inactive) and
**regression** (continuous pIC50 prediction), compares Random Forest against
SVM/SVR throughout, and includes hyperparameter tuning and class-imbalance
analysis — documented with full reasoning at each step.

---

## Target

| Field | Value |
|-------|-------|
| **Target** | Bruton's Tyrosine Kinase (BTK) |
| **ChEMBL ID** | CHEMBL2842 |
| **Organism** | *Homo sapiens* |
| **Activity type** | IC50 |
| **Raw records collected** | 6,502 |
| **Curated unique compounds** | 4,354 |
| **Approved indications** | B-cell malignancies (CLL, MCL, WM) |
| **Investigational indications** | SLE, Rheumatoid Arthritis (clinical trials ongoing) |
| **Reference drugs (oncology)** | Ibrutinib, Acalabrutinib, Zanubrutinib |
| **Reference drugs (autoimmune)** | Remibrutinib (approved for chronic spontaneous urticaria) |

### Biological rationale

BTK sits downstream of the B-cell receptor (BCR) and plays a central role
in B-cell survival, proliferation, and differentiation. Its dysregulation
drives:

- **B-cell malignancies** — the primary therapeutic area where BTK inhibitors
  are currently approved (CLL, mantle cell lymphoma, Waldenström
  macroglobulinemia)
- **Autoimmune diseases** — B-cell hyperactivation via BCR signaling
  contributes to SLE and RA pathogenesis; however, approved BTK inhibitors
  were developed for oncology, where their toxicity is acceptable. For chronic
  autoimmune use, more selective reversible inhibitors are under clinical
  investigation. Rheumatoid Arthritis is currently treated in practice with
  synthetic DMARDs, anti-TNF biologics, and JAK inhibitors — BTK inhibitors
  remain investigational for this indication.

This dataset was chosen for its strong compound volume, well-curated
ChEMBL data, and position at an active frontier of drug discovery research.

---

## Key Results

### Classification (active vs. inactive, pIC50 ≥ 6.0 threshold)

| Model | Test AUC-ROC | Test F1 |
|-------|---------------|---------|
| **Random Forest** | **0.953** | **0.946** |
| SVM | 0.938 | 0.930 |

### Regression (continuous pIC50 prediction)

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| **Random Forest** | **0.742** | **0.602** |
| SVR | 0.687 | 0.663 |

**Random Forest outperformed SVM/SVR on every metric, in both tasks**, and
trained significantly faster (~16s vs ~2min for 5-fold CV). See
`notebooks/04_modeling.ipynb` and `notebooks/05_regression.ipynb` for the
full comparison and reasoning.

### Hyperparameter tuning (notebook 06)

GridSearchCV/RandomizedSearchCV found that **default hyperparameters were
already near-optimal** for both models — tuning produced no measurable
improvement in CV AUC. This indicated the performance bottleneck was in the
data (limited inactive examples, 2D-only representation), not model
configuration — a meaningful negative result documented explicitly rather
than omitted.

### Class imbalance strategies (notebook 07)

The dataset is imbalanced (82.5% active / 17.5% inactive). Four strategies
were compared on the Random Forest classifier:

| Strategy | AUC | F1 (Inactive) | Recall (Inactive) |
|----------|-----|----------------|----------------------|
| **Baseline** (`class_weight='balanced'`) | 0.953 | **0.765** | 0.822 |
| SMOTE oversampling | 0.947 | 0.749 | 0.816 |
| Random undersampling | 0.938 | 0.682 | 0.868 |
| **Threshold adjustment (0.65)** | 0.953 | 0.744 | **0.882** |

No single strategy dominates. The project formalizes **two complementary
models as primary outputs**:

1. **Baseline** (`class_weight='balanced'`, threshold=0.5) — best general-purpose
   balance between classes
2. **Same model + threshold=0.65** — best for conservative virtual screening
   triage, maximizing inactive recall (0.882) to avoid wasting resources on
   likely-inactive candidates; requires no retraining

SMOTE and undersampling underperformed the baseline and are retained as
documented comparison evidence only.

### Chemical interpretability: bit_56

Across **both** the classification and regression Random Forest models, the
dominant fingerprint feature is consistently **bit_56**, decoded via RDKit
as a **urea/amide-centered substructure** (–NH–C(=O)–NH–). This motif is a
well-characterized hydrogen-bond donor/acceptor in kinase inhibitors,
known to interact with the ATP-binding hinge region of the kinase domain.

Its emergence as the top predictor in two independently trained models
solving two different tasks is strong evidence the models captured a real
structure-activity relationship, not statistical noise — without being
given any prior biological information.

---

## Pipeline

```
ChEMBL API (CHEMBL2842)
        ↓
01_data_collection.ipynb / src/data_collection.py
  └─ Raw IC50 data + SMILES → data/raw/btk_raw.csv (6,502 records)
        ↓
02_preprocessing.ipynb / src/preprocessing.py
  └─ Curation + IC50 → pIC50 + activity labels → data/processed/btk_curated.csv (4,354 molecules)
        ↓
03_fingerprints.ipynb / src/featurization.py
  └─ Morgan Fingerprints (ECFP4, r=2, 2048 bits) → data/processed/btk_fps.csv
        ↓
04_modeling.ipynb / src/train_classifier.py
  └─ RF vs SVM classification → AUC 0.953 (RF) vs 0.938 (SVM)
        ↓
05_regression.ipynb / src/train_regressor.py
  └─ RF vs SVR regression → R² 0.742 (RF) vs 0.687 (SVR)
        ↓
06_tuning_classification.ipynb
  └─ GridSearchCV / RandomizedSearchCV → defaults confirmed near-optimal (no AUC gain)
        ↓
07_imbalance_strategies.ipynb
  └─ SMOTE / undersampling / threshold → baseline + threshold=0.65 formalized as primary models
```

---

## Project Structure

```
projeto1-qsar-classico/
├── data/
│   ├── raw/                           # ChEMBL raw download (gitignored)
│   └── processed/                     # Curated + featurized datasets (gitignored)
├── notebooks/                         # Full narrative pipeline with analysis
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_fingerprints.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_regression.ipynb
│   ├── 06_tuning_classification.ipynb
│   └── 07_imbalance_strategies.ipynb
├── src/                               # Production-ready reusable scripts
│   ├── __init__.py
│   ├── utils.py                       # Shared helpers (fingerprinting, pIC50 conversion)
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── featurization.py
│   ├── train_classifier.py
│   ├── train_regressor.py
│   └── main.py                        # Orchestrates the full pipeline end-to-end
├── models/                            # Saved trained models (gitignored)
├── results/                           # Plots and evaluation outputs (gitignored)
├── environment.yml
└── README.md
```

---

## Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| RDKit | Morgan fingerprints (ECFP4), SMILES parsing/validation, bit decoding |
| scikit-learn | RF, SVM/SVR, cross-validation, GridSearchCV, metrics |
| imbalanced-learn | SMOTE, random undersampling |
| chembl_webresource_client | ChEMBL API data download |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |

**Environment**: conda (`qsar-proj1`), see `environment.yml`.

---

## Setup

```bash
git clone https://github.com/GiovannaBorgesBasso/qsar-classico.git
cd qsar-classico

conda env create -f environment.yml
conda activate qsar-proj1

python -m ipykernel install --user --name qsar-proj1 --display-name "Python 3 (qsar-proj1)"
```

### Run the full pipeline

```bash
# From scratch (re-downloads from ChEMBL, ~7 min total)
python -m src.main

# Reuse existing raw data, skip ChEMBL download (~1 min)
python -m src.main --skip-download
```

### Or run individual steps

```bash
python -m src.data_collection
python -m src.preprocessing
python -m src.featurization
python -m src.train_classifier
python -m src.train_regressor
```

### Or explore interactively

```bash
jupyter notebook
# open notebooks/ in order, 01 through 07
```

---

## Limitations

- **Approved indications vs. investigational use:** the BTK inhibitors used
  as reference drugs here (ibrutinib, acalabrutinib, zanubrutinib) are
  approved for B-cell malignancies, not autoimmune diseases. BTK inhibition
  for SLE and RA remains under clinical investigation. The dataset reflects
  the full range of BTK bioactivity data in ChEMBL, including compounds
  developed for oncology.
- **2D fingerprints only** — ECFP4 encodes substructure topology, not 3D
  conformation or binding geometry. Potency information dependent on
  molecular shape or specific protein pocket interactions is not captured.
- **Class imbalance partially addressed, not eliminated** — inactive-class
  performance (F1 ≈ 0.74–0.77) remains below active-class performance
  (F1 ≈ 0.93–0.95) across all strategies tested, reflecting the limited
  number of inactive examples (760 of 4,354 compounds).
- **Extrapolation limits** — RF predictions are bounded by the training set
  range and compress toward the mean at extreme pIC50 values, making the
  model less reliable for unusually potent or unusually weak compounds.
- **No applicability domain analysis** — predictions for structurally novel
  compounds (far from the training set) should be treated with caution.

---

## Context

This is **Project 1** of a personal AI/cheminformatics portfolio, building
progressively toward production-level skills in computational drug discovery.

- **Project 1** (this): Classical QSAR — fingerprints + scikit-learn
- Project 2 (planned): Graph neural networks for BBB permeability prediction (MoleculeNet/BBBP)
- Project 3 (planned): Generative models for de novo drug design

---

## Author

Giovanna — LabMol, Universidade Federal de Goiás
*Cheminformatics · Machine Learning · Drug Discovery*