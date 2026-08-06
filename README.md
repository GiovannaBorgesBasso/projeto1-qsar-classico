# projeto1-qsar-classico

> Classical QSAR pipeline: ChEMBL bioactivity data → RDKit molecular fingerprints → scikit-learn (Random Forest, SVM) → cross-validated evaluation, applied to BTK inhibitors.

Part of a personal cheminformatics/ML portfolio developed in parallel with studies at **LabMol (UFG)**, aimed at building applied AI skills for drug discovery.

---

## Overview

This project builds a complete classical QSAR pipeline from raw bioactivity
data to trained, evaluated, and tuned machine learning models — applied to
**BTK (Bruton's Tyrosine Kinase)**, a non-receptor tyrosine kinase central
to B-cell receptor (BCR) signaling.

BTK inhibition is an established therapeutic strategy for **B-cell
malignancies** (the primary approved indication). For autoimmune diseases,
the picture is more nuanced: next-generation selective BTK inhibitors are
under active clinical investigation for **SLE** (Systemic Lupus Erythematosus).

The project covers both **classification** (active/inactive) and
**regression** (continuous pIC50 prediction), compares Random Forest against
SVM/SVR throughout, and includes hyperparameter tuning and class-imbalance
analysis — documented with full reasoning at each step.

---

## Target

| Field | Value |
|-------|-------|
| **Target** | Bruton's Tyrosine Kinase (BTK) |
| **ChEMBL ID** | CHEMBL5251 |
| **Organism** | *Homo sapiens* |
| **Activity type** | IC50 |
| **Raw records collected** | 21,675 |
| **Curated unique compounds** | 9,436 |
| **Approved indications** | B-cell malignancies (CLL, MCL, WM) |
| **Investigational (active)** | SLE — next-gen selective inhibitors in clinical trials |
| **Investigational (equivocal)** | RA — clinical trials conducted, limited efficacy as monotherapy |
| **Reference drugs (oncology)** | Ibrutinib, Acalabrutinib, Zanubrutinib |
| **Reference drugs (autoimmune)** | Remibrutinib (approved for chronic spontaneous urticaria) |

*Abbreviations: CLL = Chronic Lymphocytic Leukemia; MCL = Mantle Cell Lymphoma; WM = Waldenström Macroglobulinemia; SLE = Systemic Lupus Erythematosus; RA = Rheumatoid Arthritis.*

### Biological rationale

BTK sits downstream of the B-cell receptor (BCR) and plays a central role
in B-cell survival, proliferation, and differentiation. Its dysregulation
drives:

- **B-cell malignancies** — the primary therapeutic area where BTK inhibitors
  are currently approved (CLL, mantle cell lymphoma, Waldenström macroglobulinemia).
- **SLE** — B-cell hyperactivation via BCR signaling is central to SLE
  pathogenesis; next-generation selective reversible BTK inhibitors are under
  active clinical investigation for this indication.
- **RA** — BTK was investigated in clinical trials (fenebrutinib, spebrutinib,
  evobrutinib, tirabrutinib) but results were equivocal — limited efficacy as
  monotherapy in humans, despite promising preclinical data. RA is currently
  treated with synthetic DMARDs, anti-TNF biologics, and JAK inhibitors.

First-generation BTK inhibitors (ibrutinib, acalabrutinib) show toxicity
profiles acceptable for oncology but not for chronic autoimmune use.

---

## Key Results

### Classification (active vs. inactive, pIC50 ≥ 6.0 threshold)

| Model | Test AUC-ROC | Test F1 | Test MCC |
|-------|---------------|---------|----------|
| **Random Forest** | **0.928** | **0.957** | **0.548** |
| SVM (subsample n=3,000) | 0.863 | 0.952 | 0.410 |

**Random Forest outperformed SVM on every metric**, and MCC (Matthews
Correlation Coefficient) is used as the primary evaluation metric alongside
AUC given the severe class imbalance (91.5% active / 8.5% inactive).
MCC accounts for all four quadrants of the confusion matrix symmetrically
and is more informative than F1 alone under these conditions.

### Regression (continuous pIC50 prediction)

| Model | Test R² | Test RMSE |
|-------|---------|-----------|
| **Random Forest** | **0.657** | **0.703** |
| SVR (subsample n=3,000) | 0.595 | 0.764 |

### Class imbalance strategies (notebook 07)

Three threshold configurations are formalized as primary outputs —
all use the **same trained model**, only the decision threshold changes:

| Configuration | Threshold | F1 Inactive | Recall Inactive | MCC | Use case |
|---------------|-----------|-------------|-----------------|-----|----------|
| **General-purpose** | 0.50 | **0.590** | 0.650 | **0.548** | Default, balanced |
| **Moderate triage** | 0.75 | 0.510 | 0.820 | 0.490 | Virtual screening |
| **Ultra-conservative** | 0.85 | 0.440 | **0.920** | 0.446 | Strict triage |

### Hyperparameter tuning (notebook 06)

GridSearchCV/RandomizedSearchCV confirmed **default hyperparameters are
already near-optimal** for both models — no measurable AUC improvement.
The performance bottleneck is data-side (limited inactive examples, 2D-only
representation), not model configuration.

### Chemical interpretability

| Task | Top feature | Substructure | Chemical meaning |
|------|-------------|--------------|-----------------|
| Classification | bit_575 | Aryl sulfonyl (–Ar–SO₂–) | H-bond acceptor, hinge region interaction |
| Regression | bit_339 | Chiral carbon (N, Cl substituents) | Stereospecific potency determinant |

Both features are chemically consistent with known BTK inhibitor pharmacophores,
validating that the models learned real structure-activity relationships.

---

## Pipeline

```
ChEMBL API (CHEMBL5251)
        ↓
01_data_collection.ipynb / src/data_collection.py
  └─ Raw IC50 data + SMILES → data/raw/btk_raw.csv (21,675 records)
        ↓
02_preprocessing.ipynb / src/preprocessing.py
  └─ Curation + IC50 → pIC50 + activity labels → data/processed/btk_curated.csv (9,436 molecules)
        ↓
03_fingerprints.ipynb / src/featurization.py
  └─ Morgan Fingerprints (ECFP4, r=2, 2048 bits) → data/processed/btk_fps.csv
        ↓
04_modeling.ipynb / src/train_classifier.py
  └─ RF vs SVM classification → AUC 0.928 (RF) vs 0.863 (SVM), MCC 0.548 vs 0.410
        ↓
05_regression.ipynb / src/train_regressor.py
  └─ RF vs SVR regression → R² 0.657 (RF) vs 0.595 (SVR)
        ↓
06_tuning_classification.ipynb
  └─ GridSearchCV / RandomizedSearchCV → defaults confirmed near-optimal
        ↓
07_imbalance_strategies.ipynb
  └─ Three threshold configs formalized (0.50 / 0.75 / 0.85)
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
│   ├── utils.py
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── featurization.py
│   ├── train_classifier.py
│   ├── train_regressor.py
│   └── main.py
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
# From scratch (re-downloads from ChEMBL, ~10 min total)
python -m src.main

# Reuse existing raw data, skip ChEMBL download (~2 min)
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

- **Clinical context:** ibrutinib, acalabrutinib, and zanubrutinib are
  approved for B-cell malignancies, not autoimmune diseases. BTK inhibition
  for SLE is under active clinical investigation; for RA, trials showed
  equivocal results with limited monotherapy efficacy.
- **Severe class imbalance:** 91.5% active / 8.5% inactive. Even with
  `class_weight='balanced'` and threshold adjustment, inactive-class
  performance is limited by the small number of true negative examples (802).
- **SVM/SVR subsampling:** kernel methods were trained on subsamples of
  3,000 molecules due to computational constraints. RF had access to the
  full training set, making the comparison slightly favorable to RF.
- **2D fingerprints only** — ECFP4 encodes substructure topology, not 3D
  conformation or binding geometry. Stereochemistry is not captured directly.
- **Extrapolation limits** — RF predictions compress toward the mean at
  extreme pIC50 values (below 4 or above 10).
- **No applicability domain analysis** — predictions for structurally novel
  compounds should be treated with caution.

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