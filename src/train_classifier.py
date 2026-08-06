"""
train_classifier.py

Trains and evaluates the project's primary classification model: a Random
Forest classifying BTK inhibitors as active/inactive based on pIC50.

Consolidates findings from notebooks 04 (baseline RF vs SVM), 06
(hyperparameter tuning -- found defaults to be near-optimal), and 07
(class imbalance strategies -- three threshold configurations formalized).

Usage (standalone):
    python -m src.train_classifier
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/btk_fps.csv"
MODEL_OUTPUT_PATH = "models/rf_btk_classifier_baseline.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

DEFAULT_THRESHOLD = 0.5
CONSERVATIVE_THRESHOLD = 0.75   # moderate triage -- see notebook 07
ULTRA_CONSERVATIVE_THRESHOLD = 0.85  # maximum Inactive recall -- see notebook 07


def load_features_and_labels(path=INPUT_PATH):
    """Load the fingerprint dataset and split into X (bits) and y (activity)."""
    df = pd.read_csv(path)
    bit_cols = [c for c in df.columns if c.startswith("bit_")]
    X = df[bit_cols].values
    y = df["activity"].values
    return X, y, bit_cols


def train_random_forest(X_train, y_train, random_state=RANDOM_STATE):
    """Train the project's primary Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, threshold=DEFAULT_THRESHOLD, label=""):
    """Evaluate a trained model on a test set at a given decision threshold."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)

    print(f"\n=== {label} (threshold={threshold}) ===")
    print(f"AUC-ROC:   {auc:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"MCC:       {mcc:.3f}")
    print(classification_report(y_test, pred, target_names=["Inactive", "Active"]))

    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall, "mcc": mcc}


def main():
    X, y, bit_cols = load_features_and_labels()
    print(f"Loaded data: X={X.shape}, y={y.shape}")
    print(f"Class balance: {np.bincount(y)} (0=inactive, 1=active)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    model = train_random_forest(X_train, y_train)

    evaluate(model, X_test, y_test, threshold=DEFAULT_THRESHOLD,
             label="General-purpose (balanced F1)")
    evaluate(model, X_test, y_test, threshold=CONSERVATIVE_THRESHOLD,
             label="Moderate triage (threshold=0.75)")
    evaluate(model, X_test, y_test, threshold=ULTRA_CONSERVATIVE_THRESHOLD,
             label="Ultra-conservative triage (threshold=0.85)")

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")
    print(f"Thresholds available at inference time: "
          f"{DEFAULT_THRESHOLD} (general), "
          f"{CONSERVATIVE_THRESHOLD} (moderate triage), "
          f"{ULTRA_CONSERVATIVE_THRESHOLD} (ultra-conservative triage)")


if __name__ == "__main__":
    main()