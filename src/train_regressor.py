"""
train_regressor.py

Trains and evaluates the project's primary regression model: a Random
Forest predicting continuous pIC50 values for BTK inhibitors.

Consolidates findings from notebook 05 (RF vs SVR comparison -- RF won
with R2=0.742 vs SVR's R2=0.687 on the test set). Only Random Forest is
trained here; see notebooks/05_regression.ipynb for the full SVR comparison
and reasoning.

Usage (standalone):
    python -m src.train_regressor
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/btk_fps.csv"
MODEL_OUTPUT_PATH = "models/rf_btk_regressor.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_features_and_target(path=INPUT_PATH):
    """Load the fingerprint dataset and split into X (bits) and y (pIC50)."""
    df = pd.read_csv(path)
    bit_cols = [c for c in df.columns if c.startswith("bit_")]
    X = df[bit_cols].values
    y = df["pIC50"].values
    return X, y, bit_cols


def train_random_forest_regressor(X_train, y_train, random_state=RANDOM_STATE):
    """Train the project's primary Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, label=""):
    """Evaluate a trained regressor on a test set."""
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)

    print(f"\n=== {label} ===")
    print(f"R2:   {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")

    return {"r2": r2, "rmse": rmse, "mae": mae}


def main():
    X, y, bit_cols = load_features_and_target()
    print(f"Loaded data: X={X.shape}, y={y.shape}")
    print(f"pIC50 range: {y.min():.2f} - {y.max():.2f} (mean: {y.mean():.2f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    model = train_random_forest_regressor(X_train, y_train)
    evaluate(model, X_test, y_test, label="Random Forest Regressor")

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()