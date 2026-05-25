"""
src/train.py
Trains a Random Forest Regressor and an MLP Regressor,
compares them, saves the best model, and logs all metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import load_and_clean, add_features, FEATURE_COLS, TARGET_COL


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{'='*45}")
    print(f"  Model : {name}")
    print(f"  MAE   : {mae:.4f} kWh")
    print(f"  RMSE  : {rmse:.4f} kWh")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"{'='*45}")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "MAPE": round(mape, 2)}


def train():
    # ── 1. Load & prepare data ───────────────────────────────────────────
    print("[1/5] Loading and engineering features …")
    df = load_and_clean("data/energy.csv")
    df = add_features(df)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # ── 2. Train / test split (last 20 % = test, preserving time order) ──
    split_idx  = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── 3. Scale features (important for MLP) ────────────────────────────
    print("[2/5] Scaling features …")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 4. Train models ───────────────────────────────────────────────────
    print("[3/5] Training models …")
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "MLP Neural Network": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
        ),
    }

    results  = []
    best_r2  = -np.inf
    best_key = None
    preds    = {}

    for name, m in models.items():
        X_tr = X_train_sc if "MLP" in name else X_train
        X_te = X_test_sc  if "MLP" in name else X_test
        m.fit(X_tr, y_train)
        y_pred = m.predict(X_te)
        preds[name] = y_pred
        metrics = evaluate(name, y_test, y_pred)
        results.append(metrics)
        if metrics["R2"] > best_r2:
            best_r2  = metrics["R2"]
            best_key = name

    # ── 5. Save best model + scaler + test data ───────────────────────────
    print(f"\n[4/5] Best model: {best_key}  (R² = {best_r2:.4f})")
    os.makedirs("models", exist_ok=True)
    best_model = models[best_key]
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # Save test set + predictions for visualization
    os.makedirs("outputs", exist_ok=True)
    test_df = df.iloc[split_idx:][["Energy_kWh"]].copy()
    for name, pred in preds.items():
        safe = name.replace(" ", "_")
        test_df[f"pred_{safe}"] = pred
    test_df.to_csv("outputs/predictions.csv")

    # Save metrics log
    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[5/5] Artifacts saved → models/best_model.pkl, outputs/predictions.csv")
    return best_model, scaler, test_df, results


if __name__ == "__main__":
    train()
