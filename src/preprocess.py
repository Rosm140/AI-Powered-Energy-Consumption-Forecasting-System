"""
src/preprocess.py
Loads the raw energy CSV, cleans it, and engineers time-based features.
Returns a ready-to-train DataFrame.
"""

import pandas as pd
import numpy as np


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV, parse datetimes, resample to hourly, fill gaps."""
    df = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
    df = df.resample("h").mean()                    # ensure hourly freq
    df = df.ffill()                                  # forward-fill missing rows
    df.columns = ["Energy_kWh"]
    df = df[df["Energy_kWh"] > 0]                    # drop physically impossible zeros
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features that capture human energy-usage patterns."""
    df = df.copy()
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek           # 0 = Monday
    df["month"]       = df.index.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]     = df.index.quarter

    # Cyclical encoding so the model understands hour 23 → hour 0 continuity
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features: past usage is a strong predictor of future usage
    df["lag_1h"]      = df["Energy_kWh"].shift(1)
    df["lag_24h"]     = df["Energy_kWh"].shift(24)
    df["lag_168h"]    = df["Energy_kWh"].shift(168)  # 1 week ago (same hour)
    df["rolling_24h"] = df["Energy_kWh"].shift(1).rolling(24).mean()

    df = df.dropna()
    return df


FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "month_sin","month_cos",
    "is_weekend", "quarter",
    "lag_1h", "lag_24h", "lag_168h", "rolling_24h",
]
TARGET_COL = "Energy_kWh"


if __name__ == "__main__":
    df = load_and_clean("data/energy.csv")
    df = add_features(df)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Features used: {FEATURE_COLS}")
