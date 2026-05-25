"""
src/visualize.py
Generates publication-quality plots saved to outputs/plots/.
Run after train.py has produced outputs/predictions.csv.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

OUT = "outputs/plots"
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    path = f"{OUT}/{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 1. Energy trend overview ─────────────────────────────────────────────────
def plot_trend(df_raw):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=False)

    daily = df_raw["Energy_kWh"].resample("D").mean()
    axes[0].plot(daily.index, daily.values, color="#2196F3", linewidth=1.2)
    axes[0].set_title("Daily Average Energy Consumption (2022–2023)", fontweight="bold")
    axes[0].set_ylabel("kWh")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    monthly = df_raw["Energy_kWh"].resample("ME").mean()
    axes[1].bar(monthly.index, monthly.values, width=20, color="#4CAF50", alpha=0.85)
    axes[1].set_title("Monthly Average Consumption", fontweight="bold")
    axes[1].set_ylabel("kWh")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "01_energy_trend")


# ── 2. Hourly & weekday patterns ─────────────────────────────────────────────
def plot_patterns(df_feat):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hourly = df_feat.groupby("hour")["Energy_kWh"].mean()
    axes[0].plot(hourly.index, hourly.values, marker="o", color="#FF5722", linewidth=2)
    axes[0].set_title("Average Consumption by Hour of Day", fontweight="bold")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("kWh")
    axes[0].set_xticks(range(0, 24, 2))

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = df_feat.groupby("day_of_week")["Energy_kWh"].mean()
    colors = ["#2196F3"] * 5 + ["#FF9800"] * 2
    axes[1].bar(days, weekly.values, color=colors, alpha=0.85)
    axes[1].set_title("Average Consumption by Day of Week", fontweight="bold")
    axes[1].set_ylabel("kWh")
    fig.tight_layout()
    _save(fig, "02_usage_patterns")


# ── 3. Actual vs Predicted ────────────────────────────────────────────────────
def plot_actual_vs_predicted(pred_df, n_days=14):
    """Show last n_days of test set for clarity."""
    subset = pred_df.iloc[-n_days * 24:]
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(subset.index, subset["Energy_kWh"], label="Actual", color="#1565C0",
            linewidth=2, alpha=0.9)
    palette = ["#E53935", "#43A047", "#FB8C00"]
    for col, clr in zip(pred_cols, palette):
        label = col.replace("pred_", "").replace("_", " ")
        ax.plot(subset.index, subset[col], label=f"Predicted ({label})",
                linestyle="--", linewidth=1.5, color=clr, alpha=0.85)
    ax.set_title(f"Actual vs Predicted Energy Consumption (Last {n_days} Days of Test Set)",
                 fontweight="bold")
    ax.set_ylabel("kWh")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "03_actual_vs_predicted")


# ── 4. Residuals analysis ────────────────────────────────────────────────────
def plot_residuals(pred_df):
    best_col = [c for c in pred_df.columns if "Random_Forest" in c][0]
    residuals = pred_df["Energy_kWh"] - pred_df[best_col]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(pred_df[best_col], residuals, alpha=0.2, s=5, color="#7B1FA2")
    axes[0].axhline(0, color="red", linewidth=1.5)
    axes[0].set_xlabel("Predicted kWh")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted (Random Forest)", fontweight="bold")

    axes[1].hist(residuals, bins=60, color="#7B1FA2", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=1.5)
    axes[1].set_xlabel("Residual (kWh)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution", fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_residuals")


# ── 5. Model comparison bar chart ────────────────────────────────────────────
def plot_model_comparison(metrics_path="outputs/metrics.json"):
    with open(metrics_path) as f:
        records = json.load(f)
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics_plot = [("RMSE", "lower is better ↓", "#E53935"),
                    ("MAE",  "lower is better ↓", "#FB8C00"),
                    ("R2",   "higher is better ↑", "#43A047")]
    for ax, (m, subtitle, clr) in zip(axes, metrics_plot):
        ax.bar(df["model"], df[m], color=clr, alpha=0.85, edgecolor="white")
        ax.set_title(f"{m}\n({subtitle})", fontweight="bold")
        ax.set_xticklabels(df["model"], rotation=15, ha="right", fontsize=9)
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "05_model_comparison")


# ── 6. Correlation heatmap ───────────────────────────────────────────────────
def plot_heatmap(df_feat):
    cols = ["Energy_kWh", "hour", "day_of_week", "month",
            "is_weekend", "lag_1h", "lag_24h", "lag_168h", "rolling_24h"]
    corr = df_feat[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_correlation_heatmap")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from preprocess import load_and_clean, add_features

    print("Generating visualizations …")
    df_raw  = load_and_clean("data/energy.csv")
    df_feat = add_features(df_raw)
    pred_df = pd.read_csv("outputs/predictions.csv", parse_dates=["Datetime"],
                          index_col="Datetime")

    plot_trend(df_raw)
    plot_patterns(df_feat)
    plot_actual_vs_predicted(pred_df)
    plot_residuals(pred_df)
    plot_model_comparison()
    plot_heatmap(df_feat)
    print("\nAll plots saved to outputs/plots/")
