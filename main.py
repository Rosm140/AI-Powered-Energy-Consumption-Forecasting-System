"""
main.py
One-command project runner.

Usage:
    python main.py --all          # full pipeline: generate → train → visualize
    python main.py --generate     # generate dataset only
    python main.py --train        # train models only
    python main.py --visualize    # generate plots only
    python main.py --api          # launch Flask API
"""

import argparse
import subprocess
import sys
import os

sys.path.insert(0, "src")


def run_step(label, fn):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    fn()


def generate():
    import runpy
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    runpy.run_path("data/generate_dataset.py")


def train():
    from train import train as _train
    _train()


def visualize():
    from visualize import (plot_trend, plot_patterns, plot_actual_vs_predicted,
                            plot_residuals, plot_model_comparison, plot_heatmap)
    from preprocess import load_and_clean, add_features
    import pandas as pd

    print("Loading data for plots …")
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


def api():
    subprocess.run([sys.executable, "src/app.py"])


def main():
    parser = argparse.ArgumentParser(description="AI Energy Forecasting Pipeline")
    parser.add_argument("--all",       action="store_true", help="Run full pipeline")
    parser.add_argument("--generate",  action="store_true", help="Generate dataset")
    parser.add_argument("--train",     action="store_true", help="Train models")
    parser.add_argument("--visualize", action="store_true", help="Create plots")
    parser.add_argument("--api",       action="store_true", help="Launch Flask API")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.all or args.generate:
        run_step("STEP 1 — Generating Dataset", generate)
    if args.all or args.train:
        run_step("STEP 2 — Training Models",    train)
    if args.all or args.visualize:
        run_step("STEP 3 — Generating Plots",   visualize)
    if args.api:
        run_step("LAUNCHING Flask API",         api)

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
