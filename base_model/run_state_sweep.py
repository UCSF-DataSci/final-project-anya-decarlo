#!/usr/bin/env python
"""
State-Sweep Runner
==================
Train / validate HMMs with n_states ∈ {3,4,5,6}, compute average log-likelihood,
BIC, and weighted F1, then output:
    • CSV / pretty table (results/state_sweep/metrics/sweep_metrics.csv)
    • Seaborn bar-plot (results/state_sweep/figures/sweep_barplot.png)
    • Confusion matrix for best model (results/state_sweep/figures/conf_matrix_bestN.png)

Usage (example):
    python base\ model/run_state_sweep.py \
        --data_dir ravdess-emotional-speech-audio \
        --feature_type mfcc --n_components 1 --n_iter 100

Note: This is an initial scaffold – refine BIC calculation and plotting aesthetics
as needed.
"""
from __future__ import annotations

import os
import csv
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import hmm_part3_evaluation as evaluator
import debug_dataimport as data_loader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion."""
    return -2 * log_likelihood + n_params * np.log(n_obs)


def plot_metrics(df: pd.DataFrame, output_path: str):
    """Create bar-plot of avg_logL and weighted F1 vs n_states."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    sns.barplot(x="n_states", y="avg_logL", data=df, color=color1, alpha=0.6, ax=ax1)
    ax1.set_ylabel("Avg Log-Likelihood", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    sns.pointplot(x="n_states", y="avg_F1", data=df, color=color2, ax=ax2)
    ax2.set_ylabel("Weighted F1", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Model Fit vs. Performance across Hidden-State Counts")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper to create organised result folder
def make_timestamp_dir(base: str, feature: str, n_comp: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{feature}_c{n_comp}" if n_comp != 1 else feature
    out_dir = Path(base) / "state_sweep" / f"{ts}_{suffix}"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir()
    return out_dir


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HMM state-sweep runner")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--feature_type", type=str, default="mfcc")
    parser.add_argument("--n_components", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_base", type=str, default="results",
                        help="Base directory for all results")
    args = parser.parse_args()

    out_dir = make_timestamp_dir(args.output_base, args.feature_type, args.n_components)

    # Load data
    obs, labels, classes = data_loader.load_ravdess_features(args.data_dir, args.feature_type)

    # Sweep hidden states 3-6
    sweep_results: List[Dict] = []
    for n_states in range(3, 7):
        print(f"\n=== Evaluating {n_states}-state HMM ===")
        cv_res = evaluator.cross_validate_hmm(obs, labels, classes,
                                             n_states=n_states,
                                             n_components=args.n_components,
                                             n_folds=args.n_folds,
                                             n_iter=args.n_iter,
                                             return_paths=False)
        avg_F1 = cv_res["average_metrics"]["f1_score"]

        # Approximate logL: average of fold accuracies already computed; instead, we
        # sum emotion-specific training logL across full data.
        training_data = {}
        for emo in classes:
            training_data[emo] = [o for o, l in zip(obs, labels) if l == emo]
        emotion_hmms = evaluator.train_emotion_specific_hmms(training_data, classes,
                                                             n_states=n_states,
                                                             n_components=args.n_components,
                                                             n_iter=args.n_iter,
                                                             n_jobs=-1)
        total_logL = sum(p["log_likelihood"] for p in emotion_hmms.values())
        avg_logL = total_logL / len(emotion_hmms)

        # Parameter count – rough: A (N*(N-1)), pi (N-1), means (N*D), cov (N*D)
        D = obs[0].shape[1]
        n_params = n_states * (n_states - 1) + (n_states - 1) + 2 * n_states * D
        n_obs = sum(len(o) for o in obs)
        bic = compute_bic(total_logL, n_params, n_obs)

        sweep_results.append({
            "n_states": n_states,
            "avg_logL": avg_logL,
            "avg_F1": avg_F1,
            "BIC": bic,
        })

    df = pd.DataFrame(sweep_results)
    df.to_csv(out_dir / "metrics" / "sweep_metrics.csv", index=False)
    df.to_latex(out_dir / "metrics" / "sweep_metrics.tex", index=False, float_format="{:.4f}".format)

    print("\nState-sweep summary:\n", df)

    # Bar / point plot
    plot_metrics(df, str(out_dir / "figures" / "sweep_barplot.png"))

    # Choose best model (lowest BIC, tie-break on higher F1)
    best_row = df.sort_values(["BIC", "avg_F1"]).iloc[0]
    best_n = int(best_row["n_states"])
    print(f"\nBest topology: {best_n} states (BIC {best_row['BIC']:.2f})")

    # Final evaluation for confusion matrix
    final_cv = evaluator.cross_validate_hmm(obs, labels, classes,
                                            n_states=best_n,
                                            n_components=args.n_components,
                                            n_folds=args.n_folds,
                                            n_iter=args.n_iter,
                                            return_paths=False)
    fig = evaluator.plot_confusion_matrix(final_cv["all_true_labels"],
                                          final_cv["all_predicted_labels"],
                                          classes)
    fig.savefig(out_dir / "figures" / "conf_matrix_bestN.png", dpi=300)
    plt.close(fig)

    # Save pretty JSON for paper reference
    with open(out_dir / "metrics" / "sweep_summary.json", "w") as fp:
        json.dump(sweep_results, fp, indent=2)

    print("All artifacts saved to", out_dir)


if __name__ == "__main__":
    main()
