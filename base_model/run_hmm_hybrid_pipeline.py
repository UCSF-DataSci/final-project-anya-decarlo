#!/usr/bin/env python
"""
Hybrid Pipeline Runner
======================
Runs cross_validate_hmm with return_paths, trains hybrid classifier, and saves
all artefacts (metrics CSV, LaTeX, confusion matrix, PR curves, feature
importance) into a timestamped folder to avoid overwrites.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import debug_dataimport as data_loader
import hmm_part3_evaluation as evaluator
import hybrid_classifier as hybrid


def make_timestamp_dir(base: str, prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base) / f"{prefix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir()
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Hybrid HMM+Classifier runner")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_type", default="mfcc")
    parser.add_argument("--n_states", type=int, required=True)
    parser.add_argument("--n_components", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_base", default="results")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Parallel jobs for HMM training and classifier (-1 = all cores)")
    parser.add_argument("--group_by_actor", action="store_true",
                        help="Enable speaker-independent CV via GroupKFold")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_base) / "hmm" / f"{ts}_{args.feature_type}_n{args.n_states}_hybrid"
    metrics_dir = out_dir / "metrics"
    figs = out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load data (also return file paths so GroupKFold receives matching length)
    if args.group_by_actor:
        obs, labels, classes, file_names = data_loader.load_ravdess_features(
            args.data_dir, args.feature_type, return_file_paths=True)
    else:
        obs, labels, classes = data_loader.load_ravdess_features(args.data_dir, args.feature_type)
        file_names = None

    # ---------------------------------------------------------------------
    # Step 1: HMM cross-validation with Viterbi paths
    cv_res = evaluator.cross_validate_hmm(obs, labels, classes,
                                          n_states=args.n_states,
                                          n_components=args.n_components,
                                          n_folds=args.n_folds,
                                          n_iter=args.n_iter,
                                          return_paths=True,
                                          n_jobs=args.n_jobs,
                                          group_by_actor=args.group_by_actor,
                                          file_names=file_names)

    # Save metrics json
    with open(metrics_dir / "hmm_cv_metrics.json", "w") as fp:
        json.dump(cv_res["average_metrics"], fp, indent=2)

    # ---------------------------------------------------------------------
    # Step 2: Hybrid classifier
    clf_info = hybrid.run_hybrid_pipeline(cv_res, labels, args.n_states, n_jobs=args.n_jobs)

    # Save best score summary
    with open(metrics_dir / "hybrid_summary.json", "w") as fp:
        json.dump({"best_model": clf_info["model_name"],
                   "weighted_F1": clf_info["best_score"]}, fp, indent=2)

    # Plot confusion matrix of hybrid
    fig_cm = evaluator.plot_confusion_matrix(
        cv_res["all_true_labels"],
        clf_info["best_model"].predict(hybrid.build_feature_matrix(cv_res["viterbi_paths"], args.n_states)),
        classes
    )
    fig_cm.savefig(figs / "conf_matrix_hybrid.png", dpi=300)
    plt.close(fig_cm)

    # Plot PR curves & feature importances
    X_feat = hybrid.build_feature_matrix(cv_res["viterbi_paths"], args.n_states)
    hybrid.plot_pr_curves(clf_info["best_model"], X_feat, labels, classes, figs / "pr_curves.png")

    feat_names = [f"freq_s{j}" for j in range(args.n_states)] + \
                 [f"trans_{a}_{b}" for a in range(args.n_states) for b in range(args.n_states)] + ["entropy"]
    hybrid.plot_feature_importance(clf_info["best_model"], feat_names, figs / "feature_importance.png")

    # ------------------------------------------------------------------
    # Additional visualisations to match classic HMM pipeline
    # ------------------------------------------------------------------
    # Per-emotion precision/recall/F1 bar chart
    from sklearn.metrics import classification_report
    rep = classification_report(labels, clf_info["best_model"].predict(X_feat), target_names=classes, output_dict=True)
    per_emotion = pd.DataFrame(rep).T.loc[classes, ["precision", "recall", "f1-score"]]
    per_emotion.reset_index(inplace=True)
    per_emotion.rename(columns={"index": "Emotion", "f1-score": "F1"}, inplace=True)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    pe_melt = per_emotion.melt(id_vars="Emotion", var_name="Metric", value_name="Score")
    ax = sns.barplot(data=pe_melt, x="Emotion", y="Score", hue="Metric")
    plt.ylim(0, 1)
    plt.title("Hybrid model performance by emotion")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.tight_layout()
    plt.savefig(figs / "performance_by_emotion_hybrid.png", dpi=300)
    plt.close()

    # Summary metrics CSV + LaTeX
    summary_path_csv = metrics_dir / "hybrid_summary_metrics.csv"
    pd.DataFrame([cv_res["average_metrics"]]).to_csv(summary_path_csv, index=False)
    with open(metrics_dir / "hybrid_summary_metrics.tex", "w") as fp:
        fp.write(pd.DataFrame([cv_res["average_metrics"]]).to_latex(index=False, float_format="{:.4f}".format,
                               caption="Cross-validation average metrics for Hybrid model", label="tab:hybrid_metrics"))

    # ---------------------------------------------------------------------
    # Create LaTeX table comparing baseline vs hybrid
    latex_path = metrics_dir / "hmm_vs_hybrid.tex"
    df = pd.DataFrame({
        "Model": [f"{args.n_states}-state HMM", "Hybrid"],
        "Weighted F1": [cv_res["average_metrics"]["f1_score"], clf_info["best_score"]]
    })
    with open(latex_path, "w") as fp:
        fp.write(df.to_latex(index=False, float_format="{:.4f}".format,
                             caption="Performance comparison between standalone HMM and Hybrid model",
                             label="tab:hmm_vs_hybrid"))

    print("All hybrid artefacts saved to", out_dir)


if __name__ == "__main__":
    main()
