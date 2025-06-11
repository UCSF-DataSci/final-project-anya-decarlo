#!/usr/bin/env python
"""MI-Based Feature Family Selection
=================================
Estimates mutual information (MI) between each acoustic-feature family and the
emotion labels, keeps families passing a threshold / top-k, then trains a
speaker-independent HMM and hybrid classifier on the concatenated raw
sequences of the retained families.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

import debug_dataimport as data_loader
import hmm_part3_evaluation as evaluator
import hybrid_classifier as hybrid

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def make_timestamp_dir(base: str, n_states: int, selector_tag: str) -> Path:
    """Create results/<prefix>/<ts>_n<nstates>_<tag>/ with subfolders"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base) / "feature_select" / f"{ts}_n{n_states}_{selector_tag}"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=False)
    (out_dir / "figures").mkdir()
    return out_dir


def sequence_summary(seq: np.ndarray) -> np.ndarray:
    """Return mean+std along time axis (shape: 2*D)."""
    mu = seq.mean(axis=0)
    sd = seq.std(axis=0)
    return np.hstack([mu, sd])


def compute_family_mi(summaries: np.ndarray, labels: List[str]) -> float:
    mi = mutual_info_classif(summaries, labels, discrete_features=False, random_state=42)
    return float(mi.mean())


def concatenate_sequences(family_sequences: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Horizontally concatenate per-utterance sequences across families.
    Assumes equal #utterances and equal frame counts per utterance across families.
    """
    n_utts = len(family_sequences[0])
    merged = []
    for i in range(n_utts):
        parts = [fam[i] for fam in family_sequences]
        # Ensure same frames; fall back to min len crop
        min_T = min(p.shape[0] for p in parts)
        parts = [p[:min_T] for p in parts]
        merged.append(np.hstack(parts))
    return merged

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Feature family MI selector")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--n_states", type=int, required=True)
    p.add_argument("--tau", type=float, default=0.05,
                   help="MI threshold, ignore if --top_k provided")
    p.add_argument("--top_k", type=int, default=None,
                   help="Keep top-k families by MI (overrides tau if set)")
    p.add_argument("--output_base", default="results")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--evaluate", type=int, choices=[0,1], default=0,
                   help="Set to 1 to run HMM + Hybrid after MI ranking")
    args = p.parse_args()

    selector_tag = f"topk{args.top_k}" if args.top_k is not None else f"tau{args.tau}"
    out_dir = make_timestamp_dir(args.output_base, args.n_states, selector_tag)
    figs = out_dir / "figures"
    metrics_dir = out_dir / "metrics"

    # Discover available families
    try:
        families = data_loader.AVAILABLE_FAMILIES  # type: ignore
    except AttributeError:
        families = ["mfcc", "spectral", "prosodic", "chroma"]  # fallback list

    mi_records: List[Dict] = []
    summaries_by_family = {}
    labels = None

    print("Computing MI for each family…")
    for fam in families:
        obs, labels, _ = data_loader.load_ravdess_features(args.data_dir, fam)
        # Build summary matrix
        summary_matrix = np.vstack([sequence_summary(seq) for seq in obs])
        mi_val = compute_family_mi(summary_matrix, labels)
        mi_records.append({"family": fam, "mi": mi_val})
        summaries_by_family[fam] = obs  # keep raw sequences for later

    df_mi = pd.DataFrame(mi_records).sort_values("mi", ascending=False)
    df_mi.to_csv(metrics_dir / "mi_scores.csv", index=False)

    # LaTeX table & bar plot
    with open(metrics_dir / "mi_scores.tex", "w") as fp:
        fp.write(df_mi.to_latex(index=False, float_format="{:.4f}".format,
                                caption="Mutual information of acoustic feature families",
                                label="tab:mi_scores"))
    sns.barplot(data=df_mi, x="family", y="mi", color="steelblue")
    plt.xticks(rotation=45)
    plt.ylabel("Mean MI")
    plt.tight_layout()
    plt.savefig(figs / "mi_barplot.png", dpi=300)
    plt.close()

    # Select families
    if args.top_k is not None:
        selected = df_mi.head(args.top_k)["family"].tolist()
    else:
        selected = df_mi[df_mi["mi"] >= args.tau]["family"].tolist()
    print("Selected families:", selected)

    if args.evaluate:
        # Concatenate sequences of selected families
        fam_seqs = [summaries_by_family[f] for f in selected]
        merged_obs = concatenate_sequences(fam_seqs)

        # Load labels & classes once more (using first family)
        _, labels, classes = data_loader.load_ravdess_features(args.data_dir, selected[0])
        file_names = data_loader.get_filenames(args.data_dir)  # must exist in loader

        cv_res = evaluator.cross_validate_hmm(
            merged_obs, labels, classes,
            n_states=args.n_states,
            n_folds=args.n_folds,
            n_iter=args.n_iter,
            group_by_actor=True,
            file_names=file_names,
            return_paths=True
        )
        with open(metrics_dir / "hmm_metrics.json", "w") as fp:
            json.dump(cv_res["average_metrics"], fp, indent=2)

        clf_info = hybrid.run_hybrid_pipeline(cv_res, labels, args.n_states)
        with open(metrics_dir / "hybrid_summary.json", "w") as fp:
            json.dump({"best_model": clf_info["model_name"],
                       "weighted_F1": clf_info["best_score"]}, fp, indent=2)

    print("All artefacts saved to", out_dir)


if __name__ == "__main__":
    main()
