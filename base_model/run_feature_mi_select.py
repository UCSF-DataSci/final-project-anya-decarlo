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
import feature_utils as fu
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
    p.add_argument("--mi_only", action="store_true",
                   help="Only compute MI and skip downstream HMM/hybrid training")
    p.add_argument("--perm_test", type=int, default=0,
                   help="If >0, run permutation test with this many shuffles per family to assess MI significance")
    p.add_argument("--max_combo", type=int, default=1,
                   help="Maximum combination size of families to evaluate (1 = singletons only)")
    args = p.parse_args()

    selector_tag = f"topk{args.top_k}" if args.top_k is not None else f"tau{args.tau}"
    out_dir = make_timestamp_dir(args.output_base, args.n_states, selector_tag)
    figs = out_dir / "figures"
    metrics_dir = out_dir / "metrics"

    # Discover available families
    try:
        families = fu.AVAILABLE_FAMILIES  # type: ignore
    except AttributeError:
        families = ["mfcc", "spectral", "prosodic", "chroma"]  # fallback list

    mi_records: List[Dict] = []
    rng = np.random.default_rng(42)
    summaries_by_family = {}
    labels = None

    print("Computing MI for single families…")
    for fam in families:
        obs, labels, _ = data_loader.load_ravdess_features(args.data_dir, fam)
        # Build summary matrix
        summary_matrix = np.vstack([sequence_summary(seq) for seq in obs])
        mi_val = compute_family_mi(summary_matrix, labels)
        record = {"family": fam, "mi": mi_val}
        # optional permutation test
        if args.perm_test > 0:
            null_vals = []
            for _ in range(args.perm_test):
                shuffled = rng.permutation(labels)
                null_vals.append(compute_family_mi(summary_matrix, shuffled))
            null_vals = np.array(null_vals)
            p_val = (np.sum(null_vals >= mi_val) + 1) / (len(null_vals) + 1)
            record["p_val"] = p_val
            record["mi_null_mean"] = float(null_vals.mean())
        mi_records.append(record)
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

    # Evaluate combinations if requested
    if args.max_combo > 1:
        from itertools import combinations
        fam_list = list(summaries_by_family.keys())
        print(f"Computing MI for combinations up to size {args.max_combo}…")
        for r in range(2, args.max_combo + 1):
            for combo in combinations(fam_list, r):
                combo_name = "+".join(combo)
                if combo_name in summaries_by_family:
                    continue  # skip if already computed
                # concatenate per-utterance sequences
                fam_seqs = [summaries_by_family[f] for f in combo]
                merged_obs = concatenate_sequences(fam_seqs)
                summary_matrix = np.vstack([sequence_summary(seq) for seq in merged_obs])
                mi_val = compute_family_mi(summary_matrix, labels)
                record = {"family": combo_name, "mi": mi_val}
                if args.perm_test > 0:
                    null_vals = []
                    for _ in range(args.perm_test):
                        shuffled = rng.permutation(labels)
                        null_vals.append(compute_family_mi(summary_matrix, shuffled))
                    null_vals = np.array(null_vals)
                    p_val = (np.sum(null_vals >= mi_val) + 1) / (len(null_vals) + 1)
                    record["p_val"] = p_val
                    record["mi_null_mean"] = float(null_vals.mean())
                mi_records.append(record)
        df_mi = pd.DataFrame(mi_records).sort_values("mi", ascending=False)

        # Re-write CSV/LaTeX/plot with full combination results
        df_mi.to_csv(metrics_dir / "mi_scores.csv", index=False)
        with open(metrics_dir / "mi_scores.tex", "w") as fp:
            fp.write(df_mi.to_latex(index=False, float_format="{:.4f}".format,
                                    caption="Mutual information of acoustic feature families and their combinations",
                                    label="tab:mi_scores"))
        sns.barplot(data=df_mi.head(15), x="family", y="mi", color="steelblue")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean MI")
        plt.tight_layout()
        plt.savefig(figs / "mi_barplot.png", dpi=300)
        plt.close()

    # Select families
    if args.perm_test > 0 and args.top_k is None:
        selected = df_mi[df_mi.get("p_val", 1.0) < 0.05]["family"].tolist()
    elif args.top_k is not None:
        selected = df_mi.head(args.top_k)["family"].tolist()
    else:
        selected = df_mi[df_mi["mi"] >= args.tau]["family"].tolist()
    print("Selected families:", selected)

    if args.evaluate and not args.mi_only:
        # Concatenate sequences of selected families
        fam_seqs = [summaries_by_family[f] for f in selected]
        merged_obs = concatenate_sequences(fam_seqs)

        # Load labels & classes once more (using first family)
        _, labels, classes = data_loader.load_ravdess_features(args.data_dir, selected[0])
        file_names = fu.get_filenames(args.data_dir)  # must exist in loader

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
