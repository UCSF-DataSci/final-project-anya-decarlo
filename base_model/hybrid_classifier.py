"""
Hybrid Classifier Module
=======================
Derives fixed-length feature vectors from Viterbi-decoded state paths and fits a
non-sequential classifier (Random Forest, SVM, etc.) for emotion prediction.

This is a scaffold; fill in TODOs iteratively.
"""
from __future__ import annotations

from typing import List, Sequence
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def path_to_features(path: Sequence[int], n_states: int) -> np.ndarray:
    """Convert a single Viterbi state path to a fixed-length feature vector.

    Features (suggested):
      1. Normalised state frequency (n_states dims)
      2. Flattened transition count matrix (n_states**2 dims)
      3. Entropy of state distribution (1 dim)
    """
    # TODO: refine / extend
    path = np.asarray(path, dtype=int)
    freqs = np.bincount(path, minlength=n_states).astype(float)
    freq_norm = freqs / freqs.sum()

    # Transition counts
    trans = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(path[:-1], path[1:]):
        trans[a, b] += 1
    trans_flat = trans.flatten()
    if trans_flat.sum() > 0:
        trans_flat /= trans_flat.sum()

    # Entropy
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = freq_norm[freq_norm > 0]
        entropy_val = -(probs * np.log2(probs)).sum() if probs.size else 0.0

    return np.concatenate([freq_norm, trans_flat, [entropy_val]])


def build_feature_matrix(paths: List[Sequence[int]], n_states: int) -> np.ndarray:
    """Build feature matrix for many utterances."""
    features = [path_to_features(p, n_states) if p is not None else None for p in paths]
    # Remove None entries gracefully (should align with labels)
    X = np.vstack([f for f in features if f is not None])
    return X

# ---------------------------------------------------------------------------
# Classifier helper
# ---------------------------------------------------------------------------

def train_rf_svm_cv(X: np.ndarray, y: List[str], cv_splits: int = 5):
    """Grid-search RF and SVM; return best model & metrics."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Random Forest grid
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid,
                      scoring="f1_weighted", cv=skf, n_jobs=-1)
    rf.fit(X, y)

    # SVM grid
    svm_param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf"]
    }
    svm = GridSearchCV(SVC(), svm_param_grid, scoring="f1_weighted",
                       cv=skf, n_jobs=-1)
    svm.fit(X, y)

    # Choose best of the two
    if rf.best_score_ >= svm.best_score_:
        best_model, best_score = rf.best_estimator_, rf.best_score_
        model_name = "RandomForest"
    else:
        best_model, best_score = svm.best_estimator_, svm.best_score_
        model_name = "SVM"

    return {
        "best_model": best_model,
        "model_name": model_name,
        "best_score": best_score,
        "rf_cv_results": rf.cv_results_,
        "svm_cv_results": svm.cv_results_,
    }

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_pr_curves(model, X: np.ndarray, y: List[str], classes: List[str], save_path: str):
    """Plot one-vs-rest Precision-Recall curves for each class."""
    if not hasattr(model, "predict_proba"):
        print("Model lacks predict_proba; skipping PR curves.")
        return

    y_bin = label_binarize(y, classes=classes)  # type: ignore
    y_scores = model.predict_proba(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, c in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
        ap = average_precision_score(y_bin[:, i], y_scores[:, i])
        ax.plot(recall, precision, label=f"{c} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-rest Precision-Recall Curves (Hybrid)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(model, feature_names: List[str], save_path: str):
    """Bar plot of feature importance/coefficient magnitudes."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print("Model lacks importances; skipping plot.")
        return

    # Top-20 features
    idx = np.argsort(importances)[-20:][::-1]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=vals, y=names, orient="h", ax=ax, color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (Hybrid)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Placeholder for orchestration (to be called by a runner script)
# ---------------------------------------------------------------------------

def run_hybrid_pipeline(cv_results: dict, emotion_labels: List[str], n_states: int):
    """Entry point: takes CV results with Viterbi paths, fits hybrid classifier.

    Args:
        cv_results: output dict from cross_validate_hmm(return_paths=True)
        emotion_labels: list of labels aligned with viterbi_paths
        n_states: number of HMM states used
    """
    if "viterbi_paths" not in cv_results:
        raise ValueError("cross_validate_hmm must be called with return_paths=True")

    paths = cv_results["viterbi_paths"]
    # Filter out any None paths and keep aligned labels
    valid_pairs = [(p, l) for p, l in zip(paths, emotion_labels) if p is not None]
    if not valid_pairs:
        raise ValueError("No valid Viterbi paths found.")

    X = build_feature_matrix([p for p, _ in valid_pairs], n_states)
    y = [l for _, l in valid_pairs]

    clf_info = train_rf_svm_cv(X, y)
    print(f"Hybrid classifier ({clf_info['model_name']}) weighted F1: {clf_info['best_score']:.4f}")
    print(classification_report(y, clf_info["best_model"].predict(X)))

    return clf_info

# TODO: add more utilities, confusion matrix for hybrid, etc.
