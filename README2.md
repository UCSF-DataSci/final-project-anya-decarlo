# Hidden-Markov-Model Speech Emotion Recognition

_Comprehensive execution guide & expected outputs_

---

## 1. Repository structure (simplified)
```
base model/
│   hmm_part1_load_data.py          # legacy loader (superseded by debug_dataimport)
│   debug_dataimport.py             # canonical feature extractor
│   hmm_part2_model_implementation.py
│   hmm_part3_evaluation.py         # cross-validation, metrics, plots
│   hybrid_classifier.py            # Viterbi→RF/SVM utilities
│   run_state_sweep.py              # 3-6 state HMM benchmark
│   run_hmm_hybrid_pipeline.py      # best-state HMM + Hybrid
│   run_feature_mi_select.py        # MI-based feature-family comparison
│   run_feature_mi_select.py        # MI-based feature-family selection
results/                             # auto-created by runners (timestamped sub-dirs)
```

## 2. Environment setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # numpy, scipy, scikit-learn, librosa, seaborn, matplotlib, joblib
```
If you plan to regenerate MFCCs on CPU-only hardware, ensure `librosa` ≥0.10; optional `numba` speeds up.

## 3. Data preparation
Download the RAVDESS speech subset (≈1.1 GB) and place it anywhere, e.g.
```
/data/ravdess-emotional-speech-audio/
```
No pre-processing required; file names keep their original `03-01-05-01-02-02-12.wav` pattern which encodes the actor ID.

## 4. Pipelines
### 4.1 State-sweep benchmark (3–6 states)
Evaluates MFCC HMMs with 3-,4-,5-,6-state topologies.
```bash
python base\ model/run_state_sweep.py \
       --data_dir /data/ravdess-emotional-speech-audio \
       --feature_type mfcc --n_components 1 --n_iter 100
```
Artifacts (created in `results/state_sweep/`):
* `state_sweep_metrics.csv/tex` — log-likelihood, BIC, weighted-F1 per state count
* `state_sweep_barplot.png` — dual-axis bar/line chart
* `conf_matrix_<N>states.png` — confusion matrix of best-BIC model

### 4.2 Hybrid pipeline
Runs the best-state HMM cross-validation with speaker-independent folds, extracts Viterbi paths, trains a downstream RF/SVM.
```bash
python base\ model/run_hmm_hybrid_pipeline.py \
       --data_dir /data/ravdess-emotional-speech-audio \
       --feature_type mfcc --n_states 5
```
Outputs (timestamped `results/hybrid_<n>s_YYYYMMDD_HHMMSS/`):
* `hmm_cv_metrics.json` — HMM metrics (speaker-independent)
* `hybrid_summary.json` — winning classifier & weighted-F1
* `figures/conf_matrix_hybrid.png`
* `figures/pr_curves.png`
* `figures/feature_importance.png`
* `hmm_vs_hybrid.tex` — LaTeX comparison table

### 4.3 Feature-family comparison (MI-driven)
Ranks all 11 acoustic descriptor families via mutual information and retains the informative subset before a single HMM + Hybrid evaluation.
```bash
# MI ranking only (fast)
python base\ model/run_feature_mi_select.py \
       --data_dir /data/ravdess-emotional-speech-audio \
       --n_states 5 --top_k 2

# MI + full HMM + Hybrid evaluation
python base\ model/run_feature_mi_select.py \
       --data_dir /data/ravdess-emotional-speech-audio \
       --n_states 5 --top_k 2 --evaluate 1
```
Artifacts (timestamped `results/feature_select/<ts>_n<nstates>_(topkK|tauX)/`):
* `metrics/mi_scores.csv|tex`
* `figures/mi_barplot.png`
* If `--evaluate 1` also:
* `metrics/hmm_metrics.json`, `metrics/hybrid_summary.json`
* `figures/conf_matrix_hmm.png`, `figures/conf_matrix_hybrid.png`, `figures/pr_curves.png`, `figures/feature_importance.png`

### 4.4 Full HMM evaluation (precision / recall bar-chart)
```bash
python base\ model/run_hmm_hybrid_pipeline.py \
       --data_dir /data/ravdess-emotional-speech-audio \
       --feature_type mfcc+prosodic --n_states 5 --group_by_actor
```
### 4.5 Hybrid evaluation on final config
```bash
python base\ model/run_hmm_hybrid_pipeline.py \
       --data_dir /data/ravdess-emotional-speech-audio \
       --n_states 5 --group_by_actor
```

## 5. Expected runtime (12-core CPU)
| Stage | Runtime |
|-------|---------|
| State sweep (MFCC) | ~25 min |
| Hybrid (best state) | ~10 min |
| MI selection (all 11 families) | ~15 min |

GPU is **not** required; all models are NumPy + scikit-learn.

## 6. Interpreting outputs
* **Weighted-F1** is the headline performance metric reported in JSON/CSV/TeX tables.
* **Confusion matrices** highlight per-class errors.
* **PR curves** visualise threshold-independent performance; look for area under the curve.
* **Feature-importance plot** (if RF wins) reveals which Viterbi-derived features (state frequencies, transitions, entropy) drive decisions.

## 7. Reproducibility checklist
* All random seeds fixed (`random_state=42`).
* `GroupKFold` ensures actor-exclusive test folds.
* Timestamped output directories prevent overwrite and capture config in filenames.
* Version-pinned `requirements.txt` provided.

## 8. Citation
If you use this codebase please cite:
```
@misc{decarlo2025hmmser,
  title  = {Hidden-Markov-Model Speech Emotion Recognition with Hybrid State-Sequence Classifier},
  author = {Anya DeCarlo},
  year   = {2025, the year of our Vibes},
  note   = {Course Final Project, UCSF DS}
}
