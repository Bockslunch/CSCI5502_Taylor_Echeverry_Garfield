# src

Reusable code — anything imported by more than one notebook belongs here rather
than being copy-pasted between them.

| Script | What it does | Run from the repository root |
| --- | --- | --- |
| `make_class_distribution.py` | Milestone 1 class-distribution figure | see the script's docstring |
| `m2_rebuild_clean.py` | Replays `notebooks/01-data-cleaning.ipynb` as a script and logs the row/column count of every step (`cleaning_log.json`); reproduces the notebook's 2,520,656 × 72 output exactly | `python src/m2_rebuild_clean.py data/raw data/processed` |
| `m2_quality_stats.py` | Milestone 2 Section 4–5 numbers: quality issues by class, label conflicts, redundant features, KS distances, summary statistics → `m2_stats.json` + CSV tables | `python src/m2_quality_stats.py data/processed` |
| `m2_eda_figures.py` | Milestone 2 Figures 1–7 (PNG, 300 dpi), each in a light and a `-dark` version; the optional third argument copies them into the website's `docs/assets/` | `python src/m2_eda_figures.py data/processed figures docs/assets` |

| `m3_common.py` | Shared Milestone 3 pieces: the 64-feature list and its exclusions, defect flags, models, train-only threshold rule, metrics | imported by the scripts below |
| `m3_rq4_transfer.py` | RQ4: leave-one-day-out comparison of baselines, logistic regression and gradient boosting, with nested tuning and the A/B/C benign-shift decomposition | `python src/m3_rq4_transfer.py data/processed results/m3` |
| `m3_rq5_defects.py` | RQ5: retrain without defect-flagged flows and compare per-class recall | `python src/m3_rq5_defects.py data/processed results/m3` |
| `m3_rq2_separability.py` | RQ2: 11-class attack separability with and without `Init_Win_bytes_*`, permutation importance, transfer without the window fields | `python src/m3_rq2_separability.py data/processed results/m3` |
| `m3_error_analysis.py` | Profiles of false alarms and missed attacks per held-out day | `python src/m3_error_analysis.py data/processed results/m3` |
| `m3_stability.py` | Seed-to-seed spread of the RQ4 gradient-boosting results | `python src/m3_stability.py data/processed results/m3` |
| `m3_figures.py` | Milestone 3 Figures 8–13 (light + dark) and summary tables | `python src/m3_figures.py results/m3 figures docs/assets` |

The three `m2_*` scripts run in that order; each reads what the previous one wrote.
The `m3_*` scripts run after them in the order listed; `m3_rq4_transfer.py` must come
first because the others reuse its chosen hyperparameters and saved scores.

`m2_rebuild_clean.py` peaks at about 6 GB of RAM (the full 2.8M-row dataset is held in
memory); close other programs on an 8 GB machine.

The feature selection planned at Milestone 2 (excluding destination port and identical
columns) is implemented in `m3_common.feature_list()`. Still planned: `sample.py` (the
committed `data/sample/` excerpt).

## Conventions

- Read from `data/raw/`, write to `data/interim/` or `data/processed/`. Never
  modify anything in `data/raw/`.
- Take paths as arguments; do not hard-code absolute paths from anyone's machine.
- Any randomness takes an explicit seed.

## Column name warning

The raw CICIDS2017 CSVs have inconsistent whitespace in their headers — several
column names carry a leading space (`' Flow Duration'`). Strip whitespace from
headers on load, in one place, or every downstream `KeyError` will be a mystery.
