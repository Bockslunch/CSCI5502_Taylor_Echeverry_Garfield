# src

Reusable code — anything imported by more than one notebook belongs here rather
than being copy-pasted between them.

| Script | What it does | Run from the repository root |
| --- | --- | --- |
| `make_class_distribution.py` | Milestone 1 class-distribution figure | see the script's docstring |
| `m2_rebuild_clean.py` | Replays `notebooks/01-data-cleaning.ipynb` as a script and logs the row/column count of every step (`cleaning_log.json`); reproduces the notebook's 2,520,656 × 72 output exactly | `python src/m2_rebuild_clean.py data/raw data/processed` |
| `m2_quality_stats.py` | Milestone 2 Section 4–5 numbers: quality issues by class, label conflicts, redundant features, KS distances, summary statistics → `m2_stats.json` + CSV tables | `python src/m2_quality_stats.py data/processed` |
| `m2_eda_figures.py` | Milestone 2 Figures 1–7 (PNG, 300 dpi) | `python src/m2_eda_figures.py data/processed figures` |

The three `m2_*` scripts run in that order; each reads what the previous one wrote.

`m2_rebuild_clean.py` peaks at about 6 GB of RAM (the full 2.8M-row dataset is held in
memory); close other programs on an 8 GB machine.

Planned for later milestones: `features.py` (feature selection, including the
exclusion of destination port and identical columns) and `sample.py` (the committed
`data/sample/` excerpt).

## Conventions

- Read from `data/raw/`, write to `data/interim/` or `data/processed/`. Never
  modify anything in `data/raw/`.
- Take paths as arguments; do not hard-code absolute paths from anyone's machine.
- Any randomness takes an explicit seed.

## Column name warning

The raw CICIDS2017 CSVs have inconsistent whitespace in their headers — several
column names carry a leading space (`' Flow Duration'`). Strip whitespace from
headers on load, in one place, or every downstream `KeyError` will be a mystery.
