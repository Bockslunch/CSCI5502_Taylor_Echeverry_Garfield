# Notebooks

Run in order, top to bottom, from this folder (paths are relative to `notebooks/`):

| Notebook | Reads | Writes / shows |
| --- | --- | --- |
| `01-data-cleaning.ipynb` | `../data/raw/*.csv` (8 files) | `../data/processed/cicids2017_clean.csv` (2,520,656 rows × 72 columns) |
| `02-eda.ipynb` | `../data/processed/cicids2017_clean.csv` | Destination-port and Flow Duration (120-second timeout) analysis |
| `03-kmeans-clustering.ipynb` | `../data/processed/cicids2017_clean.csv`, `../figures/rq3-class-profiles-dark.png` | RQ3: k-means at the hypothesized k = 4 (all flows and class-balanced), k search over 2–69, cluster profiles at k = 22 |
| `04-prediction-baselines.ipynb` | `../data/processed/cicids2017_clean.csv` | RQ4 baselines: majority-class and stratified dummy classifiers, leave-one-day-out |

The full Milestone 3 model comparison (baselines plus logistic regression and gradient
boosting, same leave-one-day-out folds) lives in `../src/m3_*.py`; see `../src/README.md`.

Restart-and-run-all before committing, and keep shared logic in `../src/`.
