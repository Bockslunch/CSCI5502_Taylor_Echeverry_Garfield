# Notebooks

Run in order, top to bottom, from this folder (paths are relative to `notebooks/`):

| Notebook | Reads | Writes / shows |
| --- | --- | --- |
| `01-data-cleaning.ipynb` | `../data/raw/*.csv` (8 files) | `../data/processed/cicids2017_clean.csv` (2,520,656 rows × 72 columns) |
| `02-eda.ipynb` | `../data/processed/cicids2017_clean.csv` | Destination-port and Flow Duration (120-second timeout) analysis |

Restart-and-run-all before committing, and keep shared logic in `../src/`.
