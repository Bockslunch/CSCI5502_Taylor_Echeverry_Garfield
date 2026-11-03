"""Stability check for the RQ4 gradient-boosting results.

Usage (from the repository root, after m3_rq4_transfer.py):
    python src/m3_stability.py data/processed results/m3

Refits the condition-A gradient-boosting model for each attack day with three
extra random seeds (the seed controls HGB's internal early-stopping split, so it
changes which rows the trees see and when training stops). Hyperparameters, the
fit/calibration split and the threshold rule are unchanged. If recall on the
held-out day moves a lot between seeds, a single-run number is not a stable
estimate of transfer, which matters for interpreting RQ4 and RQ5.

Output: m3_stability.csv (one row per day x seed, with recall/FPR/PR-AUC and
per-class recall as extra columns)
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, PATHS AND JSON
import json
import sys
from pathlib import Path

# IMPORT THE NUMERIC STACK
import numpy as np
import pandas as pd

# MAKE `src/` IMPORTABLE AND LOAD THE SHARED PIECES
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import (SEED, evaluate, feature_list, load_clean, make_model, pick_threshold,
                       split_fit_calib)

# READ INPUT AND OUTPUT FOLDERS FROM THE COMMAND LINE
DATA_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])

# THE ORIGINAL SEED (ALREADY RUN IN RQ4) PLUS THREE NEW ONES
NEW_SEEDS = [1, 2, 3]
ATTACK_DAYS = ["Tuesday", "Wednesday", "Thursday", "Friday"]

# LOAD THE SAME DATA AND FEATURES AS RQ4
df = load_clean(DATA_DIR)
FEATS = feature_list(df)
X = df[FEATS].to_numpy(np.float32)
y = df["is_attack"].to_numpy()
labels = df["Label"].to_numpy()
day = df["day"].to_numpy()
del df

# THE HYPERPARAMETERS RQ4 CHOSE FOR EACH DAY AND THE ORIGINAL-SEED RESULTS TO COMPARE AGAINST
rq4 = pd.read_csv(OUT_DIR / "rq4_fold_metrics.csv")
base = rq4[(rq4["condition"] == "A_lodo") & (rq4["test_set"] == "full_day") & (rq4["model"] == "hgb")]
chosen = base.set_index("day")["params"].to_dict()
rq4c = pd.read_csv(OUT_DIR / "rq4_per_class.csv")

rows = []
for d in ATTACK_DAYS:
    train_idx, test_idx = np.flatnonzero(day != d), np.flatnonzero(day == d)
    # SAME FIT/CALIBRATION SPLIT AS RQ4 (IT USES THE FIXED PROJECT SEED), SO ONLY THE MODEL SEED VARIES
    fit_i, cal_i = split_fit_calib(train_idx, labels)
    # RECORD THE ORIGINAL RQ4 RUN AS THE FIRST ROW FOR THIS DAY
    b = base[base["day"] == d].iloc[0]
    row = {"day": d, "seed": SEED, "recall": b["recall"], "fpr": b["fpr"], "pr_auc": b["pr_auc"]}
    pc = rq4c[(rq4c["day"] == d) & (rq4c["model"] == "hgb") & (rq4c["condition"] == "A_lodo")
              & (rq4c["test_set"] == "full_day")]
    row.update({f"recall {r.attack_class}": r.recall for r in pc.itertuples()})
    rows.append(row)
    for s in NEW_SEEDS:
        # SAME MODEL AND SETTINGS, DIFFERENT `random_state`
        model = make_model("hgb", **json.loads(chosen[d]))
        model.set_params(random_state=s)
        model.fit(X[fit_i], y[fit_i])
        thr = pick_threshold(model.predict_proba(X[cal_i[y[cal_i] == 0]])[:, 1])
        m, per = evaluate(y[test_idx], model.predict_proba(X[test_idx])[:, 1], thr, labels[test_idx])
        row = {"day": d, "seed": s, "recall": m["recall"], "fpr": m["fpr"], "pr_auc": m["pr_auc"]}
        row.update({f"recall {k}": v["recall"] for k, v in per.items()})
        rows.append(row)
        print(f"{d} seed {s}: recall {m['recall']:.3f} fpr {m['fpr']:.4f} pr_auc {m['pr_auc']:.3f}",
              flush=True)
    # CHECKPOINT AFTER EVERY DAY
    pd.DataFrame(rows).to_csv(OUT_DIR / "m3_stability.csv", index=False)
