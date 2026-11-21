"""Milestone 4: does adding the anomaly detector to a classifier catch more unseen attacks at the same alert budget?

Usage (from the repository root, after m4_anomaly.py and m4_importance.py):
    python src/m4_hybrid.py data/processed results/m4

Why: m4_anomaly.py shows the Isolation Forest is weaker overall than the
classifiers but catches different attacks (slow-rate DoS, Infiltration) that
the classifiers mostly miss. A security team would run both and raise an alert
when EITHER fires. The fair question is whether that helps when both together
are held to the same 1% benign alert budget as one model alone.

Rule (fixed before looking at test days): the two detectors share the budget
equally. On the benign calibration flows of the training days, find the single
per-model quantile q so that "classifier above its q-quantile OR forest above
its q-quantile" flags 1% of those flows. Both models are fit on the same fit
rows (the forest on their benign part) and calibrated on the same benign rows,
so the union's false-alarm rate is measured, not guessed.

Hybrids: gradient boosting + Isolation Forest, random forest + Isolation Forest.

Outputs (results/m4): m4_hybrid_fold_metrics.csv, m4_hybrid_per_class.csv,
m4_hybrid_run_log.txt.
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, TIMING, PATHS AND JSON
import json
import sys
import time
from pathlib import Path

# IMPORT THE NUMERIC STACK, THE ANOMALY DETECTOR AND THE SCORING HELPERS
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# MAKE `src/` IMPORTABLE WHEN RUN FROM THE REPOSITORY ROOT, THEN LOAD THE SHARED MILESTONE 3 PIECES
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import (DAYS, SEED, TARGET_FPR, evaluate, feature_list, load_clean,
                       signed_log1p, split_fit_calib)

# READ THE DATA AND MILESTONE 4 RESULTS FOLDERS FROM THE COMMAND LINE
DATA_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])

# SAME NUMBER OF TREES AS m4_anomaly.py
N_TREES = 200

# A RUN LOG ON DISK
LOG = open(OUT_DIR / "m4_hybrid_run_log.txt", "w")


# PRINT A TIMESTAMPED LINE TO THE SCREEN AND THE LOG FILE
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# --- LOAD DATA, SAVED CLASSIFIER SCORES AND CHOSEN DETECTOR SETTINGS -------------------------

# LOAD THE CLEANED DATA; THE FOREST USES THE SIGNED-LOG FEATURES, AS IN m4_anomaly.py
df = load_clean(DATA_DIR)
FEATS = feature_list(df)
XL = signed_log1p(df[FEATS].to_numpy(np.float32)).astype(np.float32)
y = df["is_attack"].to_numpy()
labels = df["Label"].to_numpy()
day = df["day"].to_numpy()
del df

# HELD-OUT-DAY SCORES OF THE TWO CLASSIFIERS, WRITTEN BY m4_importance.py
sup = np.load(OUT_DIR / "m4_supervised_scores.npz")

# THE ISOLATION-FOREST SETTING m4_anomaly.py's INNER LOOP CHOSE FOR EACH HELD-OUT DAY
am = pd.read_csv(OUT_DIR / "m4_anomaly_fold_metrics.csv")
am = am[(am["model"] == "iforest") & (am["condition"] == "A_lodo") & (am["test_set"] == "full_day")]
IF_PARAMS = {r.day: json.loads(r.params) for r in am.drop_duplicates("day").itertuples()}


# FIND THE SHARED QUANTILE q: THE UNION OF "A > its q-QUANTILE" AND "B > its q-QUANTILE" SHOULD FLAG
# TARGET_FPR OF THE BENIGN CALIBRATION FLOWS. THE UNION RATE GROWS WITH 1 - q, SO BISECTION WORKS
def shared_thresholds(a_cal, b_cal, target=TARGET_FPR):
    lo, hi = 0.0, target
    for _ in range(40):
        mid = (lo + hi) / 2
        ta, tb = np.quantile(a_cal, 1 - mid), np.quantile(b_cal, 1 - mid)
        rate = ((a_cal > ta) | (b_cal > tb)).mean()
        lo, hi = (mid, hi) if rate < target else (lo, mid)
    return float(np.quantile(a_cal, 1 - lo)), float(np.quantile(b_cal, 1 - lo)), lo


# --- MAIN LOOP OVER HELD-OUT DAYS ------------------------------------------------------------

fold_rows, class_rows = [], []
for d in DAYS:
    t0 = time.time()
    train_idx = np.flatnonzero(day != d)
    test_idx = sup[f"idx__{d}"]
    # REBUILD THE SAME 80/20 FIT/CALIBRATION SPLIT THE CLASSIFIERS USED (SAME FUNCTION, SAME SEED)
    fit_i, cal_i = split_fit_calib(train_idx, labels)
    cal_ben = cal_i[y[cal_i] == 0]
    # FIT THE ISOLATION FOREST ON THE BENIGN PART OF THE CLASSIFIERS' FIT ROWS, SO NEITHER MODEL HAS
    # SEEN THE CALIBRATION FLOWS THAT SET THE SHARED THRESHOLD
    p = IF_PARAMS[d]
    iso = IsolationForest(n_estimators=N_TREES, max_samples=p["max_samples"],
                          max_features=p["max_features"], random_state=SEED, n_jobs=-1)
    iso.fit(XL[fit_i[y[fit_i] == 0]])
    if_cal = -iso.score_samples(XL[cal_ben])
    if_test = -iso.score_samples(XL[test_idx])
    for clf in ["hgb", "rf"]:
        clf_cal, clf_test = sup[f"{clf}__{d}__calben"], sup[f"{clf}__{d}"]
        # THE CALIBRATION SCORES MUST LINE UP ROW FOR ROW WITH THE FOREST'S (SAME ROWS, SAME ORDER)
        assert len(clf_cal) == len(if_cal)
        t_clf, t_if, q = shared_thresholds(clf_cal, if_cal)
        # AN ALERT IS RAISED WHEN EITHER MODEL FIRES; EXPRESS IT AS A 0/1 SCORE WITH THRESHOLD 0.5.
        # PR-AUC FOR THE HYBRID IS NOT REPORTED: A 0/1 SCORE HAS NO RANKING
        fire_clf, fire_if = clf_test > t_clf, if_test > t_if
        alert = (fire_clf | fire_if).astype(float)
        m, pc = evaluate(y[test_idx], alert, 0.5, labels[test_idx])
        m["pr_auc"] = m["roc_auc"] = np.nan
        # HOW MANY CAUGHT ATTACKS CAME ONLY FROM THE ANOMALY DETECTOR (ITS ADDED VALUE)
        att = y[test_idx] == 1
        extra = {"per_model_budget": q, "calib_union_fpr": float(
                     ((clf_cal > t_clf) | (if_cal > t_if)).mean()),
                 "caught_by_clf_only": int((fire_clf & ~fire_if & att).sum()),
                 "caught_by_if_only": int((fire_if & ~fire_clf & att).sum()),
                 "caught_by_both": int((fire_if & fire_clf & att).sum())}
        name = f"{clf}+iforest"
        fold_rows.append({"day": d, "model": name, **m, **extra})
        for lab, v in pc.items():
            class_rows.append({"day": d, "model": name, "attack_class": lab, **v})
        log(f"{d} {name}: per-model budget {q:.4f}, recall {m['recall']:.3f}, "
            f"fpr {m['fpr']:.4f}, IF-only catches {extra['caught_by_if_only']:,}")
    log(f"=== {d} done in {time.time() - t0:.0f}s")

# SAVE RESULTS
pd.DataFrame(fold_rows).to_csv(OUT_DIR / "m4_hybrid_fold_metrics.csv", index=False)
pd.DataFrame(class_rows).to_csv(OUT_DIR / "m4_hybrid_per_class.csv", index=False)
LOG.close()
