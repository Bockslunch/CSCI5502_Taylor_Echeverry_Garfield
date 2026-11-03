"""RQ2: which flow features separate attack categories, and do the separations
survive when artifact-prone features and timeout-capped flows are removed?

Usage (from the repository root, after m3_rq4_transfer.py):
    python src/m3_rq2_separability.py data/processed results/m3

Part 1 - multi-class separability among the 11 attack classes with >= 600 flows.
  Attack rows only, each class capped at 20,000 rows (random, seeded) so DoS Hulk,
  DDoS and PortScan do not swamp the small classes. Stratified 5-fold
  cross-validation. Variants:
    V1 all 64 model features
    V2 without Init_Win_bytes_forward / _backward
    V3 V2 and flows at the 120 s timeout removed
  Baselines: majority class and stratified guessing. Metric: macro-F1 (each class
  counts equally, which is the point of a "which classes separate" question) and
  per-class F1. Permutation importance on held-out folds for V1 and V2.

Part 2 - does RQ4's benign-vs-attack transfer depend on the artifact features?
  The RQ4 condition-A gradient-boosting model, refit without the two window
  fields, scored on each full held-out day.

Outputs: rq2_cv_metrics.csv, rq2_per_class_f1.csv, rq2_confusion_V2.csv,
rq2_importance.csv, rq2_transfer_no_artifacts.csv, rq2_run_log.txt
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, TIMING, PATHS AND JSON
import json
import sys
import time
from pathlib import Path

# IMPORT THE NUMERIC STACK AND THE SCIKIT-LEARN PIECES USED HERE
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

# MAKE `src/` IMPORTABLE AND LOAD THE SHARED PIECES
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import (ARTIFACT_FEATURES, DAYS, SEED, TIMEOUT_US, evaluate, feature_list,
                       load_clean, make_model, pick_threshold, split_fit_calib)

# READ INPUT AND OUTPUT FOLDERS FROM THE COMMAND LINE
DATA_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
LOG = open(OUT_DIR / "rq2_run_log.txt", "w")

# CLASSES NEED AT LEAST 600 FLOWS TO BE COMPARED (MILESTONE 2 RQ2 WORDING: 11 ATTACK CLASSES);
# EACH CLASS IS CAPPED AT 20,000 ROWS SO THE THREE LARGEST CLASSES DO NOT DOMINATE TRAINING
MIN_ROWS, CAP = 600, 20_000


# PRINT A TIMESTAMPED LINE TO THE SCREEN AND THE LOG FILE
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# --- LOAD DATA -----------------------------------------------------------------------

# LOAD THE SAME CLEANED DATA AND FEATURE LISTS AS THE OTHER MILESTONE 3 SCRIPTS
df = load_clean(DATA_DIR)
FEATS_ALL = feature_list(df)
FEATS_NOART = feature_list(df, drop_artifacts=True)

# =====================================================================================
# PART 1: MULTI-CLASS SEPARABILITY AMONG ATTACK CLASSES
# =====================================================================================

# KEEP ATTACK ROWS OF CLASSES WITH ENOUGH FLOWS, THEN CAP EACH CLASS AT 20,000 RANDOM ROWS
atk = df[df["is_attack"] == 1]
sizes = atk["Label"].value_counts()
keep_classes = sorted(sizes[sizes >= MIN_ROWS].index)
atk = atk[atk["Label"].isin(keep_classes)]
# SHUFFLE ONCE (SEEDED), THEN TAKE THE FIRST 20,000 ROWS OF EACH CLASS = A RANDOM SAMPLE PER CLASS.
# (SHUFFLE-THEN-HEAD INSTEAD OF groupby().apply(): PANDAS 3 DROPS THE GROUPING COLUMN INSIDE apply)
atk = atk.sample(frac=1.0, random_state=SEED).groupby("Label").head(CAP)
log(f"part 1: {len(atk):,} attack rows, {len(keep_classes)} classes: "
    + ", ".join(f"{c} ({n:,})" for c, n in atk["Label"].value_counts().items()))

# THE THREE FEATURE/ROW VARIANTS; V3 DROPS TIMEOUT-CAPPED ROWS AS WELL AS THE WINDOW FIELDS
not_capped = atk["Flow Duration"] < TIMEOUT_US
VARIANTS = {"V1_all_features": (FEATS_ALL, np.ones(len(atk), bool)),
            "V2_no_init_win": (FEATS_NOART, np.ones(len(atk), bool)),
            "V3_no_init_win_no_timeout_rows": (FEATS_NOART, not_capped.to_numpy())}

# FIVE STRATIFIED FOLDS: EVERY CLASS APPEARS IN EVERY TEST FOLD IN ITS NATURAL (CAPPED) PROPORTION
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# COLLECT RESULTS
cv_rows, f1_rows, imp_rows = [], [], []

# RUN EACH VARIANT THROUGH THE SAME FOLDS
for vname, (feats, row_mask) in VARIANTS.items():
    Xv = atk.loc[row_mask, feats].to_numpy(np.float32)
    yv = atk.loc[row_mask, "Label"].to_numpy()
    preds = np.empty(len(yv), dtype=object)
    for k, (tr, te) in enumerate(skf.split(Xv, yv)):
        # BASELINES ON THE SAME FOLD: ALWAYS THE MOST COMMON CLASS, OR RANDOM IN PROPORTION
        for bname, strat in [("majority", "most_frequent"), ("stratified", "stratified")]:
            b = DummyClassifier(strategy=strat, random_state=SEED).fit(Xv[tr], yv[tr])
            cv_rows.append({"variant": vname, "model": bname, "fold": k,
                            "macro_f1": f1_score(yv[te], b.predict(Xv[te]), average="macro")})
        # GRADIENT BOOSTING WITH DEFAULT SETTINGS: THIS IS A SEPARABILITY QUESTION, SO A FIXED,
        # REASONABLE MODEL IS USED RATHER THAN ONE TUNED TO SQUEEZE OUT EXTRA ACCURACY
        m = HistGradientBoostingClassifier(max_iter=200, early_stopping=True,
                                           validation_fraction=0.1, random_state=SEED)
        m.fit(Xv[tr], yv[tr])
        preds[te] = m.predict(Xv[te])
        cv_rows.append({"variant": vname, "model": "hgb", "fold": k,
                        "macro_f1": f1_score(yv[te], preds[te], average="macro")})
        # PERMUTATION IMPORTANCE ON THE FIRST HELD-OUT FOLD ONLY (V1 AND V2): HOW MUCH MACRO-F1
        # DROPS WHEN ONE FEATURE'S VALUES ARE SHUFFLED. CORRELATED FEATURES SHARE CREDIT, SO A LOW
        # SCORE MEANS "REPLACEABLE", NOT NECESSARILY "UNINFORMATIVE"
        if k == 0 and vname != "V3_no_init_win_no_timeout_rows":
            pi = permutation_importance(m, Xv[te], yv[te], scoring="f1_macro", n_repeats=3,
                                        random_state=SEED, n_jobs=1)
            for f, mu, sd in zip(feats, pi.importances_mean, pi.importances_std):
                imp_rows.append({"variant": vname, "feature": f, "importance": mu, "std": sd})
    # PER-CLASS F1 FROM THE OUT-OF-FOLD PREDICTIONS (EVERY ROW PREDICTED EXACTLY ONCE)
    per = f1_score(yv, preds, average=None, labels=keep_classes)
    for c, v in zip(keep_classes, per):
        f1_rows.append({"variant": vname, "attack_class": c, "f1": v,
                        "n": int((yv == c).sum())})
    # THE CONFUSION MATRIX FOR THE MAIN ARTIFACT-FREE VARIANT (ROWS = TRUE CLASS, ROW-NORMALIZED)
    if vname == "V2_no_init_win":
        cm = confusion_matrix(yv, preds, labels=keep_classes, normalize="true")
        pd.DataFrame(cm, index=keep_classes, columns=keep_classes).to_csv(
            OUT_DIR / "rq2_confusion_V2.csv")
    log(f"  {vname}: macro-F1 {f1_score(yv, preds, average='macro'):.3f} on {len(yv):,} rows")

# SAVE PART 1 RESULTS
pd.DataFrame(cv_rows).to_csv(OUT_DIR / "rq2_cv_metrics.csv", index=False)
pd.DataFrame(f1_rows).to_csv(OUT_DIR / "rq2_per_class_f1.csv", index=False)
pd.DataFrame(imp_rows).to_csv(OUT_DIR / "rq2_importance.csv", index=False)
del atk

# =====================================================================================
# PART 2: BENIGN-VS-ATTACK TRANSFER WITHOUT THE ARTIFACT FEATURES
# =====================================================================================

# REBUILD THE FULL-DATA ARRAYS WITHOUT THE TWO WINDOW FIELDS
X = df[FEATS_NOART].to_numpy(np.float32)
y = df["is_attack"].to_numpy()
labels = df["Label"].to_numpy()
day = df["day"].to_numpy()
del df

# REUSE THE GRADIENT-BOOSTING SETTINGS RQ4'S INNER LOOP CHOSE FOR EACH HELD-OUT DAY
rq4 = pd.read_csv(OUT_DIR / "rq4_fold_metrics.csv")
chosen = (rq4[(rq4["condition"] == "A_lodo") & (rq4["test_set"] == "full_day")
              & (rq4["model"] == "hgb")].set_index("day")["params"].to_dict())

# REFIT ON EACH SET OF FOUR TRAINING DAYS AND SCORE THE FULL HELD-OUT DAY, EXACTLY AS IN RQ4
rows = []
for d in DAYS:
    train_idx, test_idx = np.flatnonzero(day != d), np.flatnonzero(day == d)
    fit_i, cal_i = split_fit_calib(train_idx, labels)
    model = make_model("hgb", **json.loads(chosen[d])).fit(X[fit_i], y[fit_i])
    thr = pick_threshold(model.predict_proba(X[cal_i[y[cal_i] == 0]])[:, 1])
    m, _ = evaluate(y[test_idx], model.predict_proba(X[test_idx])[:, 1], thr)
    rows.append({"day": d, "model": "hgb_no_init_win", **m})
    log(f"  part 2 {d}: recall {m['recall']:.3f} fpr {m['fpr']:.4f} pr_auc {m['pr_auc']:.3f}")
pd.DataFrame(rows).to_csv(OUT_DIR / "rq2_transfer_no_artifacts.csv", index=False)
LOG.close()
