"""Known-attack ceiling: the same models on a pooled, stratified random split.

Usage (from the repository root, after m3_rq4_transfer.py):
    python src/m3_pooled_split.py data/processed results/m3

RQ4's leave-one-day-out design always tests on attack types the model has never
seen. This script asks the easier, complementary question: how well do the same
models detect attack types they HAVE seen? All five days are pooled and split
80/20 at random, stratified by day x class, so every attack type (and every day's
benign traffic) appears in both training and test data.

Everything else matches RQ4: the same 64 features, the same models, the same
80/20 fit/calibration split of the training rows, the same 1% benign-calibration
alert threshold and the same metrics. Hyperparameters for logistic regression and
gradient boosting are chosen from the RQ4 grids by PR-AUC on the calibration rows
(training data only); the random forest uses Luis's fixed settings (notebook 05).

Caveat built into the design: a random split puts flows from the same scripted
attack session on both sides, so these numbers are an optimistic ceiling, not an
estimate of detection in deployment.

Outputs: m3_pooled_metrics.csv (overall, per day, and on defect-free test rows),
m3_pooled_per_class.csv, m3_pooled_tuning.csv, m3_pooled_run_log.txt
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, TIMING, PATHS AND JSON
import json
import sys
import time
from pathlib import Path

# IMPORT THE NUMERIC STACK AND THE TWO SCIKIT-LEARN HELPERS USED HERE
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# MAKE `src/` IMPORTABLE AND LOAD THE SHARED PIECES (SAME FEATURES, MODELS, THRESHOLD RULE, METRICS)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import (DAYS, SEED, defect_flags, evaluate, feature_list, load_clean, make_model,
                       pick_threshold, split_fit_calib)

# READ INPUT AND OUTPUT FOLDERS FROM THE COMMAND LINE
DATA_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = open(OUT_DIR / "m3_pooled_run_log.txt", "w")

# SAME HYPERPARAMETER GRIDS AS RQ4 (m3_rq4_transfer.py), SO THE TWO DESIGNS SEARCH THE SAME OPTIONS
GRIDS = {
    "logreg": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
    "hgb": [{"learning_rate": 0.1, "max_leaf_nodes": 15},
            {"learning_rate": 0.1, "max_leaf_nodes": 63},
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0}],
}

# DURING TUNING ONLY, KEEP 20% OF BENIGN FIT ROWS (ALL ATTACK ROWS), AS IN RQ4; FINAL FITS USE ALL ROWS
TUNE_BENIGN_FRAC = 0.2


# PRINT A TIMESTAMPED LINE TO THE SCREEN AND THE LOG FILE
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# --- LOAD DATA AND FLAG DEFECTS ---------------------------------------------------

# LOAD THE SAME CLEANED DATA AND FEATURES AS RQ4
df = load_clean(DATA_DIR)
FEATS = feature_list(df)

# FLAG DEFECTIVE ROWS (RQ5 DEFINITIONS) SO THE CEILING CAN ALSO BE REPORTED ON CLEAN TEST FLOWS
clean = ~defect_flags(df, FEATS)["any_defect"].to_numpy()

# BUILD THE ARRAYS THE MODELS NEED, THEN FREE THE FRAME
X = df[FEATS].to_numpy(np.float32)
y = df["is_attack"].to_numpy()
labels = df["Label"].to_numpy()
day = df["day"].to_numpy()
del df
log(f"loaded {X.shape[0]:,} rows x {X.shape[1]} features")

# --- THE POOLED 80/20 SPLIT --------------------------------------------------------

# STRATIFY BY DAY x CLASS ("Wednesday|DoS Hulk"), SO EVERY DAY'S BENIGN TRAFFIC AND EVERY ATTACK
# TYPE KEEPS ITS SHARE IN BOTH HALVES. EVERY STRATUM HAS AT LEAST 11 ROWS (HEARTBLEED), SO ALL SPLIT.
strata = np.char.add(np.char.add(day.astype(str), "|"), labels.astype(str))
all_idx = np.arange(len(y))
train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=SEED, stratify=strata)
log(f"pooled split: train {len(train_idx):,} rows, test {len(test_idx):,} rows "
    f"({int(y[test_idx].sum()):,} attacks)")

# THE SAME 80/20 FIT / CALIBRATION SPLIT OF THE TRAINING ROWS THAT RQ4 USES
fit_idx, cal_idx = split_fit_calib(train_idx, labels)

# ONE RANDOM GENERATOR FOR THE TUNING SUBSAMPLE
rng = np.random.default_rng(SEED)

# --- RESULT COLLECTION ---------------------------------------------------------------

metric_rows, class_rows, tune_rows = [], [], []


# STORE ONE MODEL'S RESULTS: ALL TEST ROWS, EACH DAY'S TEST ROWS, AND DEFECT-FREE TEST ROWS
def record(model, scores, thr, params=""):
    subsets = [("all", np.ones(len(test_idx), bool))]
    subsets += [(f"day:{d}", day[test_idx] == d) for d in DAYS]
    subsets += [("clean", clean[test_idx])]
    for name, m in subsets:
        idx = test_idx[m]
        met, per = evaluate(y[idx], scores[m], thr, labels[idx])
        metric_rows.append({"model": model, "subset": name, "threshold": thr, "params": params, **met})
        # PER-CLASS RECALL ONLY FOR THE ALL-ROWS AND CLEAN-ROWS SUBSETS (PER-DAY WOULD REPEAT IT)
        if name in ("all", "clean"):
            for lab, v in per.items():
                class_rows.append({"model": model, "subset": name, "attack_class": lab, **v})


# --- BASELINES ---------------------------------------------------------------------

# MAJORITY AND STRATIFIED DUMMIES, SCORED AT THEIR OWN 0.5 CUT AS IN RQ4
for name in ["majority", "stratified"]:
    model = make_model(name).fit(X[train_idx], y[train_idx])
    record(name, model.predict_proba(X[test_idx])[:, 1], 0.5)

# SINGLE-FEATURE RULE: THE FEATURE WITH ROC-AUC FURTHEST FROM 0.5 ON A 200K SAMPLE OF FIT ROWS
s = rng.choice(fit_idx, size=200_000, replace=False)
aucs = [roc_auc_score(y[s], X[s, j]) for j in range(X.shape[1])]
j = int(np.argmax(np.abs(np.array(aucs) - 0.5)))
sign = 1.0 if aucs[j] >= 0.5 else -1.0
thr = pick_threshold(sign * X[cal_idx[y[cal_idx] == 0], j])
record("single_feature", sign * X[test_idx, j], thr, json.dumps({"feature": FEATS[j], "sign": sign}))
log(f"single feature: {FEATS[j]} (sign {sign:+.0f})")

# --- CANDIDATE MODELS ------------------------------------------------------------------

for name in ["logreg", "hgb", "rf"]:
    # PICK HYPERPARAMETERS ON TRAINING DATA ONLY: FIT EACH GRID SETTING ON THE (SUBSAMPLED) FIT ROWS,
    # SCORE PR-AUC ON THE CALIBRATION ROWS. THE TEST ROWS ARE NOT TOUCHED UNTIL THE FINAL SCORE.
    # THE RANDOM FOREST KEEPS LUIS'S FIXED SETTINGS (HIS NOTEBOOK DID NOT TUNE IT EITHER).
    best = {}
    if name in GRIDS:
        keep = (y[fit_idx] == 1) | (rng.random(len(fit_idx)) < TUNE_BENIGN_FRAC)
        scores = []
        for params in GRIDS[name]:
            m = make_model(name, **params).fit(X[fit_idx[keep]], y[fit_idx[keep]])
            ap = average_precision_score(y[cal_idx], m.predict_proba(X[cal_idx])[:, 1])
            scores.append(ap)
            tune_rows.append({"model": name, "params": json.dumps(params), "cal_pr_auc": ap})
            log(f"  tune {name} {params}: calibration PR-AUC {ap:.4f}")
        best = GRIDS[name][int(np.argmax(scores))]
    # FINAL FIT ON ALL FIT ROWS, THRESHOLD FROM BENIGN CALIBRATION SCORES, THEN SCORE THE TEST ROWS
    ts = time.time()
    model = make_model(name, **best).fit(X[fit_idx], y[fit_idx])
    thr = pick_threshold(model.predict_proba(X[cal_idx[y[cal_idx] == 0]])[:, 1])
    record(name, model.predict_proba(X[test_idx])[:, 1], thr, json.dumps(best))
    log(f"{name} {best}: fit {time.time() - ts:.0f}s thr={thr:.4f}")
    # CHECKPOINT AFTER EVERY MODEL
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "m3_pooled_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "m3_pooled_per_class.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(OUT_DIR / "m3_pooled_tuning.csv", index=False)

# FINAL SAVE AND A SUMMARY ON SCREEN
pd.DataFrame(metric_rows).to_csv(OUT_DIR / "m3_pooled_metrics.csv", index=False)
pd.DataFrame(class_rows).to_csv(OUT_DIR / "m3_pooled_per_class.csv", index=False)
pd.DataFrame(tune_rows).to_csv(OUT_DIR / "m3_pooled_tuning.csv", index=False)
res = pd.DataFrame(metric_rows)
log("\n" + res[res["subset"].isin(["all", "clean"])].pivot_table(
    index="model", columns="subset", values=["recall", "fpr", "precision", "pr_auc"]).round(4).to_string())
LOG.close()
