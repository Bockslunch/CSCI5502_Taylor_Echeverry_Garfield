"""RQ5: how much do per-class results change when defective flows are excluded?

Usage (from the repository root, after m3_rq4_transfer.py):
    python src/m3_rq5_defects.py data/processed results/m3

Three defect flags (definitions in m3_common.defect_flags, matching Milestone 2):
zero-payload web-attack flows, flows cut at the 120 s timeout, and feature
vectors that also occur under a different label.

For each held-out day and each candidate model, with the hyperparameters that
RQ4's inner loop chose for that day:
  retained  model trained on all training rows (the RQ4 condition-A model; its
            saved scores are reused), scored on all test rows and on clean test rows
  excluded  model retrained on clean training rows only, scored on clean test rows
Comparing "retained / clean test" with "excluded / clean test" isolates the effect
of the defects on *training*; comparing "retained / all test" with "retained /
clean test" isolates their effect on *measurement*.

Outputs: rq5_defect_counts.csv, rq5_fold_metrics.csv, rq5_per_class.csv, rq5_run_log.txt
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, TIMING, PATHS AND JSON
import json
import sys
import time
from pathlib import Path

# IMPORT THE NUMERIC STACK
import numpy as np
import pandas as pd

# MAKE `src/` IMPORTABLE AND LOAD THE SHARED PIECES (SAME FEATURES, MODELS, THRESHOLD RULE, METRICS)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import (DAYS, defect_flags, evaluate, feature_list, load_clean, make_model,
                       pick_threshold, split_fit_calib)

# READ INPUT AND OUTPUT FOLDERS FROM THE COMMAND LINE
DATA_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
LOG = open(OUT_DIR / "rq5_run_log.txt", "w")


# PRINT A TIMESTAMPED LINE TO THE SCREEN AND THE LOG FILE
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# --- LOAD DATA AND FLAG DEFECTS ---------------------------------------------------

# LOAD THE SAME DATA AND FEATURES AS RQ4
df = load_clean(DATA_DIR)
FEATS = feature_list(df)

# FLAG DEFECTIVE ROWS USING THE MODELING FEATURES (A VECTOR SHARED ACROSS LABELS IS ONLY A PROBLEM
# IF THE MODEL SEES IDENTICAL INPUTS, SO THE 64 MODEL FEATURES ARE THE RIGHT COLUMNS TO COMPARE)
flags = defect_flags(df, FEATS)

# COUNT EACH DEFECT BY CLASS SO THE REPORT CAN SAY WHICH CLASSES LOSE THE MOST ROWS
counts = flags.groupby(df["Label"]).sum()
counts.insert(0, "rows", df.groupby("Label").size())
counts["clean_rows"] = counts["rows"] - counts["any_defect"]
counts["pct_defective"] = (counts["any_defect"] / counts["rows"] * 100).round(2)
counts.to_csv(OUT_DIR / "rq5_defect_counts.csv")
log("defects by class:\n" + counts.to_string())

# BUILD THE ARRAYS THE MODELS NEED, THEN FREE THE FRAME
X = df[FEATS].to_numpy(np.float32)
y = df["is_attack"].to_numpy()
labels = df["Label"].to_numpy()
day = df["day"].to_numpy()
clean = ~flags["any_defect"].to_numpy()
del df, flags

# --- REUSE RQ4'S CHOSEN SETTINGS AND HELD-OUT SCORES -------------------------------

# THE HYPERPARAMETERS THE INNER LOOP PICKED FOR EACH (MODEL, DAY), FROM THE RQ4 RESULTS FILE
rq4 = pd.read_csv(OUT_DIR / "rq4_fold_metrics.csv")
chosen = (rq4[(rq4["condition"] == "A_lodo") & (rq4["test_set"] == "full_day")
              & rq4["model"].isin(["logreg", "hgb"])]
          .set_index(["model", "day"])["params"].to_dict())

# THE CONDITION-A SCORES AND THRESHOLDS SAVED BY RQ4 FOR EVERY HELD-OUT DAY
saved = np.load(OUT_DIR / "rq4_scores.npz")

# COLLECT RESULTS ACROSS ALL FOLDS
fold_rows, class_rows = [], []


# STORE ONE (MODEL, DAY, SETTING, TEST SET) RESULT AND ITS PER-CLASS RECALLS
def record(d, model, setting, test_set, idx, scores, thr):
    m, pc = evaluate(y[idx], scores, thr, labels[idx])
    fold_rows.append({"day": d, "model": model, "training": setting, "test_set": test_set,
                      "threshold": thr, **m})
    for lab, v in pc.items():
        class_rows.append({"day": d, "model": model, "training": setting, "test_set": test_set,
                           "attack_class": lab, **v})


# --- MAIN LOOP OVER HELD-OUT DAYS ----------------------------------------------------

for d in DAYS:
    test_idx = saved[f"idx__{d}"]
    # CONFIRM THE SAVED TEST ROWS ARE EXACTLY THIS DAY'S ROWS (GUARDS AGAINST A STALE RQ4 FILE)
    assert np.array_equal(test_idx, np.flatnonzero(day == d)), f"stale rq4_scores.npz for {d}"
    clean_test = clean[test_idx]
    train_idx = np.flatnonzero((day != d) & clean)
    log(f"=== {d}: {int((~clean_test).sum()):,} of {len(test_idx):,} test rows defective; "
        f"clean training rows {len(train_idx):,}")

    for name in ["logreg", "hgb"]:
        # RETAINED: REUSE RQ4'S CONDITION-A SCORES, EVALUATED ON ALL TEST ROWS AND ON CLEAN ONES
        p_ret, thr_ret = saved[f"{name}__{d}"], float(saved[f"{name}__{d}__thr"][0])
        record(d, name, "retained", "all_test", test_idx, p_ret, thr_ret)
        record(d, name, "retained", "clean_test", test_idx[clean_test], p_ret[clean_test], thr_ret)

        # EXCLUDED: REFIT WITH THE SAME HYPERPARAMETERS ON CLEAN TRAINING ROWS ONLY; THRESHOLD FROM
        # CLEAN BENIGN CALIBRATION ROWS; SCORE THE CLEAN TEST ROWS
        params = json.loads(chosen[(name, d)])
        fit_i, cal_i = split_fit_calib(train_idx, labels)
        ts = time.time()
        model = make_model(name, **params).fit(X[fit_i], y[fit_i])
        thr = pick_threshold(model.predict_proba(X[cal_i[y[cal_i] == 0]])[:, 1])
        p_exc = model.predict_proba(X[test_idx[clean_test]])[:, 1]
        record(d, name, "excluded", "clean_test", test_idx[clean_test], p_exc, thr)
        log(f"  {name}: refit {time.time() - ts:.0f}s")

    # CHECKPOINT AFTER EVERY DAY
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "rq5_fold_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "rq5_per_class.csv", index=False)

# SUMMARY ON SCREEN: ATTACK RECALL AND FALSE-ALARM RATE UNDER EACH SETTING
res = pd.DataFrame(fold_rows)
log("\n" + res.pivot_table(index=["model", "training", "test_set"], columns="day",
                           values=["recall", "fpr"]).round(3).to_string())
LOG.close()
