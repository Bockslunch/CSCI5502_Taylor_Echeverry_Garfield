"""RQ4: does a benign-vs-malicious model transfer to an unseen capture day?

Usage (from the repository root):
    python src/m3_rq4_transfer.py data/processed results/m3

Design (all five days, leave-one-day-out):
  For each held-out day d the other four days are the training data. Every
  hyperparameter choice is made by an *inner* leave-one-day-out loop over the
  training days only, and every alert threshold is set on a calibration split of
  the training days only, so day d influences nothing until it is scored.

  Models: two baselines (majority, stratified), a single-feature rule, logistic
  regression, histogram gradient boosting.

  To answer RQ4's second half ("how much of any drop comes from shifted benign
  traffic rather than from the new attacks?") the day is split in half, stratified
  by class. Half T is the common test set for three training conditions:
    A  lodo          other four days only                (benign shift + unseen attacks)
    B  lodo+benign   A plus the *benign* flows of the other half of d
                                                          (benign shift removed, attacks still unseen)
    C  in-day        A plus *all* flows of the other half of d
                                                          (attacks seen too: in-distribution ceiling)
  A->B isolates the benign-shift effect; B->C isolates the unseen-attack effect.

Outputs in the output folder: rq4_fold_metrics.csv, rq4_per_class.csv,
rq4_tuning.csv, rq4_features_used.json, rq4_scores.npz (held-out-day scores for
error analysis), rq4_run_log.txt.
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, TIMING, PATHS AND JSON
import json
import sys
import time
from pathlib import Path

# IMPORT THE NUMERIC STACK AND THE SCORING HELPERS
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# MAKE `src/` IMPORTABLE WHEN THE SCRIPT IS RUN FROM THE REPOSITORY ROOT, THEN LOAD THE SHARED PIECES
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import (DAYS, EXCLUDED, SEED, TARGET_FPR, evaluate, feature_list, load_clean,
                       make_model, pick_threshold, split_fit_calib)

# READ INPUT AND OUTPUT FOLDERS FROM THE COMMAND LINE (NO MACHINE-SPECIFIC PATHS)
DATA_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
OUT_DIR.mkdir(parents=True, exist_ok=True)

# THE SMALL HYPERPARAMETER GRIDS SEARCHED BY THE INNER LOOP (MANUAL GRID, 3 SETTINGS EACH).
# LOGREG: C IS THE INVERSE REGULARIZATION STRENGTH; SMALLER C = SIMPLER, MORE STABLE WEIGHTS.
# HGB: TREE SIZE (max_leaf_nodes) AND STEP SIZE (learning_rate) CONTROL HOW CLOSELY THE TREES CAN
# FIT THE TRAINING DAYS; THE THIRD SETTING ADDS L2 SHRINKAGE AS A MORE CONSERVATIVE OPTION.
GRIDS = {
    "logreg": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
    "hgb": [{"learning_rate": 0.1, "max_leaf_nodes": 15},
            {"learning_rate": 0.1, "max_leaf_nodes": 63},
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0}],
}

# DURING TUNING ONLY, KEEP 20% OF BENIGN TRAINING ROWS (ALL ATTACK ROWS) SO 24 INNER FITS PER
# HELD-OUT DAY FIT IN THE TIME BUDGET; FINAL MODELS ARE REFIT ON ALL TRAINING ROWS
TUNE_BENIGN_FRAC = 0.2

# A RUN LOG ON DISK, SO A LONG BACKGROUND RUN CAN BE FOLLOWED AND AUDITED AFTERWARDS
LOG = open(OUT_DIR / "rq4_run_log.txt", "w")


# PRINT A TIMESTAMPED LINE TO THE SCREEN AND THE LOG FILE
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# --- LOAD DATA ---------------------------------------------------------------------

# LOAD THE CLEANED DATA AND BUILD THE FEATURE MATRIX ONCE AS FLOAT32 (HALF THE RAM OF FLOAT64)
df = load_clean(DATA_DIR)
FEATS = feature_list(df)
X = df[FEATS].to_numpy(np.float32)
y = df["is_attack"].to_numpy()
labels = df["Label"].to_numpy()
day = df["day"].to_numpy()
del df
log(f"loaded {X.shape[0]:,} rows x {X.shape[1]} features")

# RECORD EXACTLY WHICH FEATURES WERE USED AND WHICH WERE EXCLUDED, FOR THE WRITE-UP
(OUT_DIR / "rq4_features_used.json").write_text(json.dumps(
    {"features": FEATS, "excluded": EXCLUDED, "target_fpr": TARGET_FPR, "seed": SEED}, indent=2))

# ONE RANDOM GENERATOR FOR ALL SUBSAMPLING IN THIS SCRIPT
rng = np.random.default_rng(SEED)


# --- HELPERS -----------------------------------------------------------------------

# FIT A MODEL AND RETURN ITS ATTACK-PROBABILITY SCORES ON THE GIVEN ROWS
def fit_score(name, params, fit_idx, score_idx_list):
    """Fit on fit_idx, return the fitted model and a score array for each index set."""
    model = make_model(name, **params)
    model.fit(X[fit_idx], y[fit_idx])
    return model, [model.predict_proba(X[s])[:, 1] for s in score_idx_list]


# THE SINGLE-FEATURE RULE: PICK THE ONE FEATURE THAT BEST SEPARATES BENIGN FROM ATTACK ON THE
# FIT ROWS (ROC-AUC FURTHEST FROM 0.5), FLIP ITS SIGN IF ATTACKS SIT LOW, AND USE IT AS THE SCORE.
# THIS IS THE "WHAT DOES ONE THRESHOLD ON ONE COLUMN GET YOU?" REFERENCE FOR THE TWO REAL MODELS.
def single_feature_rule(fit_idx):
    """Return (feature index, sign) chosen on fit rows only (200k-row sample for speed)."""
    s = rng.choice(fit_idx, size=min(200_000, len(fit_idx)), replace=False)
    best, best_gap, best_sign = 0, -1.0, 1.0
    for j in range(X.shape[1]):
        auc = roc_auc_score(y[s], X[s, j])
        if abs(auc - 0.5) > best_gap:
            best, best_gap, best_sign = j, abs(auc - 0.5), (1.0 if auc >= 0.5 else -1.0)
    return best, best_sign


# INNER LEAVE-ONE-DAY-OUT TUNING ON THE TRAINING DAYS ONLY. EACH CANDIDATE SETTING IS SCORED BY
# ITS MEAN PR-AUC ON THE INNER HELD-OUT DAYS, WHICH MIMICS THE OUTER QUESTION (TRANSFER TO A
# NEW DAY) INSTEAD OF REWARDING A SETTING THAT ONLY FITS ATTACKS IT HAS ALREADY SEEN.
def tune(name, train_days, outer_day):
    """Return the best params for `name` plus a list of tuning records."""
    records = []
    for params in GRIDS[name]:
        scores = []
        for inner in train_days:
            # MONDAY HAS NO ATTACKS, SO IT CANNOT BE AN INNER TEST DAY (PR-AUC IS UNDEFINED)
            if inner == "Monday":
                continue
            tr = np.flatnonzero(np.isin(day, [d for d in train_days if d != inner]))
            te = np.flatnonzero(day == inner)
            # SUBSAMPLE BENIGN ROWS OF THE INNER TRAINING SET ONLY; KEEP EVERY ATTACK ROW
            keep = (y[tr] == 1) | (rng.random(len(tr)) < TUNE_BENIGN_FRAC)
            _, (p,) = fit_score(name, {**params, "max_iter": 300} if name == "logreg" else params,
                                tr[keep], [te])
            scores.append(average_precision_score(y[te], p))
            records.append({"outer_day": outer_day, "model": name, "params": json.dumps(params),
                            "inner_day": inner, "pr_auc": scores[-1]})
        log(f"  tune {name} {params}: inner PR-AUC mean {np.mean(scores):.3f}")
    # PICK THE SETTING WITH THE HIGHEST MEAN INNER PR-AUC (TIES GO TO THE EARLIER, SIMPLER ONE)
    means = [np.mean([r["pr_auc"] for r in records if r["params"] == json.dumps(p)])
             for p in GRIDS[name]]
    return GRIDS[name][int(np.argmax(means))], records


# --- MAIN LOOP OVER HELD-OUT DAYS ----------------------------------------------------

# COLLECT RESULTS ACROSS ALL FOLDS
fold_rows, class_rows, tune_rows = [], [], []
saved_scores = {}

# EACH CAPTURE DAY TAKES ONE TURN AS THE UNSEEN TEST DAY
for d in DAYS:
    t0 = time.time()
    train_days = [x for x in DAYS if x != d]
    train_idx = np.flatnonzero(day != d)
    test_idx = np.flatnonzero(day == d)
    log(f"=== held-out {d}: train {len(train_idx):,} rows, test {len(test_idx):,} rows "
        f"({int(y[test_idx].sum()):,} attacks)")

    # SPLIT THE HELD-OUT DAY IN HALF, STRATIFIED BY CLASS: `half_add` IS ONLY EVER ADDED TO TRAINING
    # IN CONDITIONS B/C; `half_test` IS THE COMMON TEST SET FOR THE A/B/C COMPARISON
    lab_d = labels[test_idx]
    vc = pd.Series(lab_d).value_counts()
    strat = np.where(pd.Series(lab_d).map(vc).values >= 2, lab_d, "other")
    half_add, half_test = train_test_split(test_idx, test_size=0.5, random_state=SEED, stratify=strat)

    # TRAINING SETS FOR THE THREE CONDITIONS (ROW INDICES INTO X)
    cond_train = {"A_lodo": train_idx,
                  "B_lodo_plus_day_benign": np.concatenate([train_idx, half_add[y[half_add] == 0]]),
                  "C_in_day": np.concatenate([train_idx, half_add])}
    # ON MONDAY THERE ARE NO ATTACKS TO ADD, SO B AND C WOULD BE IDENTICAL; RUN B ONLY
    if d == "Monday":
        cond_train.pop("C_in_day")

    # SCORE ONE MODEL'S RESULTS ON THE FULL HELD-OUT DAY AND/OR THE COMMON HALF, AND STORE THEM
    def record(model, cond, test_name, idx, scores, thr, params=""):
        m, pc = evaluate(y[idx], scores, thr, labels[idx])
        fold_rows.append({"day": d, "model": model, "condition": cond, "test_set": test_name,
                          "threshold": thr, "params": params, **m})
        for lab, v in pc.items():
            class_rows.append({"day": d, "model": model, "condition": cond, "test_set": test_name,
                               "attack_class": lab, **v})

    # --- BASELINES (CONDITION A ONLY: THEY HAVE NOTHING TO LEARN FROM EXTRA BENIGN ROWS) ---
    for name in ["majority", "stratified"]:
        model = make_model(name).fit(X[train_idx], y[train_idx])
        # DUMMIES OUTPUT 0/1 GUESSES, NOT A RANKING, SO THEY ARE SCORED AT THEIR OWN 0.5 CUT
        # (A PERCENTILE THRESHOLD ON A 0/1 SCORE WOULD SIMPLY SWITCH THEM OFF)
        p_full = model.predict_proba(X[test_idx])[:, 1]
        record(name, "A_lodo", "full_day", test_idx, p_full, 0.5)

    # SINGLE-FEATURE RULE, THRESHOLD FROM BENIGN CALIBRATION ROWS OF THE TRAINING DAYS
    fit_i, cal_i = split_fit_calib(train_idx, labels)
    j, sign = single_feature_rule(fit_i)
    thr = pick_threshold(sign * X[cal_i[y[cal_i] == 0], j])
    record("single_feature", "A_lodo", "full_day", test_idx, sign * X[test_idx, j], thr,
           params=json.dumps({"feature": FEATS[j], "sign": sign}))
    log(f"  single feature: {FEATS[j]} (sign {sign:+.0f})")

    # --- CANDIDATE MODELS ---
    for name in ["logreg", "hgb"]:
        # TUNE ON THE TRAINING DAYS ONLY, THEN USE THE SAME CHOSEN SETTING FOR ALL THREE CONDITIONS
        # (SO A, B AND C DIFFER ONLY IN THEIR TRAINING ROWS)
        best, recs = tune(name, train_days, d)
        tune_rows += recs
        log(f"  {name} chosen: {best}")
        for cond, tr in cond_train.items():
            # 80/20 FIT/CALIBRATION SPLIT OF THIS CONDITION'S TRAINING ROWS
            fit_i, cal_i = split_fit_calib(tr, labels)
            # CONDITION A IS ALSO SCORED ON THE WHOLE HELD-OUT DAY (THE HEADLINE RQ4 RESULT)
            targets = [cal_i, half_test] + ([test_idx] if cond == "A_lodo" else [])
            ts = time.time()
            model, scores = fit_score(name, best, fit_i, targets)
            p_cal, p_half = scores[0], scores[1]
            # THRESHOLD FROM BENIGN CALIBRATION SCORES ONLY (TARGET 1% FALSE ALARMS)
            thr = pick_threshold(p_cal[y[cal_i] == 0])
            record(name, cond, "half_day", half_test, p_half, thr, params=json.dumps(best))
            if cond == "A_lodo":
                p_full = scores[2]
                record(name, cond, "full_day", test_idx, p_full, thr, params=json.dumps(best))
                saved_scores[f"{name}__{d}"] = p_full.astype(np.float32)
                saved_scores[f"{name}__{d}__thr"] = np.array([thr])
            # NOTE WHETHER LOGISTIC REGRESSION'S SOLVER CONVERGED (REPORTED, NOT HIDDEN)
            extra = ""
            if name == "logreg":
                extra = f" n_iter={int(model[-1].n_iter_[0])}"
            log(f"  {name} {cond}: fit {time.time() - ts:.0f}s thr={thr:.4f}{extra}")

    # SAVE THE TEST-DAY ROW INDICES SO ERROR ANALYSIS CAN LINE SCORES UP WITH THE ORIGINAL ROWS
    saved_scores[f"idx__{d}"] = test_idx
    saved_scores[f"halftest__{d}"] = half_test

    # CHECKPOINT AFTER EVERY DAY SO A CRASH LOSES AT MOST ONE FOLD
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "rq4_fold_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "rq4_per_class.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(OUT_DIR / "rq4_tuning.csv", index=False)
    np.savez_compressed(OUT_DIR / "rq4_scores.npz", **saved_scores)
    log(f"=== {d} done in {(time.time() - t0) / 60:.1f} min")

# FINAL SUMMARY ON SCREEN: HEADLINE METRICS PER MODEL AND DAY ON THE FULL HELD-OUT DAY
res = pd.DataFrame(fold_rows)
head = res[res["test_set"] == "full_day"].pivot_table(
    index="model", columns="day", values=["recall", "fpr", "pr_auc"])
log("\n" + head.round(3).to_string())
LOG.close()
