"""Milestone 4: which features drive detection of UNSEEN attacks, and is the random forest's lead robust?

Usage (from the repository root):
    python src/m4_importance.py data/processed results/m3 results/m4

Three parts, all on the Milestone 3 leave-one-day-out folds:

1. Refit. Gradient boosting (with the hyperparameters Milestone 3 chose for each
   held-out day, read from results/m3/rq4_fold_metrics.csv) and the random forest
   (Luis's fixed settings) are refit on the four training days, thresholded at the
   1% benign alert budget, and scored on the held-out day. The refit recall is
   checked against Milestone 3 as a reproducibility test. Held-out and benign
   calibration scores are saved for m4_hybrid.py.

2. Permutation importance on the held-out day (instructor feedback on Milestone 3).
   Shuffle one feature (or one group of near-duplicate features) across the
   held-out flows and measure how much PR-AUC and recall at the fixed threshold
   drop. Unlike the trees' built-in importance, this is measured on traffic the
   model never saw, so it shows what actually carries detection to a new day.
   CICIDS2017 has many near-copies (e.g. mean packet size vs. average packet
   size); shuffling one copy leaves the other intact and makes both look
   useless, so features with |Spearman rho| >= 0.9 are also shuffled together.

3. Random forest robustness. The forest was run once per day with one seed and
   its Wednesday gain might come from the artifact-prone TCP window fields. It is
   rerun with seeds 1-3 and without Init_Win_bytes_forward/backward.

Outputs (results/m4): m4_refit_check.csv, m4_importance_single.csv,
m4_importance_groups.csv, m4_feature_groups.json, m4_rf_robustness.csv,
m4_rf_robustness_per_class.csv, m4_supervised_scores.npz (not committed),
m4_importance_run_log.txt.
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, TIMING, PATHS AND JSON
import json
import sys
import time
from pathlib import Path

# IMPORT THE NUMERIC STACK, CLUSTERING FOR FEATURE GROUPS, AND THE SCORING HELPERS
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# MAKE `src/` IMPORTABLE WHEN RUN FROM THE REPOSITORY ROOT, THEN LOAD THE SHARED MILESTONE 3 PIECES
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import (ARTIFACT_FEATURES, DAYS, SEED, evaluate, feature_list, load_clean,
                       make_model, pick_threshold, split_fit_calib)

# READ THE DATA, MILESTONE 3 RESULTS AND OUTPUT FOLDERS FROM THE COMMAND LINE
DATA_DIR, M3_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT_DIR.mkdir(parents=True, exist_ok=True)

# THE FOUR DAYS THAT CONTAIN ATTACKS (MONDAY IS ALL BENIGN, SO IT HAS NO RECALL OR IMPORTANCE)
ATTACK_DAYS = [d for d in DAYS if d != "Monday"]

# PERMUTATION IMPORTANCE IS COMPUTED ON A SAMPLE OF EACH HELD-OUT DAY: UP TO 30,000 ATTACK FLOWS AND
# 60,000 BENIGN FLOWS, SHUFFLED 3 TIMES PER FEATURE. 64 FEATURES x 3 REPEATS x 2 MODELS x 4 DAYS
# IS ~1,500 RESCORINGS, SO THE SAMPLE KEEPS THE RUN TO MINUTES WHILE ESTIMATES STAY STABLE
IMP_ATTACK_MAX, IMP_BENIGN_MAX, IMP_REPEATS = 30_000, 60_000, 3

# FEATURES WHOSE |SPEARMAN RHO| IS AT LEAST THIS HIGH ARE TREATED AS ONE GROUP FOR SHUFFLING
GROUP_RHO = 0.9

# EXTRA RANDOM-FOREST SEEDS FOR THE STABILITY CHECK (5502 IS THE MILESTONE 3 RUN ITSELF)
RF_SEEDS = [1, 2, 3]

# A RUN LOG ON DISK, SO A LONG BACKGROUND RUN CAN BE FOLLOWED AND AUDITED AFTERWARDS
LOG = open(OUT_DIR / "m4_importance_run_log.txt", "w")


# PRINT A TIMESTAMPED LINE TO THE SCREEN AND THE LOG FILE
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# --- LOAD DATA AND MILESTONE 3 CHOICES -------------------------------------------------

# LOAD THE CLEANED DATA AND BUILD THE RAW FEATURE MATRIX (THE CLASSIFIERS USE RAW VALUES; LOGISTIC
# REGRESSION'S LOG STEP LIVES INSIDE ITS PIPELINE AND IS NOT USED HERE)
df = load_clean(DATA_DIR)
FEATS = feature_list(df)
X = df[FEATS].to_numpy(np.float32)
y = df["is_attack"].to_numpy()
labels = df["Label"].to_numpy()
day = df["day"].to_numpy()
del df
log(f"loaded {X.shape[0]:,} rows x {X.shape[1]} features")

# ONE RANDOM GENERATOR FOR ALL SUBSAMPLING AND SHUFFLING IN THIS SCRIPT
rng = np.random.default_rng(SEED)

# READ MILESTONE 3'S HEADLINE RESULTS: THE CHOSEN GRADIENT-BOOSTING SETTINGS PER HELD-OUT DAY, AND
# THE RECALL / FALSE-ALARM RATE EACH MODEL GOT, SO THE REFIT CAN BE CHECKED AGAINST THEM
m3 = pd.read_csv(M3_DIR / "rq4_fold_metrics.csv")
m3 = m3[(m3["condition"] == "A_lodo") & (m3["test_set"] == "full_day")]
HGB_PARAMS = {r.day: json.loads(r.params) for r in m3[m3["model"] == "hgb"].itertuples()}
m3_forest = pd.read_csv(M3_DIR / "rq4_forest.csv") if (M3_DIR / "rq4_forest.csv").exists() else None


# --- FEATURE GROUPS -------------------------------------------------------------------

# CLUSTER FEATURES BY RANK CORRELATION ON A 200K-ROW SAMPLE OF ALL DAYS: DISTANCE = 1 - |RHO|,
# AVERAGE LINKAGE, CUT AT 1 - GROUP_RHO. CONSTANT COLUMNS GET RHO = 0 (THEY JOIN NO GROUP)
s = rng.choice(len(X), size=200_000, replace=False)
with np.errstate(invalid="ignore", divide="ignore"):
    rho = spearmanr(X[s]).correlation
rho = np.nan_to_num(rho, nan=0.0)
np.fill_diagonal(rho, 1.0)
dist = squareform(1 - np.abs(rho), checks=False)
cluster_id = fcluster(linkage(dist, method="average"), t=1 - GROUP_RHO, criterion="distance")
GROUPS = {}
for cid in np.unique(cluster_id):
    cols = list(np.flatnonzero(cluster_id == cid))
    # NAME EACH GROUP AFTER ITS FIRST FEATURE, PLUS "(+k)" WHEN IT HAS k NEAR-COPIES
    name = FEATS[cols[0]] + (f" (+{len(cols) - 1})" if len(cols) > 1 else "")
    GROUPS[name] = cols
(OUT_DIR / "m4_feature_groups.json").write_text(json.dumps(
    {g: [FEATS[c] for c in cols] for g, cols in GROUPS.items()}, indent=2))
log(f"{len(FEATS)} features form {len(GROUPS)} groups at |rho| >= {GROUP_RHO}")


# --- HELPERS --------------------------------------------------------------------------

# FIT A CLASSIFIER ON THE TRAINING DAYS (80% FIT / 20% CALIBRATION, AS IN MILESTONE 3) AND RETURN
# THE MODEL, ITS 1% THRESHOLD, AND THE BENIGN CALIBRATION SCORES
def fit_thresholded(name, params, train_idx, cols=None):
    cols = list(range(X.shape[1])) if cols is None else cols
    fit_i, cal_i = split_fit_calib(train_idx, labels)
    model = make_model(name, **params).fit(X[np.ix_(fit_i, cols)] if len(cols) < X.shape[1]
                                           else X[fit_i], y[fit_i])
    cal_ben = cal_i[y[cal_i] == 0]
    p_cal = model.predict_proba(X[cal_ben][:, cols])[:, 1]
    return model, pick_threshold(p_cal), p_cal


# SCORE ROWS WITH A FITTED CLASSIFIER (ATTACK PROBABILITY), OPTIONALLY ON A COLUMN SUBSET
def score(model, rows_X):
    return model.predict_proba(rows_X)[:, 1]


# PERMUTATION IMPORTANCE OF EACH FEATURE GROUP ON ONE HELD-OUT SAMPLE. FOR EVERY GROUP, ALL ITS
# COLUMNS ARE SHUFFLED WITH THE SAME ROW PERMUTATION (SO THE NEAR-COPIES STAY CONSISTENT WITH EACH
# OTHER BUT LOSE THEIR LINK TO THE LABEL), AND THE DROPS IN PR-AUC AND IN RECALL AT THE FIXED
# THRESHOLD ARE AVERAGED OVER IMP_REPEATS SHUFFLES
def permutation_importance(model, thr, idx, groups):
    Xs, ys = X[idx], y[idx]
    base = score(model, Xs)
    base_ap, base_rec = average_precision_score(ys, base), (base[ys == 1] > thr).mean()
    rows = []
    for g, cols in groups.items():
        d_ap, d_rec = [], []
        for _ in range(IMP_REPEATS):
            Xp = Xs.copy()
            Xp[:, cols] = Xs[rng.permutation(len(idx))][:, cols]
            p = score(model, Xp)
            d_ap.append(base_ap - average_precision_score(ys, p))
            d_rec.append(base_rec - (p[ys == 1] > thr).mean())
        rows.append({"feature": g, "n_features": len(cols),
                     "pr_auc_drop": float(np.mean(d_ap)), "pr_auc_drop_sd": float(np.std(d_ap)),
                     "recall_drop": float(np.mean(d_rec)), "recall_drop_sd": float(np.std(d_rec)),
                     "base_pr_auc": base_ap, "base_recall": float(base_rec)})
    return rows


# --- PARTS 1 AND 2: REFIT, CHECK, AND PERMUTATION IMPORTANCE ------------------------------

check_rows, single_rows, group_rows = [], [], []
saved = {}
SINGLE = {f: [j] for j, f in enumerate(FEATS)}

for d in DAYS:
    t0 = time.time()
    train_idx = np.flatnonzero(day != d)
    test_idx = np.flatnonzero(day == d)
    log(f"=== held-out {d}")
    for name in ["hgb", "rf"]:
        # GRADIENT BOOSTING USES THE SETTING MILESTONE 3'S INNER LOOP PICKED FOR THIS DAY; THE
        # FOREST USES LUIS'S FIXED SETTINGS WITH SEED 5502
        params = HGB_PARAMS[d] if name == "hgb" else {"random_state": SEED}
        model, thr, p_cal = fit_thresholded(name, params, train_idx)
        p_test = score(model, X[test_idx])
        m, _ = evaluate(y[test_idx], p_test, thr, labels[test_idx])
        # COMPARE WITH MILESTONE 3 (THE FOREST'S MILESTONE 3 NUMBERS COME FROM NOTEBOOK 05)
        if name == "hgb":
            ref = m3[(m3["model"] == name) & (m3["day"] == d)]
        else:
            ref = m3_forest[(m3_forest["day"] == d) & (m3_forest["seed"] == SEED)]
        check_rows.append({"day": d, "model": name, "recall": m["recall"], "fpr": m["fpr"],
                           "pr_auc": m["pr_auc"],
                           "m3_recall": float(ref["recall"].iloc[0]) if len(ref) else np.nan,
                           "m3_fpr": float(ref["fpr"].iloc[0]) if len(ref) else np.nan})
        log(f"  {name}: recall {m['recall']:.3f} fpr {m['fpr']:.4f} "
            f"(M3: {check_rows[-1]['m3_recall']:.3f} / {check_rows[-1]['m3_fpr']:.4f}) "
            f"{time.time() - t0:.0f}s")
        # KEEP SCORES FOR THE HYBRID DETECTOR (m4_hybrid.py)
        saved[f"{name}__{d}"] = p_test.astype(np.float32)
        saved[f"{name}__{d}__thr"] = np.array([thr])
        saved[f"{name}__{d}__calben"] = p_cal.astype(np.float32)
        # PERMUTATION IMPORTANCE ONLY ON DAYS WITH ATTACKS, ON A CAPPED SAMPLE OF THE HELD-OUT DAY
        if d in ATTACK_DAYS:
            att, ben = test_idx[y[test_idx] == 1], test_idx[y[test_idx] == 0]
            att = rng.choice(att, size=min(IMP_ATTACK_MAX, len(att)), replace=False)
            ben = rng.choice(ben, size=min(IMP_BENIGN_MAX, len(ben)), replace=False)
            idx = np.concatenate([att, ben])
            for rows, groups in [(single_rows, SINGLE), (group_rows, GROUPS)]:
                for r in permutation_importance(model, thr, idx, groups):
                    rows.append({"day": d, "model": name, **r})
            log(f"  {name}: importance done {time.time() - t0:.0f}s")
    saved[f"idx__{d}"] = test_idx
    # CHECKPOINT AFTER EVERY DAY
    pd.DataFrame(check_rows).to_csv(OUT_DIR / "m4_refit_check.csv", index=False)
    pd.DataFrame(single_rows).to_csv(OUT_DIR / "m4_importance_single.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUT_DIR / "m4_importance_groups.csv", index=False)
    np.savez_compressed(OUT_DIR / "m4_supervised_scores.npz", **saved)
    log(f"=== {d} done in {(time.time() - t0) / 60:.1f} min")


# --- PART 3: RANDOM FOREST SEEDS AND WITHOUT THE TCP WINDOW FIELDS ---------------------------

rob_rows, rob_class = [], []
NO_WIN = [j for j, f in enumerate(FEATS) if f not in ARTIFACT_FEATURES]
for d in ATTACK_DAYS:
    train_idx = np.flatnonzero(day != d)
    test_idx = np.flatnonzero(day == d)
    # THE VARIANTS: THREE EXTRA SEEDS ON ALL FEATURES, AND SEED 5502 WITHOUT THE WINDOW FIELDS
    variants = [(f"seed {s}", {"random_state": s}, None) for s in RF_SEEDS]
    variants.append(("no window fields", {"random_state": SEED}, NO_WIN))
    for vname, params, cols in variants:
        ts = time.time()
        model, thr, _ = fit_thresholded("rf", params, train_idx, cols)
        Xt = X[test_idx] if cols is None else X[np.ix_(test_idx, cols)]
        m, pc = evaluate(y[test_idx], score(model, Xt), thr, labels[test_idx])
        rob_rows.append({"day": d, "variant": vname, **m})
        for lab, v in pc.items():
            rob_class.append({"day": d, "variant": vname, "attack_class": lab, **v})
        log(f"  rf {d} {vname}: recall {m['recall']:.3f} fpr {m['fpr']:.4f} "
            f"pr_auc {m['pr_auc']:.3f} ({time.time() - ts:.0f}s)")
    pd.DataFrame(rob_rows).to_csv(OUT_DIR / "m4_rf_robustness.csv", index=False)
    pd.DataFrame(rob_class).to_csv(OUT_DIR / "m4_rf_robustness_per_class.csv", index=False)

# FINAL SUMMARY: TOP 10 FEATURE GROUPS BY MEAN PR-AUC DROP ACROSS THE FOUR ATTACK DAYS, PER MODEL
g = pd.DataFrame(group_rows).groupby(["model", "feature"])[["pr_auc_drop", "recall_drop"]].mean()
for name in ["hgb", "rf"]:
    log(f"\nTop groups, {name}:\n" + g.loc[name].sort_values("pr_auc_drop", ascending=False)
        .head(10).round(4).to_string())
LOG.close()
