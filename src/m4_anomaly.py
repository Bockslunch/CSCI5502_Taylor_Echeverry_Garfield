"""Milestone 4: can an anomaly detector trained on benign traffic only catch unseen attacks?

Usage (from the repository root):
    python src/m4_anomaly.py data/processed results/m4

Why: RQ4 asks whether detection transfers to attacks the model has never seen.
The Milestone 3 classifiers learn what *known* attacks look like, so they can only
catch a new attack that happens to resemble an old one. An anomaly detector learns
what *normal* traffic looks like and flags anything unusual, so in principle it
does not need to have seen an attack type before. This script tests that claim
under exactly the Milestone 3 design, so its numbers sit in the same table:

  - the same five leave-one-day-out folds, the same 64 features (m3_common.py);
  - detectors are fit on the BENIGN flows of the four training days only;
  - hyperparameters are chosen by an inner leave-one-day-out loop over the
    training days only (mean PR-AUC, as in m3_rq4_transfer.py);
  - the alert threshold is set so ~1% of benign calibration flows from the
    training days are flagged (pick_threshold); the test day never sets it;
  - the same benign-shift test as RQ4: half of the held-out day is the common
    test set, and condition B adds the other half's benign flows to training;
  - the same RQ5 view: metrics on all test flows and on defect-free test flows.

Detectors:
  iforest  Isolation Forest. Random trees split random features at random cut
           points; unusual flows end up alone after few splits. Primary detector.
  pca      PCA reconstruction error. Benign traffic is projected onto its main
           directions of variation; flows that the projection reconstructs badly
           are unusual. The classic linear baseline for network anomaly detection.

Outputs (output folder): m4_anomaly_fold_metrics.csv, m4_anomaly_per_class.csv,
m4_anomaly_tuning.csv, m4_anomaly_scores.npz (held-out-day scores, not committed),
m4_anomaly_run_log.txt.
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, TIMING, PATHS AND JSON
import json
import sys
import time
from pathlib import Path

# IMPORT THE NUMERIC STACK, THE TWO DETECTORS AND THE SCORING HELPERS
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MAKE `src/` IMPORTABLE WHEN RUN FROM THE REPOSITORY ROOT, THEN LOAD THE SHARED MILESTONE 3 PIECES
# SO FOLDS, FEATURES, THRESHOLD RULE, METRICS AND DEFECT FLAGS ARE IDENTICAL TO THE CLASSIFIERS'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import (DAYS, SEED, defect_flags, evaluate, feature_list, load_clean,
                       pick_threshold, signed_log1p)

# READ INPUT AND OUTPUT FOLDERS FROM THE COMMAND LINE (NO MACHINE-SPECIFIC PATHS)
DATA_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SMALL HYPERPARAMETER GRIDS SEARCHED BY THE INNER LOOP (THREE SETTINGS EACH, LIKE MILESTONE 3).
# IFOREST: max_samples IS HOW MANY FLOWS EACH TREE SEES (SMALL = COARSE, ROBUST; LARGE = FINER
# DETAIL); max_features < 1 MAKES EACH TREE LOOK AT A RANDOM HALF OF THE FEATURES.
# PCA: HOW MUCH OF THE BENIGN VARIANCE THE "NORMAL" SUBSPACE KEEPS. KEEPING LESS MAKES MORE OF
# EACH FLOW COUNT AS RECONSTRUCTION ERROR; KEEPING MORE ONLY FLAGS FLOWS FAR OFF THE SUBSPACE.
GRIDS = {
    "iforest": [{"max_samples": 256, "max_features": 1.0},
                {"max_samples": 4096, "max_features": 1.0},
                {"max_samples": 256, "max_features": 0.5}],
    "pca": [{"var_kept": 0.90}, {"var_kept": 0.95}, {"var_kept": 0.99}],
}

# NUMBER OF TREES IN EVERY ISOLATION FOREST (200 GIVES STABLE SCORES; SCORING TIME GROWS WITH IT)
N_TREES = 200

# PCA AND THE SCALER ARE FIT ON A RANDOM SAMPLE OF THIS MANY BENIGN FLOWS: THE MEANS, VARIANCES
# AND MAIN DIRECTIONS OF 64 FEATURES ARE ESTIMATED PRECISELY FROM 300K ROWS, AND IT SAVES RAM
PCA_FIT_ROWS = 300_000

# DURING INNER TUNING, SCORE AT MOST THIS MANY BENIGN FLOWS OF EACH INNER DAY (ALL ITS ATTACKS ARE
# KEPT) SO 3 SETTINGS x 3 INNER DAYS x 5 OUTER DAYS FIT THE TIME BUDGET ON A LAPTOP
TUNE_BENIGN_MAX = 150_000

# A RUN LOG ON DISK, SO A LONG BACKGROUND RUN CAN BE FOLLOWED AND AUDITED AFTERWARDS
LOG = open(OUT_DIR / "m4_anomaly_run_log.txt", "w")


# PRINT A TIMESTAMPED LINE TO THE SCREEN AND THE LOG FILE
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# --- LOAD DATA ---------------------------------------------------------------------

# LOAD THE CLEANED DATA, FLAG THE RQ5 DEFECTS, AND BUILD THE FEATURE MATRIX ONCE
df = load_clean(DATA_DIR)
FEATS = feature_list(df)
defect = defect_flags(df, FEATS)["any_defect"].to_numpy()
# APPLY THE SIGNED LOG ONCE UP FRONT: BOTH DETECTORS NEED IT (ISOLATION FOREST PICKS CUT POINTS
# UNIFORMLY BETWEEN MIN AND MAX, SO ON THE RAW 0..10^8 SCALE ALMOST EVERY CUT WOULD ONLY SEPARATE
# THE FEW HUGE VALUES; ON THE LOG SCALE CUTS SPREAD ACROSS THE RANGE WHERE FLOWS ACTUALLY DIFFER)
X = signed_log1p(df[FEATS].to_numpy(np.float32)).astype(np.float32)
y = df["is_attack"].to_numpy()
labels = df["Label"].to_numpy()
day = df["day"].to_numpy()
del df
log(f"loaded {X.shape[0]:,} rows x {X.shape[1]} features; {int(defect.sum()):,} defect-flagged")

# ONE RANDOM GENERATOR FOR ALL SUBSAMPLING IN THIS SCRIPT
rng = np.random.default_rng(SEED)


# --- DETECTORS -----------------------------------------------------------------------

# A SMALL WRAPPER SO BOTH DETECTORS EXPOSE THE SAME fit(rows) / score(rows) INTERFACE, WHERE A
# HIGHER SCORE ALWAYS MEANS "MORE ANOMALOUS" (SO THE MILESTONE 3 THRESHOLD AND METRICS APPLY AS-IS)
class Detector:
    def __init__(self, name, params):
        self.name, self.params = name, params

    # FIT ON BENIGN ROWS ONLY (THE CALLER GUARANTEES THAT `idx` CONTAINS NO ATTACK ROWS)
    def fit(self, idx):
        if self.name == "iforest":
            # contamination="auto" ONLY AFFECTS predict(), WHICH IS NOT USED: THE THRESHOLD COMES
            # FROM pick_threshold() ON BENIGN CALIBRATION SCORES, LIKE EVERY OTHER MODEL
            self.model = IsolationForest(
                n_estimators=N_TREES, max_samples=self.params["max_samples"],
                max_features=self.params["max_features"], random_state=SEED, n_jobs=-1)
            self.model.fit(X[idx])
        else:
            # FIT THE SCALER AND PCA ON A BENIGN SAMPLE; KEEP JUST ENOUGH COMPONENTS TO EXPLAIN
            # `var_kept` OF THE BENIGN VARIANCE
            s = rng.choice(idx, size=min(PCA_FIT_ROWS, len(idx)), replace=False)
            self.scaler = StandardScaler().fit(X[s])
            self.model = PCA(n_components=self.params["var_kept"], svd_solver="full",
                             random_state=SEED).fit(self.scaler.transform(X[s]))
        return self

    # SCORE ROWS IN CHUNKS SO 2.5M x 64 TEMPORARY ARRAYS NEVER SIT IN MEMORY AT ONCE
    def score(self, idx, chunk=250_000):
        out = np.empty(len(idx), dtype=np.float32)
        for a in range(0, len(idx), chunk):
            part = X[idx[a:a + chunk]]
            if self.name == "iforest":
                # score_samples IS HIGH FOR NORMAL FLOWS, SO NEGATE IT: HIGH = ANOMALOUS
                out[a:a + chunk] = -self.model.score_samples(part)
            else:
                # SQUARED DISTANCE BETWEEN A FLOW AND ITS PROJECTION ONTO THE BENIGN SUBSPACE
                z = self.scaler.transform(part)
                recon = self.model.inverse_transform(self.model.transform(z))
                out[a:a + chunk] = ((z - recon) ** 2).sum(axis=1)
        return out

    # HOW MANY PCA COMPONENTS WERE KEPT (REPORTED IN THE LOG; EMPTY FOR THE FOREST)
    def describe(self):
        return f" ({self.model.n_components_} components)" if self.name == "pca" else ""


# SPLIT BENIGN TRAINING ROWS 80/20 INTO FIT AND CALIBRATION PARTS. ATTACK ROWS ARE NEVER USED FOR
# FITTING OR FOR THE THRESHOLD, SO THE DETECTOR STAYS TRULY "BENIGN-ONLY"
def benign_fit_calib(idx):
    ben = idx[y[idx] == 0]
    return train_test_split(ben, test_size=0.2, random_state=SEED)


# INNER LEAVE-ONE-DAY-OUT TUNING ON THE TRAINING DAYS ONLY. THE DETECTOR IS FIT ON BENIGN FLOWS OF
# THE INNER TRAINING DAYS; THE INNER HELD-OUT DAY'S LABELS ONLY RANK THE SETTINGS (MEAN PR-AUC).
# THIS IS THE SAME SELECTION RULE THE CLASSIFIERS USED, SO NEITHER SIDE GETS AN ADVANTAGE.
def tune(name, train_days, outer_day):
    records = []
    for params in GRIDS[name]:
        scores = []
        for inner in train_days:
            # MONDAY HAS NO ATTACKS, SO IT CANNOT BE AN INNER TEST DAY (PR-AUC IS UNDEFINED)
            if inner == "Monday":
                continue
            tr = np.flatnonzero(np.isin(day, [d for d in train_days if d != inner]) & (y == 0))
            te = np.flatnonzero(day == inner)
            # KEEP EVERY INNER-DAY ATTACK, SUBSAMPLE ITS BENIGN FLOWS FOR SPEED
            ben = te[y[te] == 0]
            if len(ben) > TUNE_BENIGN_MAX:
                ben = rng.choice(ben, size=TUNE_BENIGN_MAX, replace=False)
            te = np.concatenate([te[y[te] == 1], ben])
            det = Detector(name, params).fit(tr)
            scores.append(average_precision_score(y[te], det.score(te)))
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
saved = {}

# EACH CAPTURE DAY TAKES ONE TURN AS THE UNSEEN TEST DAY (MONDAY GIVES A FALSE-ALARM RATE ONLY)
for d in DAYS:
    t0 = time.time()
    train_days = [x for x in DAYS if x != d]
    train_idx = np.flatnonzero(day != d)
    test_idx = np.flatnonzero(day == d)
    log(f"=== held-out {d}: {int((y[train_idx] == 0).sum()):,} benign training flows, "
        f"test {len(test_idx):,} flows ({int(y[test_idx].sum()):,} attacks)")

    # SPLIT THE HELD-OUT DAY IN HALF EXACTLY AS m3_rq4_transfer.py DOES (SAME SEED AND STRATA), SO
    # THE BENIGN-SHIFT COMPARISON USES THE SAME HALF-DAY TEST SET AS THE CLASSIFIERS
    lab_d = labels[test_idx]
    vc = pd.Series(lab_d).value_counts()
    strat = np.where(pd.Series(lab_d).map(vc).values >= 2, lab_d, "other")
    half_add, half_test = train_test_split(test_idx, test_size=0.5, random_state=SEED,
                                           stratify=strat)

    # TWO TRAINING CONDITIONS (C, "ATTACKS SEEN", DOES NOT EXIST FOR A BENIGN-ONLY DETECTOR):
    #   A  other four days only
    #   B  A plus the benign flows of the other half of the held-out day (benign shift removed)
    cond_train = {"A_lodo": train_idx,
                  "B_lodo_plus_day_benign": np.concatenate([train_idx, half_add[y[half_add] == 0]])}

    # SCORE ONE DETECTOR'S RESULTS ON A TEST SET AND STORE THEM (ALL FLOWS AND DEFECT-FREE FLOWS)
    def record(model, cond, test_name, idx, scores, thr, params):
        for subset, keep in [("all", np.ones(len(idx), bool)), ("clean", ~defect[idx])]:
            m, pc = evaluate(y[idx][keep], scores[keep], thr, labels[idx][keep])
            fold_rows.append({"day": d, "model": model, "condition": cond, "test_set": test_name,
                              "flows": subset, "threshold": thr, "params": params, **m})
            for lab, v in pc.items():
                class_rows.append({"day": d, "model": model, "condition": cond,
                                   "test_set": test_name, "flows": subset,
                                   "attack_class": lab, **v})

    for name in ["iforest", "pca"]:
        # TUNE ON THE TRAINING DAYS ONLY, THEN KEEP THAT SETTING FOR BOTH CONDITIONS
        best, recs = tune(name, train_days, d)
        tune_rows += recs
        log(f"  {name} chosen: {best}")
        for cond, tr in cond_train.items():
            # FIT ON 80% OF THE BENIGN TRAINING FLOWS; THE OTHER 20% SET THE 1% ALERT THRESHOLD
            fit_i, cal_i = benign_fit_calib(tr)
            ts = time.time()
            det = Detector(name, best).fit(fit_i)
            thr = pick_threshold(det.score(cal_i))
            record(name, cond, "half_day", half_test, det.score(half_test), thr, json.dumps(best))
            # CONDITION A IS ALSO SCORED ON THE WHOLE HELD-OUT DAY: THE HEADLINE RQ4 NUMBER
            if cond == "A_lodo":
                p_full = det.score(test_idx)
                record(name, cond, "full_day", test_idx, p_full, thr, json.dumps(best))
                saved[f"{name}__{d}"] = p_full
                saved[f"{name}__{d}__thr"] = np.array([thr])
            log(f"  {name} {cond}: {time.time() - ts:.0f}s thr={thr:.4f}{det.describe()}")

    # SAVE THE TEST-DAY ROW INDICES SO LATER SCRIPTS CAN LINE SCORES UP WITH THE ORIGINAL ROWS
    saved[f"idx__{d}"] = test_idx

    # CHECKPOINT AFTER EVERY DAY SO A CRASH LOSES AT MOST ONE FOLD
    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "m4_anomaly_fold_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "m4_anomaly_per_class.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(OUT_DIR / "m4_anomaly_tuning.csv", index=False)
    np.savez_compressed(OUT_DIR / "m4_anomaly_scores.npz", **saved)
    log(f"=== {d} done in {(time.time() - t0) / 60:.1f} min")

# FINAL SUMMARY ON SCREEN: HEADLINE METRICS PER DETECTOR AND DAY (CONDITION A, ALL FLOWS)
res = pd.DataFrame(fold_rows)
head = res[(res["test_set"] == "full_day") & (res["flows"] == "all")].pivot_table(
    index="model", columns="day", values=["recall", "fpr", "pr_auc"])
log("\n" + head.round(3).to_string())
LOG.close()
