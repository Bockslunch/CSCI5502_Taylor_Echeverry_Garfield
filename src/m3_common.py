"""Shared pieces for the Milestone 3 modeling scripts.

Everything the RQ2 / RQ4 / RQ5 scripts must agree on lives here, so the
comparison between methods uses one feature list, one set of folds, one
preprocessing recipe, one threshold rule and one set of metrics:

- load_clean():        the cleaned data written by m2_rebuild_clean.py
- FEATURES:            the 64 modeling features (and why the rest are excluded)
- defect_flags():      the three RQ5 defect flags, defined as in Milestone 2
- make_model():        the baselines and the two candidate methods
- split_fit_calib():   train-only split used for early stopping and the threshold
- pick_threshold():    alert threshold set from benign calibration scores only
- evaluate():          the metrics reported for every model on every fold
"""

# IMPORT PATH HANDLING AND THE NUMERIC STACK
from pathlib import Path

import numpy as np
import pandas as pd

# IMPORT THE MODELS, PREPROCESSING AND METRICS FROM SCIKIT-LEARN
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ONE SEED FOR EVERY RANDOM STEP, SO ANY TEAMMATE GETS THE SAME NUMBERS
SEED = 5502

# THE FIVE CAPTURE DAYS, IN ORDER; EACH IS ONE LEAVE-ONE-DAY-OUT FOLD
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# TARGET FALSE-ALARM RATE ON BENIGN TRAFFIC USED TO SET EVERY MODEL'S ALERT THRESHOLD
# (1% OF BENIGN FLOWS FLAGGED = AN ALERT BUDGET A SOC COULD PLAUSIBLY TRIAGE; SEE THE REPORT)
TARGET_FPR = 0.01

# COLUMNS THAT ARE NOT FLOW FEATURES: THE TARGET, THE CLASS NAME AND THE CAPTURE FILE
NON_FEATURES = ["Label", "source_file", "is_attack"]

# FEATURES DELIBERATELY EXCLUDED FROM EVERY MODEL, WITH THE REASON FOR EACH
EXCLUDED = {
    # PORT IS AN IDENTIFIER OF THE TARGET SERVICE, NOT ATTACK BEHAVIOR; MOST ATTACK CLASSES
    # HIT ONE PORT, SO A MODEL WOULD MEMORIZE "PORT 80 ON WEDNESDAY" (MILESTONE 2, RQ2 WORDING)
    "Destination Port": "identifier / shortcut (RQ2 excludes it)",
    # THESE FOUR ARE ROW-FOR-ROW IDENTICAL COPIES OF ANOTHER COLUMN (MILESTONE 2 SECTION 4)
    "Subflow Fwd Packets": "identical to Total Fwd Packets",
    "Subflow Bwd Packets": "identical to Total Backward Packets",
    "SYN Flag Count": "identical to Fwd PSH Flags",
    "CWE Flag Count": "identical to Fwd URG Flags",
}

# FEATURES THE MILESTONE 2 AUDIT FLAGGED AS ARTIFACT-PRONE: THE TCP WINDOW FIELDS USE -1 AS A
# "NOT OBSERVED" CODE AND THEIR VALUES TRACK THE FIXED ATTACKER/VICTIM HOSTS, NOT THE ATTACK.
# THEY STAY IN THE MAIN MODELS; RQ2 RERUNS WITHOUT THEM TO SEE WHICH RESULTS SURVIVE.
ARTIFACT_FEATURES = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]

# THE 120-SECOND FLOW TIMEOUT IN MICROSECONDS; FLOWS AT OR NEAR IT WERE CUT BY CICFLOWMETER,
# NOT ENDED BY THE CONNECTION (119 S, AS IN MILESTONE 2, CATCHES ROUNDING JUST UNDER THE CAP)
TIMEOUT_US = 119_000_000


# LOAD THE CLEANED FRAME AND ADD THE CAPTURE DAY AS ITS OWN COLUMN
def load_clean(data_dir):
    """Return the cleaned frame with a `day` column (Monday ... Friday)."""
    # READ THE PICKLE WRITTEN BY m2_rebuild_clean.py (FASTER AND TYPE-EXACT COMPARED WITH THE CSV)
    df = pd.read_pickle(Path(data_dir) / "cicids2017_clean.pkl")
    # THE CAPTURE FILE TAG STARTS WITH THE DAY ("Friday-Afternoon-DDoS" -> "Friday"); THREE
    # FRIDAY FILES AND TWO THURSDAY FILES THEREFORE COLLAPSE INTO ONE FOLD EACH
    df["day"] = df["source_file"].str.split("-").str[0]
    return df


# BUILD THE FINAL FEATURE LIST FROM WHATEVER COLUMNS THE CLEANED FILE HAS
def feature_list(df, drop_artifacts=False):
    """All flow features minus EXCLUDED (and minus ARTIFACT_FEATURES if asked)."""
    # START FROM EVERY COLUMN, REMOVE BOOKKEEPING COLUMNS AND THE DOCUMENTED EXCLUSIONS
    skip = set(NON_FEATURES) | {"day"} | set(EXCLUDED)
    # FOR THE RQ2 ROBUSTNESS RUN, ALSO REMOVE THE TWO ARTIFACT-PRONE WINDOW FIELDS
    if drop_artifacts:
        skip |= set(ARTIFACT_FEATURES)
    return [c for c in df.columns if c not in skip]


# FLAG THE THREE KINDS OF DEFECTIVE FLOWS THAT RQ5 EXCLUDES, USING THE MILESTONE 2 DEFINITIONS
def defect_flags(df, features):
    """Boolean columns: zero-payload web attack, timeout-capped, shared feature vector."""
    # 1) WEB-ATTACK FLOWS WITH NO FORWARD PAYLOAD: AN HTTP ATTACK THAT SENT NO BYTES CANNOT
    #    CONTAIN THE ATTACK, SO THE LABEL IS ALMOST CERTAINLY A TIME-WINDOW MISLABEL
    web = df["Label"].str.startswith("Web Attack")
    zero_payload_web = web & (df["Total Length of Fwd Packets"] == 0)
    # 2) FLOWS CUT AT THE 120 S TIMEOUT: THEIR DURATION AND RATES DESCRIBE THE TOOL, NOT THE TRAFFIC
    timeout = df["Flow Duration"] >= TIMEOUT_US
    # 3) FEATURE VECTORS THAT ALSO APPEAR UNDER A DIFFERENT LABEL: NO MODEL CAN GET BOTH RIGHT.
    #    HASH EACH ROW'S FEATURES ONCE, THEN COUNT DISTINCT LABELS PER HASH (CHEAP ON 2.5M ROWS)
    h = pd.util.hash_pandas_object(df[features], index=False)
    shared = df["Label"].groupby(h.values).transform("nunique").values > 1
    # RETURN ALL THREE PLUS THE UNION, ALIGNED WITH THE INPUT ROWS
    out = pd.DataFrame({"zero_payload_web": zero_payload_web.values,
                        "timeout_capped": timeout.values,
                        "shared_vector": shared}, index=df.index)
    out["any_defect"] = out.any(axis=1)
    return out


# SIGNED LOG: FLOW FEATURES SPAN 0 TO 10^8 AND A FEW CARRY THE -1 CODE, SO A PLAIN LOG FAILS.
# sign(x) * log(1 + |x|) KEEPS -1 DISTINCT FROM 0 AND SHRINKS 120,000,000 TO ABOUT 18.6
def signed_log1p(x):
    """Elementwise sign(x) * log1p(|x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


# BUILD ONE OF THE MODELS COMPARED IN RQ4 BY NAME, WITH OPTIONAL HYPERPARAMETERS
def make_model(name, **params):
    """Return an unfitted estimator. Names: majority, stratified, logreg, hgb."""
    # BASELINE 1 (TYLER'S): ALWAYS PREDICT THE MAJORITY CLASS (BENIGN); CATCHES NO ATTACKS
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    # BASELINE 2 (TYLER'S): GUESS AT RANDOM IN PROPORTION TO THE TRAINING CLASS MIX
    if name == "stratified":
        return DummyClassifier(strategy="stratified", random_state=SEED)
    # CANDIDATE 1: LOGISTIC REGRESSION. THE LOG STEP UNDOES THE HEAVY RIGHT SKEW AND THE SCALER
    # PUTS FEATURES ON ONE SCALE; BOTH ARE FIT INSIDE THE PIPELINE, SO ONLY ON TRAINING ROWS.
    # class_weight="balanced" STOPS THE 80% BENIGN MAJORITY FROM DOMINATING THE LOSS.
    if name == "logreg":
        return make_pipeline(
            FunctionTransformer(signed_log1p),
            StandardScaler(),
            LogisticRegression(C=params.get("C", 1.0), class_weight="balanced",
                               max_iter=params.get("max_iter", 500), random_state=SEED))
    # CANDIDATE 2: HISTOGRAM GRADIENT BOOSTING. TREES NEED NO SCALING AND HANDLE SKEW AND
    # NON-LINEAR THRESHOLDS DIRECTLY; BINNING TO 255 BUCKETS KEEPS 2M ROWS INSIDE 7 GB OF RAM.
    # EARLY STOPPING HOLDS OUT 10% OF THE TRAINING ROWS, NEVER THE TEST DAY.
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 0.0),
            max_iter=300, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, class_weight="balanced", random_state=SEED)
    # ANY OTHER NAME IS A TYPO; FAIL LOUDLY RATHER THAN SILENTLY TRAIN THE WRONG THING
    raise ValueError(f"unknown model {name!r}")


# SPLIT THE TRAINING DAYS INTO A FIT PART (80%) AND A CALIBRATION PART (20%), STRATIFIED BY CLASS
def split_fit_calib(idx, labels):
    """Return (fit_idx, calib_idx) from the training-row indices only."""
    # STRATIFYING BY THE FULL LABEL KEEPS EVERY SMALL ATTACK CLASS IN BOTH PARTS.
    # A CLASS WITH A SINGLE ROW CANNOT BE SPLIT, SO IT JOINS THE LARGEST CLASS'S STRATUM
    # (E.G. HEARTBLEED AFTER RQ5 REMOVES ITS 10 TIMEOUT-CAPPED FLOWS LEAVES 1 ROW).
    # WHEN EVERY CLASS HAS 2+ ROWS (ALL RQ4 FOLDS) THIS IS PLAIN LABEL STRATIFICATION.
    lab = labels[idx]
    counts = pd.Series(lab).value_counts()
    strat = np.where(pd.Series(lab).map(counts).values >= 2, lab, counts.index[0])
    return train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strat)


# PICK THE ALERT THRESHOLD SO THAT ABOUT TARGET_FPR OF BENIGN CALIBRATION FLOWS SCORE ABOVE IT.
# ONLY BENIGN SCORES ARE USED, AND ONLY FROM TRAINING DAYS, SO THE TEST DAY NEVER SETS IT.
# EXAMPLE: WITH TARGET_FPR = 0.01 THE THRESHOLD IS THE 99TH PERCENTILE OF BENIGN SCORES.
def pick_threshold(benign_scores, target_fpr=TARGET_FPR):
    """Score cut-off giving approximately `target_fpr` false alarms on benign calibration data."""
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


# SCORE ONE MODEL ON ONE SET OF TEST ROWS
def evaluate(y_true, scores, threshold, labels=None):
    """Metrics for one fold. `scores` = predicted attack probability; alert = score > threshold."""
    # TURN SCORES INTO ALERTS WITH THE THRESHOLD CHOSEN ON TRAINING DAYS
    alert = scores > threshold
    att, ben = y_true == 1, y_true == 0
    n_att, n_ben = int(att.sum()), int(ben.sum())
    # CONFUSION-MATRIX COUNTS (ATTACK = POSITIVE CLASS)
    tp, fn = int((alert & att).sum()), int((~alert & att).sum())
    fp, tn = int((alert & ben).sum()), int((~alert & ben).sum())
    out = {"n_attack": n_att, "n_benign": n_ben, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
           # RECALL = SHARE OF ATTACK FLOWS THAT RAISED AN ALERT (THE DETECTION RATE RQ4 ASKS ABOUT)
           "recall": tp / n_att if n_att else np.nan,
           # FPR = SHARE OF BENIGN FLOWS THAT RAISED AN ALERT (THE ANALYST'S WASTED-ALERT COST)
           "fpr": fp / n_ben if n_ben else np.nan,
           # PRECISION = SHARE OF ALERTS THAT WERE REAL ATTACKS; UNDEFINED IF NOTHING ALERTED
           "precision": tp / (tp + fp) if (tp + fp) else np.nan}
    # F1 COMBINES PRECISION AND RECALL; ONLY DEFINED WHEN BOTH ARE
    p, r = out["precision"], out["recall"]
    out["f1"] = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else np.nan
    # THRESHOLD-FREE RANKING METRICS NEED BOTH CLASSES PRESENT (MONDAY HAS NO ATTACKS).
    # PR-AUC (AVERAGE PRECISION) IS THE HEADLINE RANKING METRIC BECAUSE ATTACKS ARE THE RARE CLASS;
    # ITS NO-SKILL VALUE EQUALS THE ATTACK SHARE OF THE TEST DAY, SO IT IS REPORTED ALONGSIDE.
    both = n_att > 0 and n_ben > 0
    out["pr_auc"] = float(average_precision_score(y_true, scores)) if both else np.nan
    out["roc_auc"] = float(roc_auc_score(y_true, scores)) if both else np.nan
    out["attack_share"] = n_att / len(y_true)
    # PER-ATTACK-CLASS RECALL (WHICH UNSEEN ATTACK TYPES WERE CAUGHT), IF LABELS WERE PASSED
    per_class = {}
    if labels is not None:
        for lab in np.unique(labels[att]):
            m = labels == lab
            per_class[str(lab)] = {"n": int(m.sum()), "recall": float(alert[m].mean())}
    return out, per_class
