"""Error analysis for the RQ4 gradient-boosting model (condition A: day never seen).

Usage (from the repository root, after m3_rq4_transfer.py):
    python src/m3_error_analysis.py data/processed results/m3

For each held-out day it compares, on a handful of interpretable features:
  - false alarms (benign flows above the threshold) vs. benign flows below it
  - missed attacks vs. caught attacks, per attack class
using medians (every feature is heavily right-skewed). Destination Port is not a
model input, but it is reported here descriptively because it shows *which*
services the errors come from. Also reports how far the achieved false-alarm
rate on each unseen day drifted from the 1% target set on the training days.

Outputs: m3_errors_false_alarms.csv, m3_errors_missed_attacks.csv,
m3_errors_fp_ports.csv, m3_errors_score_quantiles.csv
"""

# IMPORT PATH HANDLING AND THE NUMERIC STACK
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# MAKE `src/` IMPORTABLE AND LOAD THE SHARED DATA LOADER
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_common import DAYS, load_clean

# READ INPUT AND OUTPUT FOLDERS FROM THE COMMAND LINE
DATA_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])

# FEATURES TO PROFILE: A SMALL, READABLE SET COVERING DURATION, VOLUME, SIZE AND TIMING,
# PLUS THE WINDOW FIELD THE MILESTONE 2 AUDIT FLAGGED AS A PROTOCOL/HOST PROXY
PROFILE = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
           "Total Length of Fwd Packets", "Packet Length Mean", "Flow IAT Mean",
           "Init_Win_bytes_forward"]

# LOAD THE DATA AND THE CONDITION-A GRADIENT-BOOSTING SCORES SAVED BY RQ4
df = load_clean(DATA_DIR)
saved = np.load(OUT_DIR / "rq4_scores.npz")

# COLLECT ONE ROW PER (DAY, GROUP) FOR EACH OUTPUT TABLE
fa_rows, miss_rows, port_rows, q_rows = [], [], [], []

for d in DAYS:
    # ROWS OF THIS HELD-OUT DAY, THEIR SCORES AND THE THRESHOLD SET ON THE TRAINING DAYS
    idx = saved[f"idx__{d}"]
    p = saved[f"hgb__{d}"]
    thr = float(saved[f"hgb__{d}__thr"][0])
    sub = df.iloc[idx].copy()
    sub["score"] = p
    sub["alert"] = p > thr
    ben = sub[sub["is_attack"] == 0]

    # SCORE DISTRIBUTION OF BENIGN FLOWS ON THE UNSEEN DAY: IF ITS 99TH PERCENTILE SITS ABOVE THE
    # THRESHOLD, THE DAY'S NORMAL TRAFFIC LOOKS MORE "ATTACK-LIKE" THAN THE TRAINING DAYS' DID
    q_rows.append({"day": d, "threshold": thr, "benign_fpr": ben["alert"].mean(),
                   "benign_p50": ben["score"].median(), "benign_p99": ben["score"].quantile(.99),
                   "benign_p999": ben["score"].quantile(.999)})

    # FALSE ALARMS VS. QUIET BENIGN FLOWS: MEDIAN OF EACH PROFILE FEATURE AND THE GROUP SIZE
    for grp_name, grp in [("false_alarm", ben[ben["alert"]]), ("benign_quiet", ben[~ben["alert"]])]:
        row = {"day": d, "group": grp_name, "n": len(grp)}
        row.update(grp[PROFILE].median().add_prefix("median ").to_dict())
        fa_rows.append(row)

    # WHICH DESTINATION PORTS THE FALSE ALARMS GO TO (TOP 5 BY COUNT, WITH EACH PORT'S OWN FPR)
    fa = ben[ben["alert"]]
    if len(fa):
        top = fa["Destination Port"].value_counts().head(5)
        for port, n in top.items():
            port_rows.append({"day": d, "port": int(port), "false_alarms": int(n),
                              "share_of_false_alarms": n / len(fa),
                              "fpr_on_port": ben.loc[ben["Destination Port"] == port, "alert"].mean()})

    # MISSED VS. CAUGHT ATTACKS, PER CLASS (CLASSES WITH FEWER THAN 5 FLOWS IN A GROUP ARE STILL
    # LISTED; THEIR MEDIANS ARE NOT INTERPRETED IN THE WRITE-UP)
    for lab, cls in sub[sub["is_attack"] == 1].groupby("Label"):
        for grp_name, grp in [("missed", cls[~cls["alert"]]), ("caught", cls[cls["alert"]])]:
            row = {"day": d, "attack_class": lab, "group": grp_name, "n": len(grp),
                   "median score": grp["score"].median() if len(grp) else np.nan}
            row.update(grp[PROFILE].median().add_prefix("median ").to_dict())
            miss_rows.append(row)

# SAVE THE FOUR TABLES
pd.DataFrame(fa_rows).to_csv(OUT_DIR / "m3_errors_false_alarms.csv", index=False)
pd.DataFrame(miss_rows).to_csv(OUT_DIR / "m3_errors_missed_attacks.csv", index=False)
pd.DataFrame(port_rows).to_csv(OUT_DIR / "m3_errors_fp_ports.csv", index=False)
pd.DataFrame(q_rows).to_csv(OUT_DIR / "m3_errors_score_quantiles.csv", index=False)
print(pd.DataFrame(q_rows).round(4).to_string())
