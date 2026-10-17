"""Compute the Milestone 2 Section 4 / Section 5 numbers from the cleaned data.

Usage (from the repository root): python src/m2_quality_stats.py data/processed

Reads cicids2017_clean.pkl and cleaning_log.json written by m2_rebuild_clean.py and
writes m2_stats.json plus a few CSV tables into the same folder.
"""

# IMPORT STANDARD-LIBRARY HELPERS FOR ARGUMENTS, PATHS AND JSON OUTPUT
import json
import sys
from pathlib import Path

# IMPORT THE NUMERIC STACK; `ks_2samp` GIVES A SCALE-FREE MEASURE OF HOW FAR APART TWO DISTRIBUTIONS ARE
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, skew

# LOCATE THE INPUTS FROM THE COMMAND LINE (NO MACHINE-SPECIFIC PATHS)
DATA_DIR = Path(sys.argv[1])
df = pd.read_pickle(DATA_DIR / "cicids2017_clean.pkl")
clean_log = json.loads((DATA_DIR / "cleaning_log.json").read_text())

# SEPARATE THE 69 FLOW FEATURES FROM THE LABEL / BOOKKEEPING COLUMNS
NON_FEATURES = ["Label", "source_file", "is_attack"]
FEATURES = [c for c in df.columns if c not in NON_FEATURES]
stats = {"n_rows": len(df), "n_features": len(FEATURES)}

# --- COMPLETENESS / IMBALANCE ---------------------------------------------------

# COUNT ROWS PER CLASS AND THE LARGEST-TO-SMALLEST RATIO (THE "ORDERS OF MAGNITUDE" CLAIM)
counts = df["Label"].value_counts()
stats["class_counts"] = counts.to_dict()
stats["imbalance_ratio_max_min"] = float(counts.max() / counts.min())
stats["attack_share_pct"] = float(df["is_attack"].mean() * 100)

# SHARE OF ALL ATTACK ROWS CONTRIBUTED BY THE THREE BIGGEST ATTACK CLASSES
atk = counts.drop("BENIGN")
stats["top3_attack_share_pct"] = float(atk.nlargest(3).sum() / atk.sum() * 100)
stats["top3_attack_classes"] = atk.nlargest(3).index.tolist()

# --- REPRESENTATIVENESS: WHERE EACH CLASS LIVES ---------------------------------

# CROSS-TABULATE CAPTURE FILE x CLASS; COUNT HOW MANY FILES EACH ATTACK CLASS APPEARS IN
day_class = pd.crosstab(df["source_file"], df["Label"])
day_class.to_csv(DATA_DIR / "clean_file_by_label.csv")
stats["files_per_attack_class"] = (day_class.drop(columns="BENIGN") > 0).sum().to_dict()
stats["benign_share_by_file_pct"] = (day_class["BENIGN"] / day_class.sum(axis=1) * 100).round(2).to_dict()

# COMPARE BENIGN TRAFFIC ACROSS DAYS: KS DISTANCE OF EACH DAY'S BENIGN FLOWS VS MONDAY'S
# (IF BENIGN ITSELF DRIFTS BY DAY, A DAY-TRANSFER TEST MEASURES DRIFT, NOT JUST NEW ATTACKS)
benign = df[df["Label"] == "BENIGN"]
mon = benign[benign["source_file"] == "Monday"]
drift = {}
for feat in ["Flow Duration", "Total Fwd Packets", "Packet Length Mean", "Flow IAT Mean", "Destination Port"]:
    drift[feat] = {}
    for src, grp in benign.groupby("source_file"):
        if src != "Monday":
            drift[feat][src] = round(float(ks_2samp(mon[feat], grp[feat]).statistic), 3)
stats["benign_ks_vs_monday"] = drift
del benign, mon

# --- VALIDITY / CONSISTENCY ISSUES BY CLASS --------------------------------------

# REBUILD PER-CLASS REMOVAL RATES FROM THE CLEANING LOG (LABELS IN THE LOG STILL HAVE THE BROKEN DASH)
fix = lambda d: {k.replace("�", "-"): v for k, v in d.items()}
raw = pd.Series(fix(clean_log["raw_label_counts"]))
miss = pd.Series(fix(clean_log["missing_rows_by_label"])).reindex(raw.index, fill_value=0)
dups = pd.Series(fix(clean_log["duplicates_by_label"])).reindex(raw.index, fill_value=0)

# FLAG ROWS IN THE CLEANED DATA WHOSE FEATURES EXACTLY MATCH A ROW OF A DIFFERENT CLASS
feat_hash = pd.util.hash_pandas_object(df[FEATURES], index=False)
n_labels = df["Label"].groupby(feat_hash.values).transform("nunique")
conflict = (n_labels > 1).values
stats["clean_label_conflict_rows"] = int(conflict.sum())
stats["clean_label_conflict_by_label"] = df.loc[conflict, "Label"].value_counts().to_dict()

# BUILD THE CLASS x ISSUE TABLE OF PERCENTAGES USED BY THE DATA-QUALITY FIGURE
g = df.groupby("Label")
issues = pd.DataFrame({
    "Zero duration (removed)": miss / raw * 100,
    "Exact duplicate (removed)": dups / (raw - miss) * 100,
    "Same features, other label": pd.Series(conflict, index=df.index).groupby(df["Label"]).mean() * 100,
    "Init_Win_bytes_forward = -1": g["Init_Win_bytes_forward"].apply(lambda s: (s == -1).mean() * 100),
    "Flow Duration >= 119 s": g["Flow Duration"].apply(lambda s: (s >= 119_000_000).mean() * 100),
    "No forward payload": g["Total Length of Fwd Packets"].apply(lambda s: (s == 0).mean() * 100),
})
issues.index.name = "Label"
issues.round(2).to_csv(DATA_DIR / "quality_issues_by_class.csv")
stats["quality_issues_by_class"] = issues.round(2).to_dict(orient="index")

# OVERALL SHARES FOR THE SENTINEL AND TIMEOUT ISSUES
stats["init_win_fwd_neg1_pct"] = float((df["Init_Win_bytes_forward"] == -1).mean() * 100)
stats["init_win_bwd_neg1_pct"] = float((df["Init_Win_bytes_backward"] == -1).mean() * 100)
stats["timeout_119s_rows"] = int((df["Flow Duration"] >= 119_000_000).sum())

# --- REDUNDANT FEATURES -----------------------------------------------------------

# FIND FEATURE PAIRS THAT ARE IDENTICAL ROW FOR ROW (PURE REDUNDANCY, NOT JUST CORRELATION)
identical = []
for i, a in enumerate(FEATURES):
    for b in FEATURES[i + 1:]:
        if df[a].dtype == df[b].dtype and df[a].equals(df[b]):
            identical.append([a, b])
stats["identical_feature_pairs"] = identical

# FIND FEATURES THAT NEVER VARY OR TAKE ONLY A HANDFUL OF VALUES AFTER CLEANING
nunique = df[FEATURES].nunique()
stats["low_cardinality_features"] = nunique[nunique <= 2].to_dict()

# SPEARMAN CORRELATION ON A CLASS-STRATIFIED SAMPLE (ALL ROWS OF SMALL CLASSES, CAPPED AT 20K PER CLASS)
sample = df.groupby("Label", group_keys=False).apply(
    lambda s: s.sample(min(len(s), 20_000), random_state=5502))
rho = sample[FEATURES].rank().corr()
upper = rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1))
stats["pairs_abs_spearman_ge_0.95"] = int((upper.abs() >= 0.95).sum().sum())
stats["pairs_abs_spearman_ge_0.99"] = int((upper.abs() >= 0.99).sum().sum())

# --- WHICH FEATURES DIFFER MOST BETWEEN BENIGN AND ATTACK ------------------------

# KS STATISTIC FOR EVERY FEATURE, BENIGN VS ATTACK (0 = SAME DISTRIBUTION, 1 = NO OVERLAP)
b_mask = df["is_attack"].values == 0
ks = {f: float(ks_2samp(df[f].values[b_mask], df[f].values[~b_mask]).statistic) for f in FEATURES}
ks = pd.Series(ks).sort_values(ascending=False)
ks.round(3).to_csv(DATA_DIR / "ks_benign_vs_attack.csv", header=["ks"])
stats["ks_top10"] = ks.head(10).round(3).to_dict()

# SAME KS RANKING BUT ATTACK SIDE BALANCED SO NO SINGLE ATTACK CLASS DOMINATES
# (EACH ATTACK CLASS CONTRIBUTES AT MOST 2,000 ROWS)
bal = df[df["is_attack"] == 1].groupby("Label", group_keys=False).apply(
    lambda s: s.sample(min(len(s), 2_000), random_state=5502))
ks_bal = pd.Series({f: float(ks_2samp(df[f].values[b_mask], bal[f].values).statistic) for f in FEATURES})
stats["ks_top10_class_balanced"] = ks_bal.sort_values(ascending=False).head(10).round(3).to_dict()

# SAVE BOTH KS RANKINGS SIDE BY SIDE SO THE WRITE-UP CAN QUOTE ANY FEATURE'S VALUES
pd.DataFrame({"ks_all_rows": ks, "ks_class_balanced": ks_bal}).round(3).to_csv(
    DATA_DIR / "ks_benign_vs_attack_both.csv")

# --- SUMMARY STATISTICS FOR THE KEY FEATURES, BENIGN VS ATTACK -------------------

# MEDIAN, QUARTILES AND SKEWNESS: MEDIAN/IQR BECAUSE EVERY FEATURE IS HEAVILY RIGHT-SKEWED
KEY = ["Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
       "Packet Length Mean", "Flow IAT Mean", "Destination Port"]
rows = []
for name, grp in df.groupby("is_attack"):
    for f in KEY:
        s = grp[f]
        rows.append({"group": "Attack" if name else "Benign", "feature": f,
                     "median": float(s.median()), "q1": float(s.quantile(.25)),
                     "q3": float(s.quantile(.75)), "mean": float(s.mean()),
                     "skew": float(skew(s))})
summary = pd.DataFrame(rows)
summary.to_csv(DATA_DIR / "key_feature_summary.csv", index=False)
stats["key_feature_summary"] = summary.round(3).to_dict(orient="records")

# --- DESTINATION PORT CONCENTRATION (EXTENDS TYLER'S CHART) ----------------------

# NUMBER OF DISTINCT PORTS AND SHARE OF THE TOP PORT, PER CLASS
port = df.groupby("Label")["Destination Port"].agg(
    n_ports="nunique", top_port=lambda s: int(s.mode().iloc[0]),
    top_share=lambda s: float(s.value_counts(normalize=True).iloc[0] * 100))
stats["dest_port_by_class"] = port.round(3).to_dict(orient="index")

# --- HOW SEPARABLE ARE CLASS PAIRS THAT CLUSTER TOGETHER? (RQ3) ------------------

# FOR EACH PAIR, THE LARGEST KS DISTANCE OVER ALL FEATURES EXCEPT DESTINATION PORT
# (A LOW MAXIMUM MEANS NO SINGLE FEATURE TELLS THE TWO CLASSES APART); BIG CLASSES SAMPLED TO 50K
PAIR_FEATS = [f for f in FEATURES if f != "Destination Port"]
PAIRS = [("Web Attack - Brute Force", "Web Attack - XSS"), ("Bot", "BENIGN"),
         ("DoS slowloris", "DoS Slowhttptest"), ("DoS Hulk", "DoS GoldenEye"),
         ("FTP-Patator", "SSH-Patator")]
pair_out = {}
for a, b in PAIRS:
    A = df[df["Label"] == a]
    B = df[df["Label"] == b]
    A = A.sample(min(len(A), 50_000), random_state=1)
    B = B.sample(min(len(B), 50_000), random_state=1)
    ks_pair = pd.Series({f: ks_2samp(A[f], B[f]).statistic for f in PAIR_FEATS}).sort_values(ascending=False)
    pair_out[f"{a} vs {b}"] = {"max_ks": round(float(ks_pair.iloc[0]), 3), "feature": ks_pair.index[0]}
stats["pairwise_max_ks"] = pair_out

# WRITE EVERYTHING OUT FOR THE WRITE-UP
(DATA_DIR / "m2_stats.json").write_text(json.dumps(stats, indent=2, default=float))
print(json.dumps(stats, indent=1, default=float)[:6000])
