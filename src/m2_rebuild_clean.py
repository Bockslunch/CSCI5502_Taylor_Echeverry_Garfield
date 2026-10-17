"""Rebuild the cleaned CICIDS2017 dataset by replaying 01-data-cleaning.ipynb.

Usage (from the repository root): python src/m2_rebuild_clean.py data/raw data/processed

Replays the team's cleaning notebook step for step so the Milestone 2
quality and EDA numbers come from the same analysis-ready data, and records
the row/column counts at every step so they can be checked against the
notebook's printed outputs.
"""

# IMPORT THE STANDARD LIBRARY PIECES FOR ARGUMENTS, PATHS AND THE STEP LOG
import json
import sys
from pathlib import Path

# IMPORT PANDAS AND NUMPY FOR THE CLEANING ITSELF
import numpy as np
import pandas as pd

# READ THE RAW AND OUTPUT FOLDERS FROM THE COMMAND LINE SO NO PATH IS HARD-CODED
RAW_DIR = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MAP EACH RAW FILE TO THE SAME `source_file` TAG THE NOTEBOOK USES, IN THE SAME
# ORDER, SO THE CONCATENATED ROW ORDER (AND THEREFORE WHICH DUPLICATE IS KEPT) MATCHES
FILES = {
    "Monday-WorkingHours.pcap_ISCX.csv": "Monday",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "Tuesday",
    "Wednesday-workingHours.pcap_ISCX.csv": "Wednesday",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "Thursday-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "Thursday-Afternoon-Infiltration",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "Friday-Morning",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "Friday-Afternoon-PortScan",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "Friday-Afternoon-DDoS",
}

# KEEP A RUNNING LOG OF EVERY STEP'S EFFECT SO THE WRITE-UP CAN QUOTE EXACT NUMBERS
log = {}

# LOAD EACH FILE, TAG IT WITH ITS SOURCE, AND STACK THEM (NO JOIN KEY: SAME SCHEMA)
frames = []
for name, tag in FILES.items():
    df = pd.read_csv(RAW_DIR / name, encoding_errors="replace", low_memory=False)
    df["source_file"] = tag
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
del frames

# STRIP THE LEADING SPACES FROM THE RAW HEADERS, COUNTING HOW MANY HAD ONE
raw_cols = list(data.columns)
log["raw_columns_with_leading_space"] = sum(c != c.lstrip() for c in raw_cols)
data.columns = data.columns.str.strip()
log["raw_rows"] = len(data)
log["raw_shape"] = list(data.shape)
log["raw_rows_per_file"] = data["source_file"].value_counts().to_dict()
log["raw_label_counts"] = data["Label"].value_counts().to_dict()

# SAVE RAW PER-FILE x LABEL COUNTS BEFORE ANY CLEANING (USED FOR BEFORE/AFTER VISUALS)
pd.crosstab(data["source_file"], data["Label"]).to_csv(OUT_DIR / "raw_file_by_label.csv")

# STEP 1: COUNT TRUE NaNs, THEN TURN +/-INF INTO NaN AND COUNT AGAIN (NOTEBOOK CELLS 5-6)
log["nan_before_inf_replace"] = int(data.isna().sum().sum())
data = data.replace([np.inf, -np.inf], np.nan)
missing = data.isna().sum()
log["missing_after_inf_replace"] = missing[missing > 0].astype(int).to_dict()

# CONFIRM THE MISSING ROWS ARE EXACTLY THE ZERO-DURATION ROWS (NOTEBOOK CELL 11)
miss_mask = data.isna().any(axis=1)
log["missing_rows"] = int(miss_mask.sum())
log["zero_duration_rows"] = int((data["Flow Duration"] == 0).sum())
log["missing_rows_all_zero_duration"] = bool((data.loc[miss_mask, "Flow Duration"] == 0).all())
log["missing_rows_by_label"] = data.loc[miss_mask, "Label"].value_counts().to_dict()

# DROP THE ZERO-DURATION ROWS WHOSE RATES ARE UNDEFINED (NOTEBOOK CELL 15)
data = data.dropna()
log["rows_after_missing"] = len(data)

# STEP 2: FIND EXACT DUPLICATES IGNORING `source_file`, PER LABEL (NOTEBOOK CELLS 19-22)
feature_cols = data.columns.drop("source_file")
dup_mask = data.duplicated(subset=feature_cols)
log["duplicate_rows"] = int(dup_mask.sum())
log["duplicates_by_label"] = data.loc[dup_mask, "Label"].value_counts().to_dict()

# HASH EACH ROW ONCE (WITH AND WITHOUT THE LABEL) SO GROUPING IS CHEAP ON 2.8M x 78 VALUES
row_hash = pd.util.hash_pandas_object(data[feature_cols], index=False)
nolabel_hash = pd.util.hash_pandas_object(data[feature_cols.drop("Label")], index=False)

# COUNT DUPLICATES WHOSE FIRST COPY SITS IN A DIFFERENT FILE (CROSS-DAY LEAKAGE RISK FOR RQ4)
first_src = data["source_file"].groupby(row_hash.values, sort=False).transform("first")
log["duplicates_cross_file"] = int((dup_mask & (first_src != data["source_file"])).sum())
del first_src

# COUNT ROWS WHOSE FEATURES ARE IDENTICAL BUT WHOSE LABELS DIFFER (LABEL CONFLICTS)
conflict = data["Label"].groupby(nolabel_hash.values, sort=False).transform("nunique")
log["rows_in_label_conflict_groups"] = int((conflict > 1).sum())
log["label_conflict_rows_by_label"] = data.loc[conflict > 1, "Label"].value_counts().to_dict()
del conflict, row_hash, nolabel_hash

# DROP DUPLICATES, KEEPING THE FIRST COPY (NOTEBOOK CELL 25)
data = data.drop_duplicates(subset=feature_cols)
log["rows_after_dedup"] = len(data)

# FIX THE BROKEN EN-DASH CHARACTER IN THE THREE WEB ATTACK LABELS (NOTEBOOK CELL 30)
data["Label"] = data["Label"].str.replace("�", "-", regex=False)

# DROP THE 8 ALWAYS-ZERO COLUMNS AND THE DUPLICATED HEADER-LENGTH COLUMN (NOTEBOOK CELL 36)
zero_columns = ["Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
                "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]
log["zero_columns_all_zero"] = bool((data[zero_columns] == 0).all().all())
log["fwd_header_dup_identical"] = bool(data["Fwd Header Length.1"].equals(data["Fwd Header Length"]))
data = data.drop(columns=zero_columns + ["Fwd Header Length.1"])

# STEP 3: COUNT NEGATIVE VALUES PER NUMERIC COLUMN (NOTEBOOK CELL 40)
num = data.select_dtypes(include="number")
neg = (num < 0).sum()
log["negatives_by_column"] = neg[neg > 0].astype(int).to_dict()

# REMOVE THE IMPOSSIBLE NEGATIVE DURATION / SIZE ROWS (NOTEBOOK CELLS 45-47)
broken = ((data["Flow Duration"] < 0) | (data["Fwd Header Length"] < 0)
          | (data["Bwd Header Length"] < 0) | (data["min_seg_size_forward"] < 0))
log["broken_rows"] = int(broken.sum())
log["broken_rows_by_label"] = data.loc[broken, "Label"].value_counts().to_dict()
data = data.loc[~broken]

# COUNT THE NEGATIVE MINIMUM INTER-ARRIVAL TIMES, THEN CLIP THEM TO 0 (NOTEBOOK CELLS 50-53)
log["neg_flow_iat_min_rows"] = int((data["Flow IAT Min"] < 0).sum())
log["neg_flow_iat_min_by_label"] = data.loc[data["Flow IAT Min"] < 0, "Label"].value_counts().to_dict()
data.loc[data["Flow IAT Min"] < 0, "Flow IAT Min"] = 0
data.loc[data["Fwd IAT Min"] < 0, "Fwd IAT Min"] = 0

# RECORD WHICH NEGATIVES REMAIN AFTER CLEANING (EXPECTED: ONLY THE -1 WINDOW CODES)
neg_after = (data.select_dtypes(include="number") < 0).sum()
log["negatives_remaining"] = neg_after[neg_after > 0].astype(int).to_dict()

# STEP 5: ADD THE BINARY TARGET (NOTEBOOK CELL 63)
data["is_attack"] = (data["Label"] != "BENIGN").astype(int)

# RECORD THE FINAL SHAPE AND CLASS COUNTS FOR CHECKING AGAINST THE NOTEBOOK (2,520,656 x 72)
log["final_shape"] = list(data.shape)
log["final_label_counts"] = data["Label"].value_counts().to_dict()
log["final_rows_per_file"] = data["source_file"].value_counts().to_dict()

# SAVE THE CLEANED FRAME AS A PICKLE (FAST TO RELOAD) AND THE STEP LOG AS JSON
data.reset_index(drop=True).to_pickle(OUT_DIR / "cicids2017_clean.pkl")
(OUT_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=int))
print(json.dumps({k: log[k] for k in ["raw_shape", "missing_rows", "duplicate_rows",
                                      "broken_rows", "final_shape"]}, indent=2))
