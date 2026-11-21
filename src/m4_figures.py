"""Build the Milestone 4 figures and summary tables from the result tables.

Usage (from the repository root, after m4_anomaly.py, m4_importance.py and m4_hybrid.py):
    python src/m4_figures.py results/m3 results/m4 figures docs/assets

Writes three figures, each twice at 300 dpi (<name>.png light, <name>-dark.png dark),
in the same themes as the Milestone 2 and 3 figures, and copies them to the site
assets folder if one is given:

  m4-recall-by-class-detectors   which unseen attack classes each approach catches
  m4-heldout-importance          which features carry detection to an unseen day
  m4-window-dependence           pooled recall with and without the TCP window fields

Also writes the summary tables quoted in the final report and on the website:
  results/m4/m4_summary_pooled.csv   every approach, pooled over the four attack days
  results/m4/m4_summary_by_day.csv   the new approaches, per held-out day
"""

# IMPORT STANDARD-LIBRARY HELPERS
import shutil
import sys
from pathlib import Path

# IMPORT THE NUMERIC AND PLOTTING STACK; "Agg" RENDERS TO FILES WITHOUT A SCREEN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# READ THE MILESTONE 3 AND 4 RESULTS FOLDERS, THE FIGURE FOLDER AND THE OPTIONAL SITE FOLDER
M3, M4, FIG_DIR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SITE_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 else None
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- THEMES (SAME VALUES AS m2_eda_figures.py AND m3_figures.py) -------------------------

# LIGHT AND DARK THEMES MATCHED TO THE WEBSITE BACKGROUNDS; BLUE AND ORANGE PASS THE COLORBLIND
# SEPARATION CHECK IN BOTH MODES. GRAY "CONTEXT" IS FOR REFERENCE MARKS ONLY
THEMES = {
    "light": dict(suffix="", surface="#ffffff", ink="#1a1d21", ink2="#5a626c", grid="#e4e3df",
                  blue="#2a78d6", orange="#eb6834", context="#b9b8b2",
                  seq=["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(suffix="-dark", surface="#14171a", ink="#e7eaed", ink2="#a3acb6", grid="#2c2f33",
                 blue="#3987e5", orange="#d95926", context="#5b6068",
                 seq=["#1c232b", "#184f95", "#3987e5", "#86b6ef", "#cde2fb"]),
}


# SET MODULE-LEVEL COLORS AND MATPLOTLIB DEFAULTS FOR ONE THEME
def apply_theme(name):
    global T, SEQ
    T = THEMES[name]
    SEQ = LinearSegmentedColormap.from_list(f"seq-{name}", T["seq"])
    plt.rcParams.update({
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"],
        "savefig.facecolor": T["surface"], "font.family": "DejaVu Sans", "font.size": 8.5,
        "text.color": T["ink"], "axes.edgecolor": T["grid"], "axes.labelcolor": T["ink2"],
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": T["ink"], "xtick.color": T["ink2"], "ytick.color": T["ink2"],
        "axes.grid": True, "grid.color": T["grid"], "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# PICK BLACK OR WHITE TEXT FOR A HEATMAP CELL FROM ITS BACKGROUND LUMINANCE
def ink_on(cmap, position):
    r, g, b, _ = cmap(min(max(position, 0.0), 1.0))
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "#ffffff"


# SAVE AT PRINT RESOLUTION WITH THE THEME SUFFIX, AND COPY TO THE SITE IF ASKED
def save(fig, name):
    out = FIG_DIR / f"{name}{T['suffix']}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if SITE_DIR is not None:
        SITE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, SITE_DIR / out.name)


# --- LOAD RESULT TABLES ----------------------------------------------------------------------

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ATTACK_DAYS = DAYS[1:]
ARTIFACT_FEATURES = {"Init_Win_bytes_forward", "Init_Win_bytes_backward"}

# MILESTONE 3: HEADLINE FOLD METRICS, PER-CLASS RECALL, THE FOREST, AND THE NO-WINDOW BOOSTING RUN
rq4 = pd.read_csv(M3 / "rq4_fold_metrics.csv")
rq4 = rq4[(rq4["condition"] == "A_lodo") & (rq4["test_set"] == "full_day")]
rq4c = pd.read_csv(M3 / "rq4_per_class.csv")
rq4c = rq4c[(rq4c["condition"] == "A_lodo") & (rq4c["test_set"] == "full_day")]
forest = pd.read_csv(M3 / "rq4_forest.csv")
forest = forest[forest["seed"] == 5502]
hgb_nowin = pd.read_csv(M3 / "rq2_transfer_no_artifacts.csv")

# MILESTONE 4: DETECTORS, HYBRIDS, IMPORTANCE, FOREST ROBUSTNESS
an = pd.read_csv(M4 / "m4_anomaly_fold_metrics.csv")
an = an[(an["condition"] == "A_lodo") & (an["test_set"] == "full_day") & (an["flows"] == "all")]
anc = pd.read_csv(M4 / "m4_anomaly_per_class.csv")
anc = anc[(anc["condition"] == "A_lodo") & (anc["test_set"] == "full_day") & (anc["flows"] == "all")]
hy = pd.read_csv(M4 / "m4_hybrid_fold_metrics.csv")
hyc = pd.read_csv(M4 / "m4_hybrid_per_class.csv")
imp = pd.read_csv(M4 / "m4_importance_groups.csv")
rob = pd.read_csv(M4 / "m4_rf_robustness.csv")


# --- SUMMARY TABLES --------------------------------------------------------------------------

# POOL ONE APPROACH'S FOUR ATTACK DAYS: SUM THE CONFUSION COUNTS, THEN RECOMPUTE THE RATES (THE SAME
# RULE m3_figures.py USES), PLUS MEAN PR-AUC OVER THE FOUR DAYS AND THE MONDAY FALSE-ALARM RATE
def pool(frame, name, group):
    att = frame[frame["day"].isin(ATTACK_DAYS)]
    tp, fn, fp, tn = (int(att[c].sum()) for c in ["tp", "fn", "fp", "tn"])
    mon = frame[frame["day"] == "Monday"]
    return {"approach": name, "group": group, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn), "fpr": fp / (fp + tn),
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "mean_pr_auc": float(att["pr_auc"].mean()) if "pr_auc" in att else np.nan,
            "monday_fpr": float(mon["fpr"].iloc[0]) if len(mon) else np.nan}


rows = [pool(rq4[rq4["model"] == m], n, "Milestone 3 classifiers")
        for m, n in [("majority", "Majority class"), ("stratified", "Stratified guess"),
                     ("single_feature", "Single-feature rule"), ("logreg", "Logistic regression"),
                     ("hgb", "Gradient boosting")]]
rows.append(pool(forest, "Random forest (seed 5502)", "Milestone 3 classifiers"))
rows += [pool(an[an["model"] == m], n, "Anomaly detectors (benign-only)")
         for m, n in [("iforest", "Isolation Forest"), ("pca", "PCA reconstruction error")]]
rows += [pool(hy[hy["model"] == m], n, "Hybrid (shared 1% budget)")
         for m, n in [("hgb+iforest", "Gradient boosting + Isolation Forest"),
                      ("rf+iforest", "Random forest + Isolation Forest")]]
rows.append(pool(hgb_nowin, "Gradient boosting, no window fields", "Robustness"))
rows.append(pool(rob[rob["variant"] == "no window fields"], "Random forest, no window fields",
                 "Robustness"))
for s in [1, 2, 3]:
    rows.append(pool(rob[rob["variant"] == f"seed {s}"], f"Random forest (seed {s})", "Robustness"))
summary = pd.DataFrame(rows)
summary.to_csv(M4 / "m4_summary_pooled.csv", index=False)

# PER-DAY TABLE FOR THE NEW APPROACHES (THE REPORT'S PER-DAY DISCUSSION QUOTES IT)
cols = ["day", "recall", "fpr", "precision", "pr_auc"]
by_day = pd.concat([an.assign(approach=an["model"])[["approach"] + cols],
                    hy.assign(approach=hy["model"])[["approach"] + cols],
                    rob.assign(approach="rf " + rob["variant"])[["approach"] + cols]])
by_day.to_csv(M4 / "m4_summary_by_day.csv", index=False)


# --- FIGURE: PER-CLASS RECALL, CLASSIFIERS vs DETECTORS vs HYBRID --------------------------------

def fig_per_class():
    # HEATMAP: ATTACK CLASS x APPROACH, VALUE = SHARE OF THAT CLASS'S FLOWS CAUGHT ON ITS UNSEEN DAY
    pieces = [rq4c[rq4c["model"] == "hgb"].assign(approach="Gradient\nboosting"),
              anc[anc["model"] == "iforest"].assign(approach="Isolation\nForest"),
              anc[anc["model"] == "pca"].assign(approach="PCA\nerror"),
              hyc[hyc["model"] == "hgb+iforest"].assign(approach="Boosting +\nIsol. Forest")]
    order_cols = [p["approach"].iloc[0] for p in pieces]
    tab = pd.concat(pieces).pivot_table(index="attack_class", columns="approach",
                                        values="recall")[order_cols]
    # CLASS SIZE AND DAY FROM THE BOOSTING ROWS (EVERY APPROACH SCORED THE SAME TEST FLOWS)
    ref = rq4c[rq4c["model"] == "hgb"].drop_duplicates("attack_class").set_index("attack_class")
    order = sorted(tab.index, key=lambda c: (DAYS.index(ref.loc[c, "day"]), -ref.loc[c, "n"]))
    tab = tab.loc[order]
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.imshow(tab.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    # PRINT EVERY CELL'S VALUE SO THE HEATMAP READS WITHOUT RELYING ON SHADE ALONE
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            txt = "<1%" if 0 < v < 0.005 else f"{v:.0%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=ink_on(SEQ, v))
    # A THIN DIVIDER SEPARATES THE SUPERVISED CLASSIFIER FROM THE BENIGN-ONLY DETECTORS
    for x in [0.5, 2.5]:
        ax.axvline(x, color=T["surface"], lw=3)
    ax.set_xticks(range(len(order_cols)))
    ax.set_xticklabels(order_cols, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}  ({ref.loc[c, 'day'][:3]}, n={int(ref.loc[c, 'n']):,})"
                        for c in order], fontsize=7.6)
    ax.grid(False)
    ax.set_title("Share of each attack class caught when its day was never seen")
    save(fig, "m4-recall-by-class-detectors")


# --- FIGURE: HELD-OUT PERMUTATION IMPORTANCE ------------------------------------------------------

def fig_importance():
    # TOP 10 FEATURE GROUPS PER MODEL BY MEAN PR-AUC DROP OVER THE FOUR UNSEEN ATTACK DAYS. BARS ARE
    # COLORED BY WHETHER THE GROUP IS AN ARTIFACT-PRONE TCP WINDOW FIELD (ORANGE) OR TRAFFIC
    # BEHAVIOR (BLUE); THE LEGEND AND A TEXT TAG CARRY THE SAME DISTINCTION, SO IT IS NOT COLOR-ONLY
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True, gridspec_kw={"wspace": 0.95})
    xmax = 0
    for ax, (m, title) in zip(axes, [("hgb", "Gradient boosting"), ("rf", "Random forest")]):
        g = imp[imp["model"] == m].groupby("feature")["pr_auc_drop"].agg(["mean", "min", "max"])
        g = g.nlargest(10, "mean")[::-1]
        art = [f in ARTIFACT_FEATURES for f in g.index]
        colors = [T["orange"] if a else T["blue"] for a in art]
        ax.barh(range(len(g)), g["mean"], color=colors, height=0.62)
        # THE THIN LINE SPANS THE SMALLEST TO LARGEST PER-DAY VALUE (HOW MUCH THE DAYS DISAGREE)
        ax.hlines(range(len(g)), g["min"].clip(lower=0), g["max"], color=T["ink2"], lw=0.8)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([f + ("  [artifact]" if a else "") for f, a in zip(g.index, art)],
                           fontsize=7.6)
        ax.set_title(title)
        ax.set_xlabel("drop in PR-AUC on the unseen day when shuffled")
        ax.grid(axis="y", visible=False)
        xmax = max(xmax, g["max"].max())
    axes[0].set_xlim(0, xmax * 1.05)
    # ONE SHARED LEGEND BELOW THE PANELS
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=T["blue"], label="Traffic behavior"),
                        Patch(color=T["orange"], label="TCP window field (artifact-prone)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    save(fig, "m4-heldout-importance")


# --- FIGURE: DEPENDENCE ON THE TCP WINDOW FIELDS ---------------------------------------------------

def fig_window():
    # POOLED RECALL WITH ALL 64 FEATURES vs WITHOUT THE TWO WINDOW FIELDS, FOR BOTH TREE MODELS. THE
    # FOREST'S FOUR SEEDS ARE DRAWN AS DOTS ON ITS ALL-FEATURES BAR TO SHOW SEED-TO-SEED SPREAD
    s = summary.set_index("approach")
    models = [("Gradient boosting", "Gradient boosting", "Gradient boosting, no window fields"),
              ("Random forest", "Random forest (seed 5502)", "Random forest, no window fields")]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    w = 0.34
    seeds = s.loc[[f"Random forest (seed {x})" for x in [5502, 1, 2, 3]], "recall"]
    for i, (label, full, nowin) in enumerate(models):
        for k, (key, col, name) in enumerate([(full, T["blue"], "All 64 features"),
                                              (nowin, T["orange"], "Without TCP window fields")]):
            v = s.loc[key, "recall"]
            ax.bar(i + (k - 0.5) * w, v, width=w - 0.02, color=col, label=name if i == 0 else None)
            # THE VALUE LABEL SITS ABOVE THE BAR, AND ABOVE THE SEED DOTS ON THE FOREST'S BAR
            top = max(v, seeds.max()) if (i == 1 and k == 0) else v
            ax.text(i + (k - 0.5) * w, top + 0.02, f"{v:.0%}", ha="center", fontsize=8,
                    color=T["ink"])
    ax.scatter([1 - 0.5 * w] * len(seeds), seeds, s=22, color=T["ink"], zorder=3,
               label="Random forest, each of 4 seeds")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylim(0, 0.68)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("unseen-day recall (4 attack days pooled)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Removing the TCP window fields breaks boosting, not the forest")
    ax.legend(loc="upper left", fontsize=7.6)
    save(fig, "m4-window-dependence")


# DRAW EVERY FIGURE IN BOTH THEMES
for theme in ["light", "dark"]:
    apply_theme(theme)
    fig_per_class()
    fig_importance()
    fig_window()
print(summary[["approach", "recall", "fpr", "precision", "mean_pr_auc", "monday_fpr"]]
      .round(3).to_string(index=False))
print("figures written to", FIG_DIR)
