"""Build the Milestone 3 figures from the result tables.

Usage (from the repository root, after the three m3_* experiment scripts):
    python src/m3_figures.py results/m3 figures docs/assets

Each figure is written twice at 300 dpi, <name>.png (light) and <name>-dark.png
(dark), using the same two themes as the Milestone 2 figures so the website
looks consistent. The optional third argument copies them into docs/assets/.
Also writes the summary tables quoted in the report and on the website
(m3_summary_*.csv) into the results folder.
"""

# IMPORT STANDARD-LIBRARY HELPERS
import json
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

# READ THE RESULTS FOLDER, FIGURE FOLDER AND OPTIONAL SITE-ASSETS FOLDER FROM THE COMMAND LINE
RES, FIG_DIR = Path(sys.argv[1]), Path(sys.argv[2])
SITE_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else None
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- THEMES (COPIED FROM m2_eda_figures.py SO BOTH MILESTONES' FIGURES MATCH) -------

# LIGHT AND DARK THEMES MATCHED TO THE WEBSITE BACKGROUNDS. THE THREE SERIES COLORS PASS THE
# ALL-PAIRS COLORBLIND CHECK; A FOURTH, NEUTRAL "CONTEXT" GRAY IS USED FOR BASELINES.
THEMES = {
    "light": dict(
        suffix="", surface="#ffffff", ink="#1a1d21", ink2="#5a626c", grid="#e4e3df",
        blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", context="#b9b8b2",
        seq=["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"],
    ),
    "dark": dict(
        suffix="-dark", surface="#14171a", ink="#e7eaed", ink2="#a3acb6", grid="#2c2f33",
        blue="#3987e5", orange="#d95926", aqua="#199e70", context="#5b6068",
        seq=["#1c232b", "#184f95", "#3987e5", "#86b6ef", "#cde2fb"],
    ),
}


# SET MODULE-LEVEL COLORS AND MATPLOTLIB DEFAULTS FOR ONE THEME
def apply_theme(name):
    global T, SEQ
    T = THEMES[name]
    SEQ = LinearSegmentedColormap.from_list(f"seq-{name}", T["seq"])
    # RECESSIVE AXES, LIGHT GRID AND NEUTRAL TEXT, AS IN MILESTONE 2
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


# --- LOAD RESULT TABLES ----------------------------------------------------------------

# RQ4 FOLD METRICS AND PER-CLASS RECALLS; RQ5 AND RQ2 TABLES
rq4 = pd.read_csv(RES / "rq4_fold_metrics.csv")
rq4c = pd.read_csv(RES / "rq4_per_class.csv")
rq5c = pd.read_csv(RES / "rq5_per_class.csv")
rq5 = pd.read_csv(RES / "rq5_fold_metrics.csv")
rq2cv = pd.read_csv(RES / "rq2_cv_metrics.csv")
rq2f1 = pd.read_csv(RES / "rq2_per_class_f1.csv")
rq2cm = pd.read_csv(RES / "rq2_confusion_V2.csv", index_col=0)
rq2imp = pd.read_csv(RES / "rq2_importance.csv")
rq2tr = pd.read_csv(RES / "rq2_transfer_no_artifacts.csv")

# DISPLAY ORDER AND NAMES SHARED BY THE FIGURES
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_ATTACKS = {"Monday": "no attacks", "Tuesday": "FTP/SSH brute force",
               "Wednesday": "DoS + Heartbleed", "Thursday": "web attacks + infiltration",
               "Friday": "Bot, PortScan, DDoS"}
MODEL_NAMES = {"majority": "Majority class", "stratified": "Stratified guess",
               "single_feature": "Single-feature rule", "logreg": "Logistic regression",
               "hgb": "Gradient boosting"}

# HEADLINE RQ4 TABLE: EVERY MODEL ON EVERY FULL HELD-OUT DAY (CONDITION A)
head = rq4[(rq4["condition"] == "A_lodo") & (rq4["test_set"] == "full_day")].copy()

# --- SUMMARY TABLES FOR THE REPORT AND THE WEBSITE ------------------------------------

# TABLE 1: PER-DAY RECALL / FALSE-ALARM RATE / PRECISION / PR-AUC FOR EACH MODEL
cols = ["day", "model", "n_attack", "n_benign", "recall", "fpr", "precision", "f1", "pr_auc",
        "attack_share", "params"]
head[cols].to_csv(RES / "m3_summary_rq4_by_day.csv", index=False)

# TABLE 2: POOLED OVER THE FOUR ATTACK DAYS (MICRO: SUM THE COUNTS, THEN DIVIDE) PLUS THE
# MEAN PR-AUC ACROSS THOSE DAYS, AND MONDAY'S FALSE-ALARM RATE SHOWN SEPARATELY
atk_days = head[head["day"] != "Monday"]
pooled = atk_days.groupby("model").agg(tp=("tp", "sum"), fn=("fn", "sum"), fp=("fp", "sum"),
                                       tn=("tn", "sum"), mean_pr_auc=("pr_auc", "mean"))
pooled["recall"] = pooled["tp"] / (pooled["tp"] + pooled["fn"])
pooled["fpr"] = pooled["fp"] / (pooled["fp"] + pooled["tn"])
pooled["precision"] = pooled["tp"] / (pooled["tp"] + pooled["fp"]).replace(0, np.nan)
pooled["monday_fpr"] = head[head["day"] == "Monday"].set_index("model")["fpr"]
pooled = pooled.reindex([m for m in MODEL_NAMES if m in pooled.index])
pooled.to_csv(RES / "m3_summary_rq4_pooled.csv")

# TABLE 3: THE A / B / C DECOMPOSITION ON THE COMMON HALF-DAY TEST SETS
dec = rq4[rq4["test_set"] == "half_day"].pivot_table(
    index=["model", "day"], columns="condition", values=["recall", "fpr", "pr_auc"])
dec.to_csv(RES / "m3_summary_rq4_decomposition.csv")


# --- FIGURE 8: RQ4 HEADLINE ------------------------------------------------------------
def fig_rq4_headline():
    # THREE PANELS PER HELD-OUT DAY: ATTACK RECALL, FALSE-ALARM RATE, PR-AUC
    models = ["single_feature", "logreg", "hgb"]
    colors = {"single_feature": T["context"], "logreg": T["blue"], "hgb": T["orange"]}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"wspace": 0.38})
    y = np.arange(len(DAYS))[::-1]
    h = 0.26
    for k, m in enumerate(models):
        sub = head[head["model"] == m].set_index("day").reindex(DAYS)
        off = (1 - k) * h
        # PANEL 1: SHARE OF THE HELD-OUT DAY'S ATTACK FLOWS THAT RAISED AN ALERT
        axes[0].barh(y + off, sub["recall"].fillna(0), height=h, color=colors[m],
                     label=MODEL_NAMES[m])
        # PANEL 2: SHARE OF THE HELD-OUT DAY'S BENIGN FLOWS THAT RAISED AN ALERT
        axes[1].barh(y + off, sub["fpr"] * 100, height=h, color=colors[m])
        # PANEL 3: THRESHOLD-FREE RANKING QUALITY (UNDEFINED ON MONDAY: NO ATTACKS)
        axes[2].barh(y + off, sub["pr_auc"].fillna(0), height=h, color=colors[m])
    # THE NO-SKILL PR-AUC EQUALS THE DAY'S ATTACK SHARE; MARK IT SO "0.3" CAN BE READ IN CONTEXT
    share = head[head["model"] == "hgb"].set_index("day").reindex(DAYS)["attack_share"]
    for yi, s in zip(y, share):
        if s > 0:
            axes[2].plot([s, s], [yi - 0.45, yi + 0.45], color=T["ink"], lw=1.2)
    # THE 1% FALSE-ALARM BUDGET EVERY THRESHOLD WAS SET FOR (ON TRAINING DAYS)
    axes[1].axvline(1.0, color=T["ink"], lw=1, ls="--")
    # LABEL THE LINE IN THE EMPTY SPACE BESIDE MONDAY'S ROW (MONDAY'S BARS STOP NEAR 1.6%)
    axes[1].annotate("1% target", xy=(1.0, y[0] + 0.47), xytext=(2.6, y[0] + 0.47), color=T["ink2"],
                     fontsize=7.5, va="center",
                     arrowprops={"arrowstyle": "->", "color": T["ink2"], "lw": 0.8})
    # LABELS AND AXES
    labels = [f"{d}\n{DAY_ATTACKS[d]}" for d in DAYS]
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=7.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Attacks caught (recall)")
    axes[0].set_xlabel("share of held-out day's attack flows")
    axes[1].set_title("False alarms (FPR)")
    axes[1].set_xlabel("% of held-out day's benign flows")
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Ranking quality (PR-AUC)")
    axes[2].set_xlabel("vertical tick = no-skill level")
    axes[0].text(0.02, y[0], "n/a", va="center", color=T["ink2"], fontsize=7.5)
    axes[2].text(0.02, y[0], "n/a", va="center", color=T["ink2"], fontsize=7.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.75, -0.2), ncol=3, fontsize=8)
    save(fig, "m3-rq4-transfer-by-day")


# --- FIGURE 9: RQ4 DECOMPOSITION (A -> B -> C) -------------------------------------------
def fig_rq4_decomposition():
    # FOR EACH MODEL, ONE PANEL PER METRIC; EACH DAY IS A LINE THROUGH THE THREE CONDITIONS
    conds = ["A_lodo", "B_lodo_plus_day_benign", "C_in_day"]
    cond_lab = ["A: other days\nonly", "B: + that day's\nbenign", "C: + that day's\nattacks too"]
    day_colors = {"Tuesday": T["blue"], "Wednesday": T["orange"], "Thursday": T["aqua"],
                  "Friday": T["ink2"], "Monday": T["context"]}
    half = rq4[rq4["test_set"] == "half_day"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    for r, m in enumerate(["logreg", "hgb"]):
        for c, (metric, title) in enumerate([("recall", "Attacks caught (recall)"),
                                             ("fpr", "False alarms (% of benign)")]):
            ax = axes[r, c]
            for d in DAYS:
                s = half[(half["model"] == m) & (half["day"] == d)].set_index("condition")
                vals = s.reindex(conds)[metric]
                if metric == "fpr":
                    vals = vals * 100
                if vals.notna().sum() == 0:
                    continue
                ax.plot(range(3), vals.values, marker="o", color=day_colors[d], lw=1.6, ms=4,
                        label=d)
            ax.set_xticks(range(3))
            ax.set_xticklabels(cond_lab, fontsize=7.5)
            ax.set_title(f"{MODEL_NAMES[m]}: {title}", fontsize=9)
            if metric == "recall":
                ax.set_ylim(-0.03, 1.03)
            else:
                ax.axhline(1.0, color=T["ink"], lw=0.9, ls="--")
    axes[0, 1].legend(loc="upper right", fontsize=7.5)
    save(fig, "m3-rq4-shift-decomposition")


# --- FIGURE 10: PER-CLASS RECALL ON THE HELD-OUT DAY -----------------------------------
def fig_rq4_per_class():
    # HEATMAP: ATTACK CLASS x MODEL, VALUE = SHARE OF THAT CLASS'S FLOWS CAUGHT WHEN ITS DAY WAS UNSEEN
    sub = rq4c[(rq4c["condition"] == "A_lodo") & (rq4c["test_set"] == "full_day")
               & rq4c["model"].isin(["single_feature", "logreg", "hgb"])]
    tab = sub.pivot_table(index="attack_class", columns="model", values="recall")
    tab = tab[["single_feature", "logreg", "hgb"]]
    n = sub.drop_duplicates("attack_class").set_index("attack_class")["n"]
    day_of = sub.drop_duplicates("attack_class").set_index("attack_class")["day"]
    order = sorted(tab.index, key=lambda c: (DAYS.index(day_of[c]), -n[c]))
    tab = tab.loc[order]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(tab.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            # A NONZERO RECALL UNDER 1% IS SHOWN AS "<1%" SO IT IS NOT READ AS EXACTLY ZERO
            txt = "<1%" if 0 < v < 0.005 else f"{v:.0%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=ink_on(SEQ, v))
    ax.set_xticks(range(3))
    ax.set_xticklabels([MODEL_NAMES[m] for m in tab.columns], fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}  ({day_of[c][:3]}, n={n[c]:,})" for c in order], fontsize=7.6)
    ax.grid(False)
    ax.set_title("Share of each attack class caught when its day was never seen")
    save(fig, "m3-rq4-recall-by-class")


# --- FIGURE 11: RQ5 RETAINED vs EXCLUDED, PER CLASS --------------------------------------
def fig_rq5():
    # DOT PLOT FOR GRADIENT BOOSTING: RECALL ON ALL TEST ROWS (RETAINED MODEL), ON CLEAN TEST
    # ROWS (RETAINED MODEL), AND ON CLEAN TEST ROWS (MODEL RETRAINED WITHOUT DEFECTS)
    sub = rq5c[rq5c["model"] == "hgb"].copy()
    sub["key"] = sub["training"] + "|" + sub["test_set"]
    tab = sub.pivot_table(index="attack_class", columns="key", values="recall")
    nall = sub[sub["key"] == "retained|all_test"].set_index("attack_class")["n"]
    nclean = sub[sub["key"] == "excluded|clean_test"].set_index("attack_class")["n"]
    day_of = sub.drop_duplicates("attack_class").set_index("attack_class")["day"]
    order = sorted(tab.index, key=lambda c: (DAYS.index(day_of[c]), -nall[c]))[::-1]
    tab = tab.loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    yy = np.arange(len(order))
    spec = [("retained|all_test", "Defects kept: all test flows", T["context"], "o"),
            ("retained|clean_test", "Defects kept: clean test flows only", T["blue"], "s"),
            ("excluded|clean_test", "Defects removed from training: clean test flows", T["orange"], "D")]
    for key, lab, col, mk in spec:
        if key in tab:
            ax.scatter(tab[key], yy, color=col, marker=mk, s=28, label=lab, zorder=3)
    for i, c in enumerate(order):
        ax.plot([np.nanmin(tab.loc[c]), np.nanmax(tab.loc[c])], [i, i], color=T["grid"], lw=2,
                zorder=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{c} (n {nall[c]:,}" + (f" -> {int(nclean[c]):,})" if c in nclean
                        and nclean[c] != nall[c] else ")") for c in order], fontsize=7.4)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("share of attack flows caught on the unseen day (gradient boosting)")
    ax.set_title("Removing defective flows: effect on per-class recall")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=1, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    save(fig, "m3-rq5-defects-by-class")


# --- FIGURE 12: RQ2 MULTI-CLASS SEPARABILITY --------------------------------------------
def fig_rq2():
    # LEFT: PER-CLASS F1 UNDER THE THREE VARIANTS. RIGHT: CONFUSION MATRIX FOR V2 (NO WINDOW FIELDS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                 gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.55})
    var = [("V1_all_features", "All 64 features", T["context"]),
           ("V2_no_init_win", "Without Init_Win_bytes_*", T["blue"]),
           ("V3_no_init_win_no_timeout_rows", "...and without 120 s-capped flows", T["orange"])]
    order = (rq2f1[rq2f1["variant"] == "V2_no_init_win"].sort_values("f1")["attack_class"].tolist())
    yy = np.arange(len(order))
    for k, (v, lab, col) in enumerate(var):
        s = rq2f1[rq2f1["variant"] == v].set_index("attack_class").reindex(order)["f1"]
        a1.barh(yy + (1 - k) * 0.26, s, height=0.26, color=col, label=lab)
    a1.set_yticks(yy)
    a1.set_yticklabels(order, fontsize=7.8)
    a1.set_xlim(0, 1.0)
    a1.set_xlabel("F1 (5-fold cross-validation, attack flows only)")
    a1.set_title("How well each attack class is told apart")
    a1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), fontsize=7.5, ncol=1)
    a1.grid(axis="y", visible=False)
    # CONFUSION MATRIX: ROW = TRUE CLASS, VALUE = SHARE PREDICTED AS EACH COLUMN CLASS
    cm = rq2cm.loc[order[::-1], order[::-1]]
    a2.imshow(cm.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            if v >= 0.005:
                a2.text(j, i, f"{v:.0%}" if v >= 0.01 else "<1%", ha="center", va="center",
                        fontsize=6.3, color=ink_on(SEQ, v))
    short = [c.replace("Web Attack - ", "Web ").replace("DoS ", "") for c in cm.columns]
    a2.set_xticks(range(len(short)))
    a2.set_xticklabels(short, rotation=45, ha="right", fontsize=7.2)
    a2.set_yticks(range(len(short)))
    a2.set_yticklabels(short, fontsize=7.2)
    a2.set_xlabel("predicted class")
    a2.set_ylabel("true class")
    a2.grid(False)
    a2.set_title("Where the errors go (without Init_Win_bytes_*)")
    save(fig, "m3-rq2-attack-separability")


# --- FIGURE 13: RQ2 PERMUTATION IMPORTANCE ------------------------------------------------
def fig_rq2_importance():
    # TOP 12 FEATURES BY IMPORTANCE WITH THE WINDOW FIELDS PRESENT (V1) AND ABSENT (V2)
    # SHARED X-AXIS SO THE TWO PANELS' BAR LENGTHS ARE DIRECTLY COMPARABLE
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, gridspec_kw={"wspace": 0.75})
    for ax, (v, title, col) in zip(axes, [("V1_all_features", "All features", T["context"]),
                                          ("V2_no_init_win", "Without Init_Win_bytes_*", T["blue"])]):
        s = rq2imp[rq2imp["variant"] == v].nlargest(12, "importance")[::-1]
        ax.barh(s["feature"], s["importance"], xerr=s["std"], color=col,
                error_kw={"ecolor": T["ink2"], "lw": 0.8})
        ax.set_title(title)
        ax.set_xlabel("drop in macro-F1 when the feature is shuffled")
        ax.tick_params(axis="y", labelsize=7.6)
        ax.grid(axis="y", visible=False)
    save(fig, "m3-rq2-feature-importance")


# DRAW EVERY FIGURE IN BOTH THEMES
for theme in ["light", "dark"]:
    apply_theme(theme)
    fig_rq4_headline()
    fig_rq4_decomposition()
    fig_rq4_per_class()
    fig_rq5()
    fig_rq2()
    fig_rq2_importance()
print("figures written to", FIG_DIR)
