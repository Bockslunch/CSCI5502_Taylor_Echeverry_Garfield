"""Build the Milestone 2 EDA figures from the cleaned CICIDS2017 data.

Usage (from the repository root):
    python src/m2_eda_figures.py data/processed figures docs/assets

Needs cicids2017_clean.pkl, cleaning_log.json, quality_issues_by_class.csv and
m2_stats.json from m2_rebuild_clean.py and m2_quality_stats.py. Writes each
figure twice at 300 dpi: <name>.png for light backgrounds and <name>-dark.png
for the website's dark theme. The optional third argument also copies every
PNG into the website's assets folder (GitHub Pages only publishes docs/).
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage

# READ INPUT AND OUTPUT FOLDERS FROM THE COMMAND LINE (THE SITE-ASSETS FOLDER IS OPTIONAL)
DATA_DIR, FIG_DIR = Path(sys.argv[1]), Path(sys.argv[2])
SITE_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else None
FIG_DIR.mkdir(parents=True, exist_ok=True)

# LOAD THE CLEANED DATA AND THE PRE-COMPUTED TABLES
df = pd.read_pickle(DATA_DIR / "cicids2017_clean.pkl")
clean_log = json.loads((DATA_DIR / "cleaning_log.json").read_text())
issues = pd.read_csv(DATA_DIR / "quality_issues_by_class.csv", index_col="Label")
stats = json.loads((DATA_DIR / "m2_stats.json").read_text())

# --- THEMES ----------------------------------------------------------------------

# TWO THEMES MATCHED TO THE WEBSITE'S LIGHT (#FFFFFF) AND DARK (#14171A) BACKGROUNDS.
# THE THREE SERIES COLORS ARE THE FIRST THREE SLOTS OF THE VALIDATED REFERENCE PALETTE
# (THE ONLY THREE THAT PASS THE ALL-PAIRS COLORBLIND CHECK); EACH MODE HAS ITS OWN STEPS.
# SEQ = ONE-HUE RAMP FROM "NEAR ZERO" (CLOSE TO THE SURFACE) TO "HIGH"; DIV = BLUE <- GRAY -> RED.
THEMES = {
    "light": dict(
        suffix="", surface="#ffffff", ink="#1a1d21", ink2="#5a626c", grid="#e4e3df",
        blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", context="#d6d5d0",
        seq=["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"],
        div=["#1c5cab", "#86b6ef", "#f0efec", "#ec9a99", "#b83232"],
    ),
    "dark": dict(
        suffix="-dark", surface="#14171a", ink="#e7eaed", ink2="#a3acb6", grid="#2c2f33",
        blue="#3987e5", orange="#d95926", aqua="#199e70", context="#3a3e44",
        seq=["#1c232b", "#184f95", "#3987e5", "#86b6ef", "#cde2fb"],
        div=["#6da7ec", "#35587f", "#383835", "#8a3d3d", "#e66767"],
    ),
}


# SET THE MODULE-LEVEL COLOR NAMES AND MATPLOTLIB DEFAULTS FOR ONE THEME
def apply_theme(name):
    global THEME, SURFACE, INK, INK2, GRID, BLUE, ORANGE, AQUA, CONTEXT, SEQ, DIV
    THEME = THEMES[name]
    SURFACE, INK, INK2, GRID = THEME["surface"], THEME["ink"], THEME["ink2"], THEME["grid"]
    BLUE, ORANGE, AQUA, CONTEXT = THEME["blue"], THEME["orange"], THEME["aqua"], THEME["context"]
    SEQ = LinearSegmentedColormap.from_list(f"seq-{name}", THEME["seq"])
    DIV = LinearSegmentedColormap.from_list(f"div-{name}", THEME["div"])
    # RECESSIVE AXES, LIGHT GRID AND NEUTRAL TEXT ON EVERY FIGURE
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# PICK DARK OR LIGHT TEXT FOR A HEATMAP CELL FROM THE CELL COLOR'S LUMINANCE (WORKS IN BOTH THEMES)
def ink_on(cmap, position):
    r, g, b, _ = cmap(min(max(position, 0.0), 1.0))
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "#ffffff"


# CLASS ORDER USED ACROSS FIGURES: LARGEST TO SMALLEST AFTER CLEANING
CLASS_ORDER = df["Label"].value_counts().index.tolist()

# CAPTURE FILES IN CHRONOLOGICAL ORDER WITH SHORT DISPLAY NAMES
FILE_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday-Morning-WebAttacks",
              "Thursday-Afternoon-Infiltration", "Friday-Morning",
              "Friday-Afternoon-PortScan", "Friday-Afternoon-DDoS"]
FILE_SHORT = ["Mon", "Tue", "Wed", "Thu AM", "Thu PM", "Fri AM", "Fri PM (PortScan)", "Fri PM (DDoS)"]


# SAVE HELPER: TIGHT BOUNDING BOX AT PRINT RESOLUTION, THEN FREE THE FIGURE
def save(fig, name):
    out = FIG_DIR / name.replace(".png", f"{THEME['suffix']}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if SITE_DIR is not None:
        SITE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, SITE_DIR / out.name)


# DRAW ALL SEVEN FIGURES WITH WHICHEVER THEME IS ACTIVE
def draw_figures():
    # --- FIGURE 1: CLASS COUNTS BEFORE AND AFTER CLEANING (RQ1) ----------------------

    # LINE UP RAW COUNTS (BROKEN DASH FIXED) WITH CLEANED COUNTS IN THE SAME CLASS ORDER
    raw = pd.Series({k.replace("�", "-"): v for k, v in clean_log["raw_label_counts"].items()})
    clean = df["Label"].value_counts()
    raw, clean = raw.reindex(CLASS_ORDER), clean.reindex(CLASS_ORDER)

    # DRAW RAW AS A THIN MUTED BAR BEHIND THE CLEANED BAR SO THE LOSS IS VISIBLE AS A GAP
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    y = np.arange(len(CLASS_ORDER))
    ax.barh(y, raw, height=0.78, color=CONTEXT, label="As downloaded")
    ax.barh(y, clean, height=0.5, color=BLUE, label="After cleaning")

    # DIRECT-LABEL EACH BAR WITH THE CLEANED COUNT AND THE PERCENT REMOVED (ONLY WHERE >= 1%)
    for yi, (r, c) in enumerate(zip(raw, clean)):
        lost = (1 - c / r) * 100
        txt = f"{c:,}" + (f"   (-{lost:.0f}%)" if lost >= 1 else "")
        ax.text(r * 1.25, yi, txt, va="center", fontsize=7.5, color=INK2)

    # LOG SCALE BECAUSE THE CLASSES SPAN FIVE ORDERS OF MAGNITUDE
    ax.set_xscale("log")
    ax.set_xlim(1, 2e8)
    ax.set_yticks(y, CLASS_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Flow records (log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Figure 1. Class counts before and after cleaning")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "rq1-class-counts-before-after.png")

    # --- FIGURE 2: CAPTURE FILE x CLASS COMPOSITION (RQ1, RQ4) -----------------------

    # COUNT CLEANED ROWS PER FILE AND CLASS; ZEROS BECOME BLANK CELLS
    ct = pd.crosstab(df["source_file"], df["Label"]).reindex(index=FILE_ORDER, columns=CLASS_ORDER)
    masked = ct.where(ct > 0)

    # HEATMAP WITH A LOG COLOR SCALE SO 11 HEARTBLEED ROWS AND 500K BENIGN ROWS ARE BOTH VISIBLE
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    im = ax.imshow(masked.values, cmap=SEQ, norm=LogNorm(vmin=1, vmax=ct.values.max()), aspect="auto")
    ax.grid(False)

    # PRINT THE COUNT IN EVERY NON-EMPTY CELL, LIGHT TEXT ON DARK CELLS FOR CONTRAST
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.iat[i, j]
            if v > 0:
                label = f"{v/1000:.0f}k" if v >= 10_000 else f"{v:,}"
                ax.text(j, i, label, ha="center", va="center", fontsize=6.2,
                        color=ink_on(SEQ, np.log10(v) / np.log10(ct.values.max())))

    # LABEL AXES, ROTATE CLASS NAMES, AND ADD A COLOR KEY
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=50, ha="right", fontsize=7)
    ax.set_yticks(range(len(FILE_ORDER)), FILE_SHORT, fontsize=7.5)
    ax.set_xlabel("Traffic class")
    ax.set_ylabel("Capture file (July 3–7, 2017)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Flow records (log)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Figure 2. Every attack class appears in exactly one capture file")
    save(fig, "rq1-rq4-class-by-capture-file.png")

    # --- FIGURE 3: DATA-QUALITY ISSUES BY CLASS (SECTION 4) --------------------------

    # PERCENT OF EACH CLASS AFFECTED BY EACH ISSUE, ROWS IN THE SHARED CLASS ORDER
    iss = issues.reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    im = ax.imshow(iss.values, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    ax.grid(False)

    # WRITE THE PERCENTAGE IN EVERY CELL (BLANK FOR EXACT ZERO) SO VALUES NEVER RELY ON COLOR
    for i in range(iss.shape[0]):
        for j in range(iss.shape[1]):
            v = iss.iat[i, j]
            if v > 0:
                label = f"{v:.0f}%" if v >= 1 else f"{v:.2f}%"
                ax.text(j, i, label, ha="center", va="center", fontsize=6.8,
                        color=ink_on(SEQ, v / 100))

    # AXIS LABELS: ISSUES ACROSS THE TOP SO THE LONG NAMES READ HORIZONTALLY; ADD CLASS SIZES TO ROWS
    ax.xaxis.tick_top()
    HEAD = ["Zero duration\n(removed)", "Exact duplicate\n(removed)", "Same features,\nother label",
            "Init_Win_bytes_\nforward = -1", "Flow Duration\n≥ 119 s", "No forward\npayload"]
    ax.set_xticks(range(iss.shape[1]), HEAD, fontsize=6.8, rotation=0)
    ax.set_yticks(range(len(CLASS_ORDER)), [f"{c}  (n={clean[c]:,})" for c in CLASS_ORDER], fontsize=7)
    ax.tick_params(axis="x", length=0, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% of class rows", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Figure 3. Data-quality issues are concentrated in specific classes", pad=34)
    save(fig, "quality-issues-by-class.png")

    # --- FIGURE 4: BENIGN VS ATTACK DISTRIBUTIONS FOR KEY FEATURES (RQ1, RQ2) --------

    # SIX FEATURES SPANNING DURATION, VOLUME, SIZE, TIMING AND THE -1 WINDOW CODE
    PANELS = [("Flow Duration", "Flow Duration (µs)"),
              ("Total Fwd Packets", "Total Fwd Packets"),
              ("Total Length of Fwd Packets", "Total Length of Fwd Packets (bytes)"),
              ("Packet Length Mean", "Packet Length Mean (bytes)"),
              ("Flow IAT Mean", "Flow IAT Mean (µs)"),
              ("Init_Win_bytes_forward", "Init_Win_bytes_forward (bytes; -1 = code)")]

    # CLASS-BALANCED ATTACK SAMPLE: AT MOST 2,000 ROWS PER ATTACK CLASS SO DOS HULK/DDOS DON'T DOMINATE
    attacks = df[df["is_attack"] == 1]
    balanced = attacks.groupby("Label", group_keys=False).apply(
        lambda s: s.sample(min(len(s), 2_000), random_state=5502))
    benign_rows = df[df["is_attack"] == 0]

    # EMPIRICAL CDF FROM 600 QUANTILES (EXACT ENOUGH, AND AVOIDS PLOTTING 2.5M POINTS)
    qs = np.linspace(0, 1, 600)
    def ecdf(values):
        return np.quantile(values, qs), qs

    # THREE LINES PER PANEL: BENIGN, ALL ATTACK ROWS, CLASS-BALANCED ATTACKS (DASHED FOR SECONDARY ENCODING)
    SERIES = [(benign_rows, "Benign", BLUE, "-"),
              (attacks, "Attack (all rows)", ORANGE, "-"),
              (balanced, "Attack (class-balanced)", AQUA, "--")]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6), sharey=True)
    for ax, (col, label) in zip(axes.flat, PANELS):
        for frame, name, color, ls in SERIES:
            x, p = ecdf(frame[col].values)
            ax.plot(x, p, color=color, lw=1.6, ls=ls, label=name)
        # SYMLOG HANDLES THE ZEROS AND -1 CODES THAT A PURE LOG AXIS WOULD DROP
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel(label, fontsize=7.2)
        ax.tick_params(labelsize=6.5)
        # PRINT THE KS DISTANCE (BENIGN VS ALL ATTACK ROWS) IN THE CORNER OF EACH PANEL
        ks_all = pd.read_csv(DATA_DIR / "ks_benign_vs_attack.csv", index_col=0)["ks"]
        ax.set_title(f"KS benign vs. attack = {ks_all[col]:.2f}", fontsize=7, fontweight="normal",
                     color=INK2)
        # READABLE TICKS FOR THE WINDOW PANEL, WHERE -1 IS A CODE RATHER THAN A SIZE
        if col == "Init_Win_bytes_forward":
            ax.set_xticks([-1, 0, 10, 1e3, 1e5], ["-1", "0", "10", "1k", "100k"])
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of flows ≤ x")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Figure 4. Benign vs. attack distributions (cumulative, symlog x-axis)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, "rq2-benign-vs-attack-ecdf.png")

    # --- FIGURE 5: DESTINATION PORT CONCENTRATION (RQ2; REDRAWN FROM TYLER'S ANALYSIS) --

    # DISTINCT PORTS AND TOP-PORT SHARE PER CLASS FROM THE STATS FILE
    port = pd.DataFrame(stats["dest_port_by_class"]).T.reindex(CLASS_ORDER)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 4.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(CLASS_ORDER))

    # LEFT PANEL: SHARE OF THE CLASS SENT TO ITS SINGLE MOST COMMON PORT, LABELED WITH THE PORT NUMBER
    a1.barh(y, port["top_share"].astype(float), height=0.6, color=BLUE)
    for yi, (s, p) in enumerate(zip(port["top_share"].astype(float), port["top_port"].astype(int))):
        txt = f"port {p}" if s >= 99 else f"{s:.1f}% · port {p}"
        a1.text(s + 2, yi, txt, va="center", fontsize=6.8, color=INK2)
    a1.set_xlim(0, 130)
    a1.set_xticks([0, 25, 50, 75, 100])
    a1.set_xlabel("% of class to its most common port")
    a1.set_yticks(y, CLASS_ORDER, fontsize=7.2)
    a1.invert_yaxis()
    a1.grid(axis="y", visible=False)

    # RIGHT PANEL: HOW MANY DISTINCT PORTS EACH CLASS TOUCHES (LOG SCALE: 1 TO 53,788)
    a2.barh(y, port["n_ports"].astype(float), height=0.6, color=BLUE)
    for yi, n in enumerate(port["n_ports"].astype(int)):
        a2.text(n * 1.3, yi, f"{n:,}", va="center", fontsize=6.8, color=INK2)
    a2.set_xscale("log")
    a2.set_xlim(0.8, 1e6)
    a2.set_xlabel("Distinct destination ports (log)")
    a2.grid(axis="y", visible=False)
    fig.suptitle("Figure 5. Most attack classes target a single destination port",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "rq2-destination-port.png")

    # --- FIGURE 6: THE 120-SECOND FLOW-DURATION CEILING (RQ2; REDRAWN FROM TYLER'S ANALYSIS) --

    # ALL FLOWS IN THE LAST TWO SECONDS BEFORE THE CICFLOWMETER TIMEOUT
    dur_s = df["Flow Duration"] / 1e6
    near = dur_s[dur_s >= 118]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})

    # LEFT: HISTOGRAM WITH 20 MS BINS, A MARKER AT 120 S, AND THE TRUE MAXIMUM ANNOTATED
    a1.hist(near, bins=np.arange(118, 120.02, 0.02), color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    a1.axvline(120, color=INK, lw=1, ls=":")
    a1.text(119.97, a1.get_ylim()[1] * 0.92, "120 s timeout", ha="right", fontsize=7, color=INK)
    a1.set_xlabel("Flow Duration (s)")
    a1.set_ylabel("Flow records")
    a1.set_title("Durations pile up just under 120 s", fontsize=8.5)
    a1.grid(axis="x", visible=False)

    # RIGHT: PERCENT OF EACH CLASS WITHIN ONE SECOND OF THE TIMEOUT (ONLY CLASSES WITH ANY)
    share = (df.assign(t=dur_s >= 119).groupby("Label")["t"].mean() * 100)
    share = share[share > 0].sort_values()
    a2.barh(range(len(share)), share.values, height=0.6, color=ORANGE)
    for yi, v in enumerate(share.values):
        a2.text(v + 1.5, yi, ("<0.01%" if v < 0.01 else f"{v:.2f}%") if v < 1 else f"{v:.0f}%", va="center", fontsize=6.8, color=INK2)
    a2.set_yticks(range(len(share)), share.index, fontsize=7)
    a2.set_xlim(0, 115)
    a2.set_xlabel("% of class with duration ≥ 119 s")
    a2.set_title("…and hit Heartbleed hardest", fontsize=8.5)
    a2.grid(axis="y", visible=False)
    fig.suptitle("Figure 6. The 120 s CICFlowMeter timeout caps Flow Duration",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "rq2-flow-duration-timeout.png")

    # --- FIGURE 7: PER-CLASS FEATURE PROFILES (RQ3) ----------------------------------

    # BEHAVIORAL FEATURES ONLY: NO PORT, NO -1-CODED WINDOW FIELDS, NO 0/1 FLAGS
    PROFILE = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
               "Total Length of Fwd Packets", "Total Length of Bwd Packets",
               "Fwd Packet Length Max", "Bwd Packet Length Mean", "Packet Length Std",
               "Flow IAT Mean", "Flow IAT Max", "Fwd Packets/s", "Bwd Packets/s",
               "Down/Up Ratio", "Active Mean", "Idle Mean"]

    # MEDIAN OF LOG10(1 + VALUE) PER CLASS: MEDIAN BECAUSE SKEWED, LOG BECAUSE MULTIPLICATIVE SCALES
    logged = np.log10(1 + df[PROFILE].clip(lower=0))
    prof = logged.groupby(df["Label"]).median()

    # STANDARDIZE EACH FEATURE ACROSS CLASSES SO COLOR MEANS "ABOVE / BELOW THE TYPICAL CLASS"
    z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)

    # ORDER CLASSES BY WARD CLUSTERING SO SIMILAR PROFILES SIT NEXT TO EACH OTHER
    tree = linkage(z.values, method="ward")
    order = z.index[leaves_list(tree)]
    groups = pd.Series(fcluster(tree, 4, criterion="maxclust"), index=z.index).loc[order]
    z = z.loc[order]

    # DRAW THE HEATMAP, CLIPPED TO +/-2.5 SD, WITH CLASS SIZES IN THE ROW LABELS
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(z.values.clip(-2.5, 2.5), cmap=DIV, vmin=-2.5, vmax=2.5, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(PROFILE)), PROFILE, rotation=50, ha="right", fontsize=7)
    ax.set_yticks(range(len(order)), [f"{c}  (n={clean[c]:,})" for c in order], fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("SD above (red) / below (blue) the class average", color=INK2)
    cb.outline.set_visible(False)
    for i in np.where(groups.values[1:] != groups.values[:-1])[0]:
        ax.axhline(i + 0.5, color=SURFACE, lw=2.5)
    ax.set_title("Figure 7. Median flow profile per class (Ward clustering, 4 groups)")
    save(fig, "rq3-class-profiles.png")
    # RETURN THE PROFILE TABLE SO IT CAN BE SAVED ONCE (IT DOES NOT DEPEND ON THE THEME)
    return z.assign(group=groups)


# RENDER THE LIGHT SET, THEN THE DARK SET
for theme_name in THEMES:
    apply_theme(theme_name)
    profile = draw_figures()

# SAVE THE PROFILE TABLE AND ROW ORDER SO THE WRITE-UP CAN CITE THE GROUPINGS
profile.round(2).to_csv(DATA_DIR / "class_profile_z.csv")
print("figures written to", FIG_DIR)
