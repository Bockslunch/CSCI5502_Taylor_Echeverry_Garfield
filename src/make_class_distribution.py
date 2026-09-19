"""
Build the class-distribution figure for RQ1.

Reads only the label column from each CICIDS2017 CSV, counts records per class,
and renders a horizontal bar chart on a log scale — the range spans roughly five
orders of magnitude, so a linear axis would collapse every attack class into the
baseline.

Outputs
-------
figures/class_distribution.csv        counts per label per capture file
figures/class_distribution.png        light-mode chart
figures/class_distribution-dark.png   dark-mode chart
docs/assets/                          copies of both PNGs, so the website can
                                      serve them (GitHub Pages only publishes
                                      the docs/ folder)

Usage
-----
    python src/make_class_distribution.py

Run from the repository root with the raw CSVs in data/raw/.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "figures"
SITE_ASSETS = ROOT / "docs" / "assets"

# Validated two-colour categorical pair. Both modes clear the colourblind
# separation and contrast gates against this site's surfaces — do not substitute
# by eye.
THEMES = {
    "light": dict(
        suffix="",
        surface="#ffffff",
        benign="#2a78d6",
        attack="#eb6834",
        ink="#1a1d21",
        muted="#5a626c",
        grid="#e1e0d9",
    ),
    "dark": dict(
        suffix="-dark",
        surface="#14171a",
        benign="#3987e5",
        attack="#d95926",
        ink="#e7eaed",
        muted="#a3acb6",
        grid="#2c2c2a",
    ),
}


def find_label_column(path: Path, encoding: str) -> str:
    """Return the label column's name exactly as it appears in the file.

    The published CSVs have inconsistent header whitespace — the label column is
    ' Label' with a leading space in most files — so match on the stripped name.
    """
    header = pd.read_csv(path, nrows=0, encoding=encoding)
    for col in header.columns:
        if col.strip().casefold() == "label":
            return col
    raise ValueError(f"no label column found in {path.name}: {list(header.columns)}")


def normalize_label(value: str) -> str:
    """Collapse the Web Attack labels' mangled dash into a plain hyphen.

    'Web Attack \\x96 Brute Force' and its mojibake variants all refer to the same
    class; left alone they split into several near-duplicate categories.
    """
    text = re.sub(r"[^\x20-\x7E]+", " - ", str(value))
    return re.sub(r"\s+", " ", text).strip()


def read_labels(path: Path) -> pd.Series:
    """Read just the label column. Encoding varies between files."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "latin-1"):
        try:
            column = find_label_column(path, encoding)
            series = pd.read_csv(
                path, usecols=[column], encoding=encoding, low_memory=False
            )[column]
            return series.map(normalize_label)
        except (UnicodeDecodeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"could not read {path.name}: {last_error}")


def collect() -> pd.DataFrame:
    files = sorted(RAW.glob("*.csv"))
    if not files:
        sys.exit(
            f"No CSV files in {RAW}.\n"
            "Follow data/README.md to download CICIDS2017 first."
        )

    frames = []
    for path in files:
        labels = read_labels(path)
        counts = labels.value_counts().rename_axis("label").reset_index(name="records")
        counts.insert(0, "source_file", path.name)
        frames.append(counts)
        print(f"  {path.name:<52} {len(labels):>9,} rows")

    return pd.concat(frames, ignore_index=True)


def render(totals: pd.Series, theme: dict) -> Path:
    """One horizontal bar per class, largest at the top, log-scaled x axis."""
    labels = list(totals.index)
    values = list(totals.values)
    colors = [
        theme["benign"] if label.casefold() == "benign" else theme["attack"]
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(labels) + 1.9), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    positions = range(len(labels))
    ax.barh(positions, values, height=0.62, color=colors, zorder=3)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, max(values) * 12)

    for pos, value in zip(positions, values):
        ax.text(
            value * 1.35, pos, f"{value:,}",
            va="center", ha="left", fontsize=8.5, color=theme["muted"],
        )

    ax.set_xlabel("Flow records (log scale)", fontsize=9, color=theme["muted"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.grid(axis="x", color=theme["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=8.5, length=0)
    for tick in ax.get_yticklabels():
        tick.set_color(theme["ink"])

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=theme["benign"]),
        plt.Rectangle((0, 0), 1, 1, color=theme["attack"]),
    ]
    legend = ax.legend(
        handles, ["Benign", "Attack"],
        loc="lower right", frameon=False, fontsize=8.5, ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(theme["ink"])

    ax.set_title(
        "CICIDS2017 records per traffic class, all seven capture files",
        fontsize=11, color=theme["ink"], loc="left", pad=14,
    )

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / f"class_distribution{theme['suffix']}.png"
    fig.savefig(out, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print(f"Reading label columns from {RAW}")
    per_file = collect()

    FIGURES.mkdir(exist_ok=True)
    csv_path = FIGURES / "class_distribution.csv"
    per_file.to_csv(csv_path, index=False)

    totals = (
        per_file.groupby("label")["records"].sum().sort_values(ascending=False)
    )

    print(f"\n{len(totals)} classes, {totals.sum():,} records total")
    benign = totals.get("BENIGN", 0)
    print(f"  benign      {benign:>10,}  ({benign / totals.sum():.1%})")
    print(f"  malicious   {totals.sum() - benign:>10,}")
    print(f"  rarest      {totals.index[-1]} — {totals.iloc[-1]:,} records")
    print(f"  imbalance   {benign / max(totals.iloc[-1], 1):,.0f}:1 "
          "(most to least common)\n")

    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    for theme in THEMES.values():
        out = render(totals, theme)
        shutil.copy2(out, SITE_ASSETS / out.name)
        print(f"  wrote {out.relative_to(ROOT)} and docs/assets/{out.name}")

    print(f"  wrote {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
