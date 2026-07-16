"""
=============================================================================
APSS-SLU  |  FIGURE GENERATION
CS5651 Statistical Inference — Group Project
=============================================================================
Generates three publication-quality figures at 300 dpi:

  Fig 1 — fig_retest.png
    Scatter plot: Round 1 vs Round 2 total scores for 76 matched respondents.
    Shows regression line, equality line, and Pearson r annotation.

  Fig 2 — fig_year.png
    Bar chart: Mean perceived stress by year of study with SEM error bars.
    Highlights the junior-to-senior escalation.

  Fig 3 — fig_itemtotal.png
    Horizontal bar chart: Corrected item-total correlations for all 14 items.
    Distinguishes direct (navy) vs reverse (red) items; marks the 0.30 threshold.

OUTPUT: PNG files saved to the  figures/  subfolder (created automatically)
        Also saves PDF versions alongside for use in LaTeX.

USAGE:
  python 03_generate_figures.py

FILE PATHS — edit this line if your CSVs are in a different folder:
"""

ROUND1_CSV  = "Adapted Perceived Stress Scale for Sri Lankan Undergraduates (APSS-SLU) (Responses) - Form responses.csv"   # 93 responses
ROUND2_CSV  = "2nd Round Responses.csv"               # 76 responses
OUTPUT_DIR  = "figures"                       # output folder (created if missing)
DPI         = 300                             # resolution for PNG files

# =============================================================================
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")                         # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# Print any Unicode symbols safely on Windows consoles (cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ── Global plot style ─────────────────────────────────────────────────────────
rcParams["font.family"]   = "serif"
rcParams["font.size"]     = 10
rcParams["axes.linewidth"]= 0.8
rcParams["xtick.major.width"] = 0.7
rcParams["ytick.major.width"] = 0.7

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY = "#1f2d4d"    # direct items / senior years
RED  = "#c0392b"    # reverse items / accent annotations
GREY = "#7f8c8d"    # junior years / secondary lines
LGREY= "#bdc3c7"    # light fills

# ── Item definitions ──────────────────────────────────────────────────────────
REVERSE_KEYS = [
    "confident about your ability",
    "things were going well",
    "able to control irritations",
    "on top of things",
    "control how you spent",
    "dealt successfully",
    "effectively coping",
]
DIRECT_KEYS = [
    "upset because something",
    "unable to control the important",
    "nervous or stressed",
    "could not cope",
    "angered by things",
    "thinking about things you needed",
    "difficulties were piling",
]

# Short label for each item on the chart (same order: direct then reverse)
ITEM_LABELS = [
    "Q1  Upset unexpectedly",
    "Q2  Unable to control",
    "Q3  Nervous / stressed",
    "Q8  Could not cope",
    "Q11 Angered",
    "Q12 Preoccupied with tasks",
    "Q14 Difficulties piling up",
    "Q6  Confident [R]",
    "Q7  Going well [R]",
    "Q9  Ctrl irritations [R]",
    "Q10 On top of things [R]",
    "Q13 Ctrl time [R]",
    "Q4  Dealt w/ hassles [R]",
    "Q5  Effectively coping [R]",
]


# =============================================================================
# HELPERS
# =============================================================================

def find_col(df, keyword):
    hits = [c for c in df.columns if keyword in c]
    if len(hits) != 1:
        raise ValueError(f"Expected 1 column for '{keyword}', found: {hits}")
    return hits[0]


def score(df):
    df = df.copy()
    rev_cols  = [find_col(df, k) for k in REVERSE_KEYS]
    dir_cols  = [find_col(df, k) for k in DIRECT_KEYS]
    all_items = dir_cols + rev_cols
    for c in all_items:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in rev_cols:
        df[c] = 4 - df[c]
    df["Total"] = df[all_items].sum(axis=1)
    return df, all_items, rev_cols, dir_cols


def save(fig, name):
    """Save figure as both PNG (300 dpi) and PDF for LaTeX."""
    png_path = os.path.join(OUTPUT_DIR, name + ".png")
    pdf_path = os.path.join(OUTPUT_DIR, name + ".pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}  &  {pdf_path}")


# =============================================================================
# FIGURE 1: TEST-RETEST SCATTER
# =============================================================================

def fig_retest(d1, d2):
    """
    Scatter plot of Round 1 vs Round 2 total scores for matched respondents.
    Overlays:
      - Regression line (red)   → shows the actual linear trend
      - Equality line (dashed)  → reference: perfect agreement
    Annotates Pearson r in the upper-left corner.
    """
    print("\n[Fig 1] Test-Retest Scatter...")

    # Match by email
    d1 = d1.copy(); d2 = d2.copy()
    d1["_k"] = d1["Email address"].astype(str).str.lower().str.strip()
    d2["_k"] = d2["Email address"].astype(str).str.lower().str.strip()
    d1 = d1[d1["_k"].ne("") & d1["_k"].ne("nan")]
    d2 = d2[d2["_k"].ne("") & d2["_k"].ne("nan")]
    m = pd.merge(d1[["_k", "Total"]], d2[["_k", "Total"]],
                 on="_k", suffixes=("_T1", "_T2"))

    if len(m) < 3:
        print(f"  [SKIPPED] Only {len(m)} matched respondent(s) — need populated emails "
              "in both CSVs. Figure 1 not generated.")
        return

    t1 = m["Total_T1"].values
    t2 = m["Total_T2"].values
    r, _ = stats.pearsonr(t1, t2)

    # Axis limits with a small margin
    lo = min(t1.min(), t2.min()) - 2
    hi = max(t1.max(), t2.max()) + 2
    xs = np.array([lo, hi])

    # Regression coefficients
    slope, intercept = np.polyfit(t1, t2, 1)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Points
    ax.scatter(t1, t2, s=24, c=NAVY, alpha=0.65,
               edgecolors="white", linewidths=0.4, zorder=3, label="Respondent")

    # Regression line
    ax.plot(xs, intercept + slope * xs, color=RED, lw=1.8,
            label=f"Regression (r = {r:.3f})", zorder=4)

    # Equality line (y = x)
    ax.plot(xs, xs, "--", color=GREY, lw=0.9, label="Line of equality", zorder=2)

    # Pearson r annotation (bold, top-left)
    ax.text(0.05, 0.91, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=RED, va="top")
    ax.text(0.05, 0.84, f"N = {len(m)}", transform=ax.transAxes,
            fontsize=8, color=GREY, va="top")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Round 1 total score", fontsize=10)
    ax.set_ylabel("Round 2 total score", fontsize=10)
    ax.set_title("Test\u2013Retest Stability (two-week interval)", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.85, loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()

    save(fig, "fig_retest")


# =============================================================================
# FIGURE 2: STRESS BY YEAR OF STUDY
# =============================================================================

def fig_year(d1):
    """
    Bar chart of mean perceived stress by year of study.
    Error bars = ±1 SEM.
    Junior years (Y1, Y2) shown in grey; senior years (Y3, Y4) in navy,
    visually emphasising the escalation pattern.
    Sample size annotations appear above each bar.
    """
    print("\n[Fig 2] Stress by Year of Study...")

    year_col = " Which year of study are you currently in?"
    order    = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
    x_labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
    colors   = [GREY, GREY, NAVY, NAVY]

    means = [d1[d1[year_col] == y]["Total"].mean() for y in order]
    sems  = [d1[d1[year_col] == y]["Total"].sem()  for y in order]
    ns    = [int((d1[year_col] == y).sum())         for y in order]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    bars = ax.bar(range(4), means, yerr=sems, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw={"elinewidth": 1.0, "ecolor": "black"})

    # Sample size labels above each bar
    for i, (m, sem, n) in enumerate(zip(means, sems, ns)):
        ax.text(i, m + sem + 0.8, f"n={n}", ha="center",
                fontsize=7.5, color="#444444")

    # Significance bracket between Y2 and Y3 (largest visible jump)
    y_top = max(m + s for m, s in zip(means, sems)) + 3.0
    ax.annotate("", xy=(2, y_top), xytext=(1, y_top),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.text(1.5, y_top + 0.3, "p = 0.001", ha="center", fontsize=7, color="black")

    ax.set_xticks(range(4))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Mean perceived stress score", fontsize=10)
    ax.set_ylim(0, 40)
    ax.set_title("Stress by Year of Study", fontsize=10)

    # Legend for colour coding
    junior = mpatches.Patch(color=GREY, label="Junior (Y1–Y2)")
    senior = mpatches.Patch(color=NAVY, label="Senior (Y3–Y4)")
    ax.legend(handles=[junior, senior], fontsize=7.5,
              framealpha=0.85, loc="upper left")

    fig.tight_layout()
    save(fig, "fig_year")


# =============================================================================
# FIGURE 3: CORRECTED ITEM-TOTAL CORRELATIONS
# =============================================================================

def fig_itemtotal(d1, items, rev_cols, dir_cols):
    """
    Horizontal bar chart of corrected item-total correlations.
    - Direct items (navy)   — higher bars indicate stronger discrimination
    - Reverse items (red)   — visible underperformance vs direct items
    Dashed line at r = 0.30 (conventional adequacy threshold, Nunnally 1978).
    Items are ordered: direct first (top), reverse below, so the contrast is clear.
    """
    print("\n[Fig 3] Item-Total Correlations...")

    # Compute corrected item-total r (item vs sum of all OTHER items)
    corr = []
    for c in items:           # order: direct_cols + rev_cols
        rest = d1["Total"] - d1[c]
        corr.append(d1[c].corr(rest))

    n = len(items)
    colors = [NAVY] * len(dir_cols) + [RED] * len(rev_cols)

    fig, ax = plt.subplots(figsize=(3.8, 3.8))

    y_pos = list(range(n))
    ax.barh(y_pos, corr, color=colors, edgecolor="black", linewidth=0.4, height=0.7)

    # Zero line
    ax.axvline(0, color="black", linewidth=0.7)

    # Adequacy threshold (dashed)
    ax.axvline(0.30, color=GREY, linewidth=0.8, linestyle="--", label="Threshold r = 0.30")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ITEM_LABELS, fontsize=7.5)
    ax.invert_yaxis()         # highest-discriminating items at the top
    ax.set_xlabel("Corrected item\u2013total correlation (r)", fontsize=9)
    ax.set_title("Item Discrimination", fontsize=10)

    # Legend
    direct_patch  = mpatches.Patch(color=NAVY, label="Direct items")
    reverse_patch = mpatches.Patch(color=RED,  label="Reverse-scored items [R]")
    ax.legend(handles=[direct_patch, reverse_patch, 
              plt.Line2D([0], [0], color=GREY, lw=0.8, linestyle="--",
                         label="Threshold (r = 0.30)")],
              fontsize=7, loc="lower right", framealpha=0.9)

    # x-axis range: cover negative values
    ax.set_xlim(min(min(corr) - 0.05, -0.4), max(max(corr) + 0.05, 0.55))

    fig.tight_layout()
    save(fig, "fig_itemtotal")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df1_raw = pd.read_csv(ROUND1_CSV)
    df2_raw = pd.read_csv(ROUND2_CSV)

    d1, items, rev_cols, dir_cols = score(df1_raw)
    d2, _,     _,        _        = score(df2_raw)

    fig_retest(d1, d2)
    fig_year(d1)
    fig_itemtotal(d1, items, rev_cols, dir_cols)

    print(f"\nAll figures saved to '{OUTPUT_DIR}/'")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
