#!/usr/bin/env python3
"""Dedicated 2x2-ablation figure for the Lessons Learned paper.

Four short fine-tunes from one common checkpoint cross the supervised target (corrected vs
pre-correction prediction) with the imbalance-penalty weight (0.0 vs 0.1). Only cell A
(corrected supervision, no penalty) runs away; flipping either knob (B or C), or both (D),
pins drift near zero.

(a) Required water-budget correction vs epoch, one color/marker per cell.
(b) Raw global precipitation sink vs epoch for the same four cells, against the CAM6 mean
    sink. Cell A's raw sink collapses while the three stable cells hold near the reference,
    the physical consequence behind the drift in (a).

Reads experiments/figure_data/screen_{A,B,C,D}.csv and the cached CAM6 truth budget.
Writes experiments/figures/fig_ablation.{pdf,png}.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "figure_data")
OUT = os.path.join(HERE, "fig_ablation")

PSINK_SCALE = 1e10
C_BROKEN = "#D55E00"
C_FIXED = "#0072B2"
C_TRUTH = "#009E73"

# (cell, color, linestyle, marker, label) — warm = corrected supervision, cool = pre-correction;
# darker shade = no penalty, lighter = penalty. Matches the old combined-figure encoding.
ABLATION = [
    ("A", C_BROKEN, "-", "o", "corrected, no penalty"),
    ("B", "#E69F00", "--", "s", "corrected, penalty"),
    ("C", "#56B4E9", "-", "^", "pre-correction, no penalty"),
    ("D", C_FIXED, "--", "D", "pre-correction, penalty"),
]


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def tag(ax, t):
    ax.text(-0.02, 1.05, t, transform=ax.transAxes, fontweight="bold", fontsize=10,
            va="bottom", ha="right")


def main():
    style()
    truth = pd.read_csv(os.path.join(DATA, "truth_water_budget_1980.csv"))
    tm, ts = truth["drift_pct"].mean(), truth["drift_pct"].std()
    p_truth = truth["P_sink"].mean()
    cells = {c: pd.read_csv(os.path.join(DATA, f"screen_{c}.csv")) for c in "ABCD"}
    cells = {c: df[df["epoch"] >= 0.25] for c, df in cells.items()}  # drop cold-start transient

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.90, bottom=0.15, wspace=0.27)

    # (a) required correction vs epoch
    ax = ax_a
    ax.axhspan(tm - ts, tm + ts, color=C_TRUTH, alpha=0.15, zorder=0)
    ax.axhline(tm, color=C_TRUTH, lw=1.0, zorder=1)
    ax.axhline(0, color="0.6", lw=0.6)
    for c, color, ls, mk, lab in ABLATION:
        df = cells[c]
        ax.plot(df["epoch"], df["drift_pct"], color=color, ls=ls, marker=mk, ms=2.6,
                mew=0, label=f"{c}: {lab}")
    ax.legend(loc="center left", handlelength=2.0)
    ax.set_ylim(-12, 78)
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Required water-budget correction (%)")
    tag(ax, "(a)")

    # (b) raw precipitation sink vs epoch, same four cells. Cell colors are established in
    # (a); here only the CAM6 reference needs a legend entry.
    ax = ax_b
    ax.axhline(p_truth / PSINK_SCALE, color=C_TRUTH, lw=1.1, ls="--", label="CAM6 mean sink")
    for c, color, ls, mk, lab in ABLATION:
        df = cells[c]
        ax.plot(df["epoch"], df["P_sink"] / PSINK_SCALE, color=color, ls=ls, marker=mk,
                ms=2.6, mew=0)
    ax.annotate("A", xy=(cells["A"]["epoch"].iloc[-1], cells["A"]["P_sink"].iloc[-1] / PSINK_SCALE),
                xytext=(3, -8), textcoords="offset points", color=C_BROKEN, fontsize=8,
                fontweight="bold")
    ax.legend(loc="center left")
    ax.set_xlabel("Training epoch"); ax.set_ylabel(r"Precip sink ($10^{10}\ \mathrm{kg\,s^{-1}}$)")
    tag(ax, "(b)")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png  "
          f"(ablation pts: " + ", ".join(f"{c}={len(cells[c])}" for c in 'ABCD') + ")")


if __name__ == "__main__":
    main()
