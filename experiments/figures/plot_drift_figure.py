#!/usr/bin/env python3
"""Build the global water-budget drift figure for the Lessons Learned paper.

Reads the small cached trajectories in experiments/figure_data/ (parsed once from the
training logs) and the CAM6 ground-truth budget, and writes a clean, title-free,
colorblind-safe figure to experiments/figures/drift_comparison_combined.{pdf,png}.

Journal conventions applied: no figure title (the caption carries it), panel labels (a)-(c)
as bold corner tags, legends and annotations placed off the data, Okabe-Ito palette,
vector (PDF) output, P_sink axis in the correct 10^10 kg/s units.

Run from the repo root:  python experiments/figures/plot_drift_figure.py
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "figure_data")
OUT = os.path.join(HERE, "drift_comparison_combined")

STEPS_PER_EPOCH = 150
PSINK_SCALE = 1e10  # global precipitation sink is O(1.7e10) kg/s

# Okabe-Ito colorblind-safe palette
C_BROKEN = "#D55E00"  # vermillion  — feedback (loss on corrected output)
C_FIXED = "#0072B2"   # blue        — decoupled loss + conservation penalty
C_TRUTH = "#009E73"   # bluish green — CAM6 ground truth


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # editable text in vector output
    })


def panel_tag(ax, tag):
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes,
            fontweight="bold", fontsize=10, va="bottom", ha="right")


def epoch_axis(df):
    return df["global_step"] / STEPS_PER_EPOCH


def main():
    style()
    broken = pd.read_csv(os.path.join(DATA, "drift_broken.csv"))
    fixed = pd.read_csv(os.path.join(DATA, "drift_fixed.csv"))
    truth = pd.read_csv(os.path.join(DATA, "truth_water_budget_1980.csv"))

    t_mean, t_std = truth["drift_pct"].mean(), truth["drift_pct"].std()
    t_psink = truth["P_sink"].mean() / PSINK_SCALE
    trans = fixed["epoch"].min()  # transition epoch (penalty enabled)

    eb, ef = epoch_axis(broken), epoch_axis(fixed)

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                          hspace=0.42, wspace=0.30,
                          left=0.10, right=0.97, top=0.95, bottom=0.09)
    ax_d = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1])

    # ---- (a) drift trajectory --------------------------------------------------
    ax = ax_d
    ax.axhspan(t_mean - t_std, t_mean + t_std, color=C_TRUTH, alpha=0.15, zorder=0)
    ax.axhline(t_mean, color=C_TRUTH, lw=1.1, zorder=1)
    ax.axhline(0, color="0.6", lw=0.6, zorder=1)
    ax.axvline(trans, color="0.35", lw=0.9, ls="--", zorder=2)

    ax.plot(eb, broken["drift_pct"], color=C_BROKEN, marker="o", ms=2.5,
            label="Loss on corrected output (feedback)", zorder=3)
    ax.plot(ef, fixed["drift_pct"], color=C_FIXED, marker="o", ms=2.5,
            label="Decoupled loss + penalty", zorder=3)

    truth_patch = mpatches.Patch(color=C_TRUTH, alpha=0.5,
                                 label=f"CAM6 truth ({t_mean:.2f}% ± {t_std:.2f}%)")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [truth_patch], lab + [truth_patch.get_label()],
              loc="upper right", bbox_to_anchor=(0.99, 0.99))
    ax.annotate("conservation penalty enabled",
                xy=(trans, 6), xytext=(trans + 12, 14),
                fontsize=7.5, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Water-budget drift (%)")
    ax.set_ylim(-6, 30)
    panel_tag(ax, "(a)")

    # ---- (b) precipitation sink ------------------------------------------------
    ax = ax_p
    ax.plot(eb, broken["P_sink"] / PSINK_SCALE, color=C_BROKEN, label="feedback")
    ax.plot(ef, fixed["P_sink"] / PSINK_SCALE, color=C_FIXED, label="decoupled + penalty")
    ax.axhline(t_psink, color=C_TRUTH, lw=1.1, ls="--", label="CAM6 truth")
    ax.axvline(trans, color="0.35", lw=0.9, ls="--")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"Precip. sink ($10^{10}\ \mathrm{kg\,s^{-1}}$)")
    ax.legend(loc="lower right")
    panel_tag(ax, "(b)")

    # ---- (c) drift distribution ------------------------------------------------
    ax = ax_h
    bins = np.linspace(-8, 30, 40)
    ax.hist(truth["drift_pct"], bins=bins, density=True, color=C_TRUTH, alpha=0.55,
            label="CAM6 truth")
    ax.hist(broken["drift_pct"], bins=bins, density=True, color=C_BROKEN, alpha=0.55,
            label="feedback")
    ax.hist(fixed[fixed["epoch"] > trans]["drift_pct"], bins=bins, density=True,
            color=C_FIXED, alpha=0.6, label="decoupled + penalty")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Water-budget drift (%)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    panel_tag(ax, "(c)")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png")
    print(f"truth drift {t_mean:.3f}% ± {t_std:.2f}% | transition epoch {trans}")


if __name__ == "__main__":
    main()
