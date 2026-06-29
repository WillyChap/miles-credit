#!/usr/bin/env python3
"""Figure 1 for the Lessons Learned paper (2x2).

(a) Production fine-tune: drift ramps under the feedback, the penalty is enabled at epoch
    16, then drift holds near zero for tens of epochs.
(b) Controlled screen: two runs from the identical cp00092_extended checkpoint differing
    only in conservation_loss_weight (0.0 vs 0.1); the feedback runs away, the penalty
    holds drift at zero. This isolates the loss coupling as the cause.
(c) Global precipitation sink vs the CAM6 reference (production runs): the raw sink
    collapses under feedback and recovers under the penalty.
(d) Drift distributions for the feedback and penalty configurations vs the CAM6 truth.

Reads cached CSVs in experiments/figure_data/. Re-run after the screen jobs finish to
refresh (b). Writes experiments/figures/fig1.{pdf,png}.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "figure_data")
OUT = os.path.join(HERE, "fig1")

STEPS_PER_EPOCH = 150
PSINK_SCALE = 1e10
C_BROKEN = "#D55E00"
C_FIXED = "#0072B2"
C_TRUTH = "#009E73"


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def tag(ax, t):
    ax.text(-0.02, 1.05, t, transform=ax.transAxes, fontweight="bold", fontsize=10,
            va="bottom", ha="right")


def main():
    style()
    pb = pd.read_csv(os.path.join(DATA, "drift_broken.csv"))
    pf = pd.read_csv(os.path.join(DATA, "drift_fixed.csv"))
    truth = pd.read_csv(os.path.join(DATA, "truth_water_budget_1980.csv"))
    sb = pd.read_csv(os.path.join(DATA, "screen_broken.csv"))
    sf = pd.read_csv(os.path.join(DATA, "screen_fix.csv"))
    sb, sf = sb[sb["step"] > 1], sf[sf["step"] > 1]  # drop cold-start transient

    tm, ts = truth["drift_pct"].mean(), truth["drift_pct"].std()
    p_truth = truth["P_sink"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.94, bottom=0.09, hspace=0.45, wspace=0.27)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    def truthband(ax):
        ax.axhspan(tm - ts, tm + ts, color=C_TRUTH, alpha=0.15, zorder=0)
        ax.axhline(tm, color=C_TRUTH, lw=1.0, zorder=1)

    # (a) production fine-tune drift
    ax = ax_a
    truthband(ax); ax.axhline(0, color="0.6", lw=0.6)
    ax.plot(pb["global_step"] / STEPS_PER_EPOCH, pb["drift_pct"], color=C_BROKEN,
            label="feedback")
    ax.plot(pf["global_step"] / STEPS_PER_EPOCH, pf["drift_pct"], color=C_FIXED,
            label="decoupled + penalty")
    trans = pf["epoch"].min()
    ax.axvline(trans, color="0.35", lw=0.9, ls="--")
    ax.annotate("penalty enabled", xy=(trans, 6), xytext=(trans + 16, 16), fontsize=7,
                color="0.35", arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))
    ax.legend(loc="upper right"); ax.set_ylim(-8, 30)
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Water-budget drift (%)")
    tag(ax, "(a)")

    # (b) controlled screen drift
    ax = ax_b
    truthband(ax); ax.axhline(0, color="0.6", lw=0.6)
    ax.plot(sb["epoch"], sb["drift_pct"], color=C_BROKEN, marker="o", ms=2.5, label="weight 0.0")
    ax.plot(sf["epoch"], sf["drift_pct"], color=C_FIXED, marker="o", ms=2.5, label="weight 0.1")
    ax.legend(loc="center right", title="same checkpoint", title_fontsize=6.5)
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Water-budget drift (%)")
    tag(ax, "(b)")

    # (c) precipitation sink vs truth (production)
    ax = ax_c
    ax.plot(pb["global_step"] / STEPS_PER_EPOCH, pb["P_sink"] / PSINK_SCALE, color=C_BROKEN,
            label="feedback")
    ax.plot(pf["global_step"] / STEPS_PER_EPOCH, pf["P_sink"] / PSINK_SCALE, color=C_FIXED,
            label="decoupled + penalty")
    ax.axhline(p_truth / PSINK_SCALE, color=C_TRUTH, lw=1.1, ls="--", label="CAM6 truth")
    ax.axvline(trans, color="0.35", lw=0.9, ls="--")
    ax.set_xlabel("Training epoch"); ax.set_ylabel(r"Precip sink ($10^{10}\ \mathrm{kg\,s^{-1}}$)")
    ax.legend(loc="lower right")
    tag(ax, "(c)")

    # (d) drift distributions
    ax = ax_d
    bins = np.linspace(-8, 30, 40)
    ax.hist(truth["drift_pct"], bins=bins, density=True, color=C_TRUTH, alpha=0.55,
            label="CAM6 truth")
    ax.hist(pb["drift_pct"], bins=bins, density=True, color=C_BROKEN, alpha=0.55,
            label="feedback")
    ax.hist(pf[pf["epoch"] > trans]["drift_pct"], bins=bins, density=True, color=C_FIXED,
            alpha=0.6, label="decoupled + penalty")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Water-budget drift (%)"); ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    tag(ax, "(d)")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png  (screen pts: broken={len(sb)}, fix={len(sf)})")


if __name__ == "__main__":
    main()
