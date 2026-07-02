#!/usr/bin/env python3
"""Fine-tuning-run figure for the Lessons Learned paper (2x2).

The production fine-tune only: the corrected-output feedback and its recovery under
pre-correction supervision with a penalty. The controlled 2x2 ablation lives in its own
figure now (plot_fig_ablation.py).

(a) Required water-budget correction vs epoch: drift ramps under corrected-output
    supervision, the penalty is enabled at the handoff, then drift holds near zero.
(b) Correction distributions for the feedback and penalty configurations vs the CAM6 truth.
(c) Global precipitation sink vs the CAM6 reference: the raw sink collapses under feedback
    and recovers under the penalty.
(d) Only the water correction runs away: magnitudes of the global mass, water, and energy
    corrections (|r-1|, log scale) during the feedback run. Mass and energy stay near the
    CAM6 natural per-step spread; water climbs to ~24%.

Reads cached CSVs in experiments/figure_data/. Writes experiments/figures/fig_finetune.{pdf,png}.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "figure_data")
OUT = os.path.join(HERE, "fig_finetune")

STEPS_PER_EPOCH = 150
PSINK_SCALE = 1e10
C_BROKEN = "#D55E00"   # vermillion
C_FIXED = "#0072B2"    # blue
C_TRUTH = "#009E73"    # green
C_WATER = C_BROKEN     # water correction is the broken corrector
C_MASS = C_FIXED
C_ENERGY = "#CC79A7"   # reddish purple


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
    fx = pd.read_csv(os.path.join(DATA, "fixer_control_broken.csv"))

    tm, ts = truth["drift_pct"].mean(), truth["drift_pct"].std()
    p_truth = truth["P_sink"].mean()
    trans = pf["epoch"].min()

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
            label="corrected, no penalty")
    ax.plot(pf["global_step"] / STEPS_PER_EPOCH, pf["drift_pct"], color=C_FIXED,
            label="pre-correction + penalty")
    ax.axvline(trans, color="0.35", lw=0.9, ls="--")
    ax.annotate("penalty enabled", xy=(trans, 6), xytext=(trans + 16, 16), fontsize=7,
                color="0.35", arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))
    ax.legend(loc="upper right"); ax.set_ylim(-8, 30)
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Required water-budget correction (%)")
    tag(ax, "(a)")

    # (b) correction distributions
    ax = ax_b
    bins = np.linspace(-8, 30, 40)
    ax.hist(truth["drift_pct"], bins=bins, density=True, color=C_TRUTH, alpha=0.55,
            label="CAM6 truth")
    ax.hist(pb["drift_pct"], bins=bins, density=True, color=C_BROKEN, alpha=0.55,
            label="corrected, no penalty")
    ax.hist(pf[pf["epoch"] > trans]["drift_pct"], bins=bins, density=True, color=C_FIXED,
            alpha=0.6, label="pre-correction + penalty")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Required water-budget correction (%)"); ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    tag(ax, "(b)")

    # (c) precipitation sink vs truth
    ax = ax_c
    ax.plot(pb["global_step"] / STEPS_PER_EPOCH, pb["P_sink"] / PSINK_SCALE, color=C_BROKEN,
            label="corrected, no penalty")
    ax.plot(pf["global_step"] / STEPS_PER_EPOCH, pf["P_sink"] / PSINK_SCALE, color=C_FIXED,
            label="pre-correction + penalty")
    ax.axhline(p_truth / PSINK_SCALE, color=C_TRUTH, lw=1.1, ls="--", label="CAM6 mean sink")
    ax.axvline(trans, color="0.35", lw=0.9, ls="--")
    ax.set_xlabel("Training epoch"); ax.set_ylabel(r"Precip sink ($10^{10}\ \mathrm{kg\,s^{-1}}$)")
    ax.legend(loc="lower right")
    tag(ax, "(c)")

    # (d) only water runs away: mass/water/energy correction magnitudes (log)
    ax = ax_d
    nat = ts  # CAM6 natural per-step budget spread
    ep = fx["global_step"] / STEPS_PER_EPOCH
    ax.axhspan(1e-3, nat, color="0.85", zorder=0)
    ax.axhline(nat, color="0.6", lw=0.8, ls=":", zorder=1)
    ax.text(ep.max(), nat * 1.25, "CAM6 natural spread", ha="right", va="bottom",
            fontsize=6.5, color="0.45")
    ax.plot(ep, fx["water"], color=C_WATER, marker="o", ms=3, label="water (precip)")
    ax.plot(ep, fx["mass"], color=C_MASS, marker="s", ms=3, label="mass (q, $p_s$)")
    ax.plot(ep, fx["energy"], color=C_ENERGY, marker="^", ms=3, label="energy (T)")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1e2)
    ax.set_xlabel("Training epoch"); ax.set_ylabel(r"Correction magnitude $|r-1|$ (%)")
    ax.legend(loc="center right")
    tag(ax, "(d)")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
