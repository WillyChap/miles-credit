#!/usr/bin/env python3
"""Figure 1 for the Lessons Learned paper: the phenomenon, the controlled cause, and the
physical consequence.

(a) Production fine-tune (the phenomenon at scale): drift ramps under the feedback, the
    penalty is enabled at epoch 16, and drift then holds near zero for tens of epochs.
(b) Controlled screen (the cause): two runs from the identical cp00092_extended checkpoint
    differing only in conservation_loss_weight (0.0 vs 0.1); the feedback runs away while
    the penalty holds drift at zero. This isolates the loss coupling.
(c) The physical field the corrector hides: raw (pre-correction) precip sink relative to
    truth in the screen runs.

Reads cached CSVs in experiments/figure_data/: production drift (drift_broken/drift_fixed),
the controlled screen (screen_broken/screen_fix, from build_screen_data.py), and the truth
budget. Re-run after the screen jobs finish to refresh (b) and (c). Writes
experiments/figures/fig1.{pdf,png}.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "figure_data")
OUT = os.path.join(HERE, "fig1")

STEPS_PER_EPOCH = 150
C_BROKEN = "#D55E00"
C_FIXED = "#0072B2"
C_TRUTH = "#009E73"


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def tag(ax, t):
    ax.text(-0.02, 1.04, t, transform=ax.transAxes, fontweight="bold", fontsize=10,
            va="bottom", ha="right")


def main():
    style()
    pb = pd.read_csv(os.path.join(DATA, "drift_broken.csv"))
    pf = pd.read_csv(os.path.join(DATA, "drift_fixed.csv"))
    truth = pd.read_csv(os.path.join(DATA, "truth_water_budget_1980.csv"))
    sb = pd.read_csv(os.path.join(DATA, "screen_broken.csv"))
    sf = pd.read_csv(os.path.join(DATA, "screen_fix.csv"))
    # drop the cold-start transient (first WaterFixer call after loading the checkpoint)
    sb = sb[sb["step"] > 1]
    sf = sf[sf["step"] > 1]

    tm, ts = truth["drift_pct"].mean(), truth["drift_pct"].std()
    p_truth = truth["P_sink"].mean()

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.46, wspace=0.30,
                          left=0.09, right=0.97, top=0.95, bottom=0.09)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    def truthband(ax):
        ax.axhspan(tm - ts, tm + ts, color=C_TRUTH, alpha=0.15, zorder=0)
        ax.axhline(tm, color=C_TRUTH, lw=1.0, zorder=1)

    # (a) production fine-tune
    ax = ax_a
    truthband(ax)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.plot(pb["global_step"] / STEPS_PER_EPOCH, pb["drift_pct"], color=C_BROKEN, lw=1.3,
            label="feedback (loss on corrected)")
    ax.plot(pf["global_step"] / STEPS_PER_EPOCH, pf["drift_pct"], color=C_FIXED, lw=1.3,
            label="decoupled loss + penalty")
    trans = pf["epoch"].min()
    ax.axvline(trans, color="0.35", lw=0.9, ls="--")
    ax.annotate("penalty enabled", xy=(trans, 6), xytext=(trans + 14, 15), fontsize=7.5,
                color="0.35", arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8))
    tp = mpatches.Patch(color=C_TRUTH, alpha=0.5, label=f"CAM6 truth ({tm:.2f}% ± {ts:.2f}%)")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [tp], lab + [tp.get_label()], loc="upper right")
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Water-budget drift (%)")
    ax.set_ylim(-8, 30)
    tag(ax, "(a)"); ax.set_title("Production fine-tune", fontsize=8.5, loc="left", color="0.3")

    # (b) controlled screen
    ax = ax_b
    truthband(ax)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.plot(sb["epoch"], sb["drift_pct"], color=C_BROKEN, marker="o", ms=2.5, label="weight 0.0")
    ax.plot(sf["epoch"], sf["drift_pct"], color=C_FIXED, marker="o", ms=2.5, label="weight 0.1")
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Water-budget drift (%)")
    ax.legend(loc="center right", title="same checkpoint", title_fontsize=7)
    tag(ax, "(b)"); ax.set_title("Controlled comparison", fontsize=8.5, loc="left", color="0.3")

    # (c) screen raw precip sink / truth
    ax = ax_c
    ax.axhline(1.0, color=C_TRUTH, lw=1.0, ls="--", label="CAM6 truth")
    ax.plot(sb["epoch"], sb["P_sink"] / p_truth, color=C_BROKEN, marker="o", ms=2.5, label="weight 0.0")
    ax.plot(sf["epoch"], sf["P_sink"] / p_truth, color=C_FIXED, marker="o", ms=2.5, label="weight 0.1")
    ax.set_xlabel("Training epoch"); ax.set_ylabel("Raw precip sink / truth")
    ax.legend(loc="lower left")
    tag(ax, "(c)"); ax.set_title("Hidden field", fontsize=8.5, loc="left", color="0.3")

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png  (screen pts: broken={len(sb)}, fix={len(sf)})")


if __name__ == "__main__":
    main()
