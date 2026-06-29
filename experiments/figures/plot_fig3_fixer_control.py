#!/usr/bin/env python3
"""Figure 3: only the water correction runs away (a within-run control).

In the feedback run all three global correctors are active and all enter the loss, yet the
magnitude of the correction the model demands grows only for water. Mass and energy stay
near machine-zero throughout. The difference is the field each correction is carried by:
water's lands on precipitation (a single diagnostic channel whose global amplitude the rest
of the loss leaves nearly free), while mass and energy land on surface pressure, humidity,
and temperature (prognostic fields supervised at every level). The runaway is specific to
the corrector whose adjustment falls on a weakly constrained field.

Reads experiments/figure_data/fixer_control_broken.csv (parsed from the broken-run logs by
build_fixer_control_data.py) and the cached truth budget. Writes
experiments/figures/fig3_fixer_control.{pdf,png}. Title-free, colorblind-safe.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "figure_data")
OUT = os.path.join(HERE, "fig3_fixer_control")

STEPS_PER_EPOCH = 150
C_WATER = "#D55E00"   # vermillion
C_MASS = "#0072B2"    # blue
C_ENERGY = "#CC79A7"  # reddish purple


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def main():
    style()
    d = pd.read_csv(os.path.join(DATA, "fixer_control_broken.csv"))
    truth = pd.read_csv(os.path.join(DATA, "truth_water_budget_1980.csv"))
    nat = truth["drift_pct"].std()  # CAM6 natural per-step budget spread
    ep = d["global_step"] / STEPS_PER_EPOCH

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.15)

    ax.axhspan(0, nat, color="0.85", zorder=0)
    ax.axhline(nat, color="0.6", lw=0.8, ls=":", zorder=1)
    ax.text(d["global_step"].max() / STEPS_PER_EPOCH, nat * 1.15,
            "CAM6 natural per-step spread", ha="right", va="bottom", fontsize=6.8, color="0.45")

    ax.plot(ep, d["water"], color=C_WATER, marker="o", ms=3, label="water  (carried by precip)")
    ax.plot(ep, d["mass"], color=C_MASS, marker="s", ms=3, label="mass  (q, surface pressure)")
    ax.plot(ep, d["energy"], color=C_ENERGY, marker="^", ms=3, label="energy  (temperature)")

    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Correction magnitude $|r-1|$  (%)")
    ax.legend(loc="center right")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
