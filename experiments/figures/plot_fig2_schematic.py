#!/usr/bin/env python3
"""Figure 2 — constraint-placement schematic for the Lessons Learned paper.

Three stacked panels showing the training computational graph for:
  (a) no constraint
  (b) hard corrector inside the loss (scale degeneracy): the loss sees only the corrected
      output, so the raw-amplitude direction is annihilated by the corrector (J_C(P)P = 0,
      gate highlighted) and no amplitude signal reaches the network
  (c) decoupled loss on the pre-correction prediction + additive imbalance penalty

All configurations with the corrector active deliver a water budget closed to machine
precision (shared green terminal box); they differ only in what reaches the supervised
gradient. The drift gauge is read off the correction magnitude |r-1|, not the delivered
field. Pure schematic, no data. Writes fig2_constraint_placement.{pdf,png}.

Run from the repo root:  python experiments/figures/plot_fig2_schematic.py
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig2_constraint_placement")

# palette consistent with Figure 1 (Okabe-Ito)
C_BROKEN = "#D55E00"   # canceled raw-amplitude gradient path
C_FIXED = "#0072B2"    # clean gradient path
C_TRUTH = "#009E73"    # "closed" delivered field
FC_NET = "#dce7f2"
FC_RAW = "#f2f2f2"
FC_COR = "#fbe3d0"
FC_CLOSED = "#d8efe6"
FC_OPEN = "#ededed"
FC_LOSS = "#ece8f3"
FC_PEN = "#eef2fb"


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
    })


def box(ax, cx, cy, w, h, text, fc, ec="0.35", tc="black", fs=7.5, z=3):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.10",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1)
    return dict(c=(cx, cy), L=(cx - w / 2, cy), R=(cx + w / 2, cy),
                T=(cx, cy + h / 2), B=(cx, cy - h / 2))


def arrow(ax, p0, p1, color="0.25", ls="-", lw=1.3, rad=0.0, z=2):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                                shrinkA=1.5, shrinkB=1.5))


def label(ax, x, y, text, color="0.25", fs=6.8, style="italic", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, fontstyle=style, zorder=6)


def gauge(ax):
    """Drift-gauge annotation in the empty lower-right: read off the correction magnitude."""
    ax.annotate("drift $=|r-1|$\n(correction magnitude)",
                xy=(X_COR, YF - 0.50), xytext=(X_DEL - 0.2, YF - 1.25),
                ha="center", va="center", fontsize=6.8, color="0.3", fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8), zorder=6)


def grad_return(ax, x_from, color):
    """Dashed gradient bus: down from the loss, left along the bottom lane, up into the network."""
    ax.plot([x_from, x_from], [YL - 0.33, YG], color=color, ls=(0, (4, 2)), lw=1.4, zorder=2)
    ax.plot([x_from, X_NET], [YG, YG], color=color, ls=(0, (4, 2)), lw=1.4, zorder=2)
    arrow(ax, (X_NET, YG), (X_NET, YF - 0.45), color=color, ls=(0, (4, 2)), lw=1.4)


def feedback(ax):
    """Autoregressive loop: the delivered next-state prediction becomes the next input.
    Routed in a dedicated bottom lane so it never crosses the gradient bus."""
    ax.annotate("", xy=(X_IN, YF - 0.40), xytext=(X_DEL, YF - 0.52), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color="0.72", lw=1.1,
                                connectionstyle="arc3,rad=-0.55", shrinkA=3, shrinkB=3))
    ax.text((X_IN + X_DEL) / 2, -0.70,
            "autoregressive rollout: delivered state is the next input  $x_{t+1}\\!\\rightarrow\\!x_t$",
            ha="center", va="center", fontsize=6.6, color="0.6", fontstyle="italic", zorder=1)


def setup(ax, tag, descr):
    ax.set_xlim(0, 12.6)
    ax.set_ylim(-0.95, 4.45)
    ax.axis("off")
    ax.text(0.02, 4.30, f"({tag})", fontweight="bold", fontsize=10, ha="left", va="top")
    ax.text(0.66, 4.30, descr, fontstyle="italic", fontsize=8.5, ha="left", va="top", color="0.3")


# fixed lanes / x positions shared across panels
YF = 2.35     # forward lane
YL = 3.55     # loss / penalty row
YG = 0.85     # gradient return lane
X_IN, X_NET, X_RAW, X_COR, X_DEL = 0.9, 2.8, 4.7, 6.8, 10.3


def _common(ax, with_corrector, del_text, del_fc, del_ec):
    inp = box(ax, X_IN, YF, 1.0, 0.7, "input\n$x_t$", FC_RAW)
    net = box(ax, X_NET, YF, 1.7, 0.9, "Network", FC_NET)
    rawtxt = r"$\hat{x}_{t+1}$" if not with_corrector else r"$\hat{x}^{\mathrm{pre}}_{t+1}$"
    raw = box(ax, X_RAW, YF, 1.2, 0.7, rawtxt, FC_RAW)
    cor = box(ax, X_COR, YF, 1.9, 0.9, "Water fixer $C$\n$(\\times\\, r)$", FC_COR) if with_corrector else None
    del_ = box(ax, X_DEL, YF, 3.0, 0.95, del_text, del_fc, ec=del_ec)
    arrow(ax, inp["R"], net["L"]); arrow(ax, net["R"], raw["L"])
    return inp, net, raw, cor, del_


def panel_a(ax):
    setup(ax, "a", "No constraint")
    _, net, raw, _, del_ = _common(ax, False, "Delivered output  ($=\\hat{x}_{t+1}$)\nbudget NOT closed", FC_OPEN, "0.5")
    loss = box(ax, X_RAW, YL, 2.4, 0.66, "Supervised loss vs $x_{t+1}$", FC_LOSS)
    arrow(ax, raw["R"], del_["L"])
    arrow(ax, raw["T"], loss["B"])
    grad_return(ax, loss["c"][0], "0.45")
    label(ax, (X_NET + X_RAW) / 2, YG - 0.30, "gradient", color="0.45")
    feedback(ax)


def panel_b(ax):
    setup(ax, "b", "Hard corrector inside the loss  —  scale degeneracy")
    _, net, raw, cor, del_ = _common(ax, True, "Delivered output\nbudget closed to\nmachine precision", FC_CLOSED, C_TRUTH)
    loss = box(ax, 8.6, YL, 2.7, 0.66, "Supervised loss vs $x_{t+1}$\n(sees $C(\\hat{x})$)", FC_LOSS)
    arrow(ax, raw["R"], cor["L"]); arrow(ax, cor["R"], del_["L"])
    arrow(ax, cor["T"], loss["B"])
    # Backward gradient: from the loss along the bottom lane toward the network, but the
    # raw-amplitude component is annihilated AT the corrector (J_C(P)P = 0). Solid up to the
    # corrector gate, faded beyond it: no amplitude signal reaches the network.
    ax.plot([loss["c"][0], loss["c"][0]], [YL - 0.33, YG], color=C_BROKEN, ls=(0, (4, 2)), lw=1.7, zorder=2)
    ax.plot([loss["c"][0], X_COR], [YG, YG], color=C_BROKEN, ls=(0, (4, 2)), lw=1.7, zorder=2)
    ax.plot([X_COR, X_NET], [YG, YG], color=C_BROKEN, ls=(0, (4, 2)), lw=1.7, alpha=0.25, zorder=2)
    ax.plot([X_NET, X_NET], [YG, YF - 0.55], color=C_BROKEN, ls=(0, (4, 2)), lw=1.7, alpha=0.25, zorder=2)
    # cancellation gate at the corrector (circle-slash)
    ax.scatter([X_COR], [YG], s=170, facecolors="white", edgecolors=C_BROKEN, linewidths=1.7, zorder=5)
    ax.plot([X_COR - 0.14, X_COR + 0.14], [YG - 0.14, YG + 0.14], color=C_BROKEN, lw=1.7, zorder=6)
    label(ax, (X_COR + X_NET) / 2 - 0.1, YG - 0.30,
          "raw-amplitude direction canceled:  $J_C(P)\\,P = 0$",
          color=C_BROKEN, fs=7.2)
    gauge(ax)
    feedback(ax)


def panel_c(ax):
    setup(ax, "c", "Decoupled loss + imbalance penalty")
    _, net, raw, cor, del_ = _common(ax, True, "Delivered output\nbudget closed to\nmachine precision", FC_CLOSED, C_TRUTH)
    loss = box(ax, 5.5, YL, 2.5, 0.66, "Supervised loss vs $x_{t+1}$\n(sees $\\hat{x}^{\\mathrm{pre}}_{t+1}$)", FC_LOSS)
    pen = box(ax, 8.3, YL, 2.2, 0.66, "Imbalance penalty\n$(r-1)^2$", FC_PEN, ec=C_FIXED)
    arrow(ax, raw["R"], cor["L"])
    arrow(ax, cor["R"], del_["L"], color="0.55")
    label(ax, (cor["R"][0] + del_["L"][0]) / 2, YF + 0.40, "forward only\n(detached)", color="0.55", fs=6.6)
    arrow(ax, raw["T"], loss["B"])
    arrow(ax, cor["T"], pen["B"])
    arrow(ax, pen["L"], (loss["R"][0], pen["c"][1]), color=C_FIXED, lw=1.1)  # penalty added to loss
    label(ax, (loss["R"][0] + pen["L"][0]) / 2, YL + 0.34, "+", color=C_FIXED, fs=12, style="normal")
    grad_return(ax, loss["c"][0], C_FIXED)
    label(ax, (X_NET + X_RAW) / 2, YG - 0.30, "gradient to network (not through corrector)",
          color=C_FIXED, fs=7.0)
    gauge(ax)
    feedback(ax)


def main():
    style()
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.16)
    panel_a(axes[0]); panel_b(axes[1]); panel_c(axes[2])
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
