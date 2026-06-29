#!/usr/bin/env python3
"""
plot_drift_comparison.py

Parse WaterFixer diagnostics from training logs and compare to ground-truth
water budget.  Three output modes:

  --mode old        original figure (broken run only, camulator_ext_v2_finetune.o*)
  --mode new        new run only    (camulator_v2_consfix.o*)
  --mode combined   publication figure: full trajectory spanning both runs (default)

Usage examples:
  python plot_drift_comparison.py                          # combined (default)
  python plot_drift_comparison.py --mode old               # original 4-panel
  python plot_drift_comparison.py --mode combined --out my_fig.pdf
"""

import argparse
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
LOGS_DIR  = "/glade/work/wchapman/Roman_Coupling/train_johns"
TRUTH_CSV = "/glade/work/wchapman/Roman_Coupling/train_johns/water_budget_1980.csv"

OLD_PREFIX = "camulator_ext_v2_finetune.o"
NEW_PREFIX = "camulator_v2_consfix.o"

STEPS_PER_EPOCH = 150

# ── palette ───────────────────────────────────────────────────────────────────
C_BROKEN  = "#d62728"   # red   — feedback-loop (broken) epochs
C_FIXED   = "#1f77b4"   # blue  — conservation-penalty fix epochs
C_TRUTH   = "#2ca02c"   # green — CESM ground truth
C_ANNOT   = "#555555"   # gray  — annotations / transition line
C_PSINK   = "#1f77b4"
C_ESRC    = "#ff7f0e"


# ── parsing ───────────────────────────────────────────────────────────────────
PAT_WF = re.compile(
    r"WaterFixer step (\d+) \| dTWC/dt=([-\d.e+]+)\s+E_src=([-\d.e+]+)\s+"
    r"P_sink=([-\d.e+]+)\s+residual=([-\d.e+]+)\s+P_ratio mean=([-\d.e+]+)"
    r".*?drift_pct=([-\d.e+]+)%"
)
PAT_EP = re.compile(r"Beginning epoch (\d+)")


def parse_logs(prefix: str) -> pd.DataFrame:
    """Parse all log files matching *prefix* in LOGS_DIR."""
    log_files = sorted(
        os.path.join(LOGS_DIR, f)
        for f in os.listdir(LOGS_DIR)
        if f.startswith(prefix)
    )
    all_rows = []
    for logf in log_files:
        epochs, rows = set(), []
        with open(logf) as fh:
            for line in fh:
                em = PAT_EP.search(line)
                if em:
                    epochs.add(int(em.group(1)))
                wm = PAT_WF.search(line)
                if wm:
                    rows.append(dict(
                        step=int(wm.group(1)),
                        dTWC_dt=float(wm.group(2)),
                        E_src=float(wm.group(3)),
                        P_sink=float(wm.group(4)),
                        residual=float(wm.group(5)),
                        P_ratio=float(wm.group(6)),
                        drift_pct=float(wm.group(7)),
                    ))
        if not rows or not epochs:
            continue
        df = pd.DataFrame(rows).groupby("step").mean().reset_index()
        df["start_epoch"] = min(epochs)
        df["epoch"]       = df["start_epoch"] + (df["step"] - 1) // STEPS_PER_EPOCH
        df["global_step"] = df["start_epoch"] * STEPS_PER_EPOCH + df["step"]
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()
    return (
        pd.concat(all_rows, ignore_index=True)
        .sort_values("global_step")
        .drop_duplicates("global_step", keep="first")
        .reset_index(drop=True)
    )


def epoch_float(df: pd.DataFrame) -> pd.Series:
    return df["global_step"] / STEPS_PER_EPOCH


# ── figure helpers ────────────────────────────────────────────────────────────
def apply_pub_style():
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.framealpha": 0.85,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth":   1.2,
        "figure.dpi":        150,
    })


# ── OLD figure (original 4-panel) ────────────────────────────────────────────
def plot_old(broken_df: pd.DataFrame, truth_df: pd.DataFrame, out: str):
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Model water budget drift vs ground truth (CESM 1980)\n"
        "v2 fine-tune — WaterFixer active, feedback loop present (epochs 4–15)",
        fontsize=11, fontweight="bold",
    )

    truth_mean   = truth_df["drift_pct"].mean()
    truth_std    = truth_df["drift_pct"].std()
    truth_p_mean = truth_df["P_sink"].mean()
    truth_e_mean = truth_df["E_src"].mean()
    scale        = 1e12

    ep = epoch_float(broken_df)
    cmap = plt.cm.plasma
    ep_min, ep_max = broken_df["epoch"].min(), broken_df["epoch"].max()

    # drift over training
    ax = axes[0, 0]
    sc = ax.scatter(ep, broken_df["drift_pct"], c=broken_df["epoch"],
                    cmap=cmap, s=30, zorder=3, vmin=ep_min, vmax=ep_max)
    ax.plot(ep, broken_df["drift_pct"], color="gray", lw=0.5, alpha=0.5)
    ax.axhline(truth_mean, color=C_TRUTH, lw=1.5, ls="--",
               label=f"truth mean={truth_mean:.3f}%")
    ax.axhline(truth_mean + truth_std, color=C_TRUTH, lw=0.8, ls=":",
               label=f"truth ±1σ={truth_std:.2f}%")
    ax.axhline(truth_mean - truth_std, color=C_TRUTH, lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.6)
    plt.colorbar(sc, ax=ax, label="epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("drift_pct (%)"); ax.set_ylim(-10, 35)
    ax.set_title("Model drift_pct over training"); ax.legend(loc="lower right"); ax.grid(False)

    # P_sink / E_src
    ax = axes[0, 1]
    ax.plot(ep, broken_df["P_sink"] / scale, color=C_PSINK, lw=1.0, label="model P_sink")
    ax.plot(ep, -broken_df["E_src"] / scale, color=C_ESRC,  lw=1.0, label="model |E_src|")
    ax.axhline(truth_p_mean  / scale, color=C_PSINK, lw=1.2, ls="--", label="truth P_sink mean")
    ax.axhline(-truth_e_mean / scale, color=C_ESRC,  lw=1.2, ls="--", label="truth |E_src| mean")
    ax.set_xlabel("Epoch"); ax.set_ylabel("×10¹² kg/s")
    ax.set_title("P_sink vs |E_src|"); ax.legend(); ax.grid(False)

    # histogram
    ax = axes[1, 0]
    bins = np.linspace(-8, 32, 50)
    ax.hist(truth_df["drift_pct"],    bins=bins, density=True, color=C_TRUTH,  alpha=0.55,
            label=f"truth  μ={truth_mean:.2f}%  σ={truth_std:.2f}%")
    ax.hist(broken_df["drift_pct"],   bins=bins, density=True, color=C_BROKEN, alpha=0.55,
            label=f"model  μ={broken_df.drift_pct.mean():.1f}%  σ={broken_df.drift_pct.std():.1f}%")
    ax.axvline(truth_mean,                      color=C_TRUTH,  lw=1.5, ls="--")
    ax.axvline(broken_df["drift_pct"].mean(),   color=C_BROKEN, lw=1.5, ls="--")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("drift_pct (%)"); ax.set_ylabel("density")
    ax.set_title("Distribution: model vs truth"); ax.legend(); ax.grid(False)

    # components
    ax = axes[1, 1]
    ax.plot(ep, broken_df["dTWC_dt"]  / scale, color="purple",    lw=0.9, label="dTWC/dt")
    ax.plot(ep, broken_df["E_src"]    / scale, color=C_ESRC,      lw=0.9, label="E_src")
    ax.plot(ep, broken_df["residual"] / scale, color=C_BROKEN,    lw=1.2, label="residual")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(truth_df["E_src"].mean()    / scale, color=C_ESRC,   ls="--", lw=0.9, alpha=0.6)
    ax.axhline(truth_df["residual"].mean() / scale, color=C_BROKEN, ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("×10¹² kg/s")
    ax.set_title("Budget components (model)"); ax.legend(ncol=2); ax.grid(False)

    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved → {out}")


# ── COMBINED publication figure ───────────────────────────────────────────────
def plot_combined(broken_df: pd.DataFrame, fixed_df: pd.DataFrame,
                  truth_df: pd.DataFrame, out: str):
    apply_pub_style()

    truth_mean   = truth_df["drift_pct"].mean()
    truth_std    = truth_df["drift_pct"].std()
    truth_p_mean = truth_df["P_sink"].mean()
    truth_e_mean = truth_df["E_src"].mean()
    scale = 1e12

    # transition epoch
    trans_epoch = fixed_df["epoch"].min() if not fixed_df.empty else None

    fig = plt.figure(figsize=(9, 11))
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.35,
                           left=0.10, right=0.96, top=0.93, bottom=0.07)
    ax_drift  = fig.add_subplot(gs[0, :])      # full width — drift_pct
    ax_psink  = fig.add_subplot(gs[1, 0])      # P_sink trajectory
    ax_zoom   = fig.add_subplot(gs[1, 1])      # zoom around transition
    ax_hist   = fig.add_subplot(gs[2, 0])      # histogram
    ax_comps  = fig.add_subplot(gs[2, 1])      # budget components

    fig.suptitle(
        "Global water budget drift: feedback-loop failure vs conservation-penalty fix",
        fontsize=11, fontweight="bold", y=0.97,
    )

    def _transition_vline(ax, label=True):
        if trans_epoch is None:
            return
        ax.axvline(trans_epoch, color=C_ANNOT, lw=1.0, ls="--", zorder=2)
        if label:
            ax.text(trans_epoch + 0.3, ax.get_ylim()[1] * 0.97,
                    "conservation\nfix applied", fontsize=7, color=C_ANNOT,
                    va="top", ha="left", style="italic")

    # ── Panel 1: drift_pct full trajectory ───────────────────────────────────
    ax = ax_drift
    ep_b = epoch_float(broken_df)
    ep_f = epoch_float(fixed_df)

    # shaded background regions
    if trans_epoch:
        ax.axvspan(broken_df["epoch"].min(), trans_epoch,
                   color=C_BROKEN, alpha=0.07, zorder=0, label="_nolegend_")
        ax.axvspan(trans_epoch, fixed_df["epoch"].max() + 1,
                   color=C_FIXED,  alpha=0.07, zorder=0, label="_nolegend_")

    # ground truth band
    ax.axhspan(truth_mean - truth_std, truth_mean + truth_std,
               color=C_TRUTH, alpha=0.18, zorder=1)
    ax.axhline(truth_mean, color=C_TRUTH, lw=1.2, ls="-", zorder=2)

    # model lines
    ax.plot(ep_b, broken_df["drift_pct"],
            color=C_BROKEN, lw=1.4, zorder=3, label="Feedback loop (epochs 4–15)")
    ax.scatter(ep_b, broken_df["drift_pct"],
               color=C_BROKEN, s=18, zorder=4)
    if not fixed_df.empty:
        ax.plot(ep_f, fixed_df["drift_pct"],
                color=C_FIXED, lw=1.4, zorder=3, label="Conservation fix (epoch 16+)")
        ax.scatter(ep_f, fixed_df["drift_pct"],
                   color=C_FIXED, s=18, zorder=4)

    ax.axhline(0, color="k", lw=0.6, zorder=2)
    _transition_vline(ax)

    # legend entries for truth band
    truth_patch = mpatches.Patch(color=C_TRUTH, alpha=0.5,
                                 label=f"CESM truth  μ={truth_mean:.3f}%  ±1σ={truth_std:.2f}%")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [truth_patch], labels + [truth_patch.get_label()],
              loc="upper left", framealpha=0.9)

    ax.set_xlabel("Epoch"); ax.set_ylabel("drift_pct  (%)")
    ax.set_title("(a)  WaterFixer drift_pct across training epochs", loc="left", fontweight="bold")
    ax.set_ylim(-8, 32)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))

    # ── Panel 2: P_sink trajectory ────────────────────────────────────────────
    ax = ax_psink
    ax.plot(ep_b, broken_df["P_sink"] / scale, color=C_BROKEN, lw=1.3,
            label="model (broken)")
    if not fixed_df.empty:
        ax.plot(ep_f, fixed_df["P_sink"] / scale, color=C_FIXED,  lw=1.3,
                label="model (fixed)")
    ax.axhline(truth_p_mean / scale, color=C_TRUTH, lw=1.2, ls="--",
               label="CESM truth mean")
    ax.axhline(-truth_e_mean / scale, color=C_TRUTH, lw=1.2, ls=":",
               label="truth |E_src|", alpha=0.7)
    _transition_vline(ax, label=False)
    ax.set_xlabel("Epoch"); ax.set_ylabel("×10¹² kg/s")
    ax.set_title("(b)  P_sink & truth reference", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5)

    # ── Panel 3: zoom around transition ───────────────────────────────────────
    ax = ax_zoom
    zoom_lo = max(0, (trans_epoch or 16) - 3)
    zoom_hi = (trans_epoch or 16) + 5

    ax.axhspan(truth_mean - truth_std, truth_mean + truth_std,
               color=C_TRUTH, alpha=0.2)
    ax.axhline(truth_mean, color=C_TRUTH, lw=1.2)

    mask_b = (ep_b >= zoom_lo) & (ep_b <= zoom_hi)
    mask_f = (ep_f >= zoom_lo) & (ep_f <= zoom_hi)

    if mask_b.any():
        ax.plot(ep_b[mask_b], broken_df.loc[mask_b, "drift_pct"],
                color=C_BROKEN, lw=1.5, marker="o", ms=4, zorder=3)
    if not fixed_df.empty and mask_f.any():
        ax.plot(ep_f[mask_f], fixed_df.loc[mask_f, "drift_pct"],
                color=C_FIXED,  lw=1.5, marker="o", ms=4, zorder=3)

    _transition_vline(ax, label=True)
    ax.set_xlim(zoom_lo, zoom_hi)
    ax.set_xlabel("Epoch"); ax.set_ylabel("drift_pct  (%)")
    ax.set_title("(c)  Zoom: transition at epoch 16", loc="left", fontweight="bold")
    ax.axhline(0, color="k", lw=0.6)

    # ── Panel 4: histogram ────────────────────────────────────────────────────
    ax = ax_hist
    bins = np.linspace(-8, 32, 45)
    ax.hist(truth_df["drift_pct"],  bins=bins, density=True, color=C_TRUTH,
            alpha=0.55, label=f"CESM truth  μ={truth_mean:.2f}%")
    ax.hist(broken_df["drift_pct"], bins=bins, density=True, color=C_BROKEN,
            alpha=0.55, label=f"Broken  μ={broken_df.drift_pct.mean():.1f}%")
    if not fixed_df.empty:
        ax.hist(fixed_df["drift_pct"], bins=bins, density=True, color=C_FIXED,
                alpha=0.55, label=f"Fixed   μ={fixed_df.drift_pct.mean():.1f}%")
    ax.axvline(truth_mean,                     color=C_TRUTH,  lw=1.4, ls="--")
    ax.axvline(broken_df["drift_pct"].mean(),  color=C_BROKEN, lw=1.4, ls="--")
    if not fixed_df.empty:
        ax.axvline(fixed_df["drift_pct"].mean(), color=C_FIXED, lw=1.4, ls="--")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("drift_pct  (%)"); ax.set_ylabel("Density")
    ax.set_title("(d)  Distribution comparison", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5)

    # ── Panel 5: budget components ────────────────────────────────────────────
    ax = ax_comps
    for df_, col_, lbl_, ls_ in [
        (broken_df, "residual", "Residual (broken)", "-"),
        (fixed_df,  "residual", "Residual (fixed)",  "-"),
        (broken_df, "E_src",    "E_src (broken)",    "--"),
        (fixed_df,  "E_src",    "E_src (fixed)",     "--"),
    ]:
        if df_.empty:
            continue
        ep_ = epoch_float(df_)
        c_  = C_BROKEN if "broken" in lbl_ else C_FIXED
        ax.plot(ep_, df_[col_] / scale, color=c_, lw=1.1, ls=ls_,
                label=lbl_, alpha=0.85)

    ax.axhline(truth_df["residual"].mean() / scale, color=C_TRUTH,  lw=1.0,
               ls="-",  alpha=0.7, label="truth residual ≈ 0")
    ax.axhline(truth_df["E_src"].mean()    / scale, color=C_TRUTH,  lw=1.0,
               ls="--", alpha=0.7, label="truth E_src mean")
    ax.axhline(0, color="k", lw=0.6)
    _transition_vline(ax, label=False)
    ax.set_xlabel("Epoch"); ax.set_ylabel("×10¹² kg/s")
    ax.set_title("(e)  Budget components (residual & E_src)", loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2)

    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved → {out}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\nBroken run  drift_pct: μ={broken_df.drift_pct.mean():.1f}%  "
          f"σ={broken_df.drift_pct.std():.1f}%")
    if not fixed_df.empty:
        print(f"Fixed  run  drift_pct: μ={fixed_df.drift_pct.mean():.2f}%  "
              f"σ={fixed_df.drift_pct.std():.2f}%")
    print(f"CESM truth  drift_pct: μ={truth_mean:.3f}%  σ={truth_std:.2f}%")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["old", "new", "combined"], default="combined",
                        help="Figure mode (default: combined)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: drift_comparison_<mode>.png)")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(LOGS_DIR, f"drift_comparison_{args.mode}.png")

    truth_df  = pd.read_csv(TRUTH_CSV)
    broken_df = parse_logs(OLD_PREFIX)
    fixed_df  = parse_logs(NEW_PREFIX)

    print(f"Broken epochs: {sorted(broken_df.epoch.unique()) if not broken_df.empty else 'none'}")
    print(f"Fixed  epochs: {sorted(fixed_df.epoch.unique())  if not fixed_df.empty  else 'none'}")

    if args.mode == "old":
        plot_old(broken_df, truth_df, args.out)
    elif args.mode == "new":
        plot_old(fixed_df, truth_df, args.out)   # reuse 4-panel layout for new-only
    else:
        plot_combined(broken_df, fixed_df, truth_df, args.out)


if __name__ == "__main__":
    main()
