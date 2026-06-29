#!/usr/bin/env python3
"""Plot water_budget_1980.csv — ground-truth water budget diagnostics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV = "/glade/work/wchapman/Roman_Coupling/train_johns/water_budget_1980.csv"
OUT = "/glade/work/wchapman/Roman_Coupling/train_johns/water_budget_1980.png"

df = pd.read_csv(CSV, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

# Scale fluxes to kg/m²/s (divide by Earth surface area ≈ 5.1e14 m²)
# But units are already area-weighted sums from weighted_sum — keep as-is,
# just convert to more readable units by dividing by 1e12 (Tg/s equivalent)
scale = 1e12   # → ×10¹² kg/s  (Tg/s)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("CESM Training Data — Ground-Truth Water Budget (1980)\n"
             "Same physics as GlobalWaterFixer", fontsize=13, fontweight="bold")

t = df["time"]

# ── Panel 1: P_sink and |E_src| ───────────────────────────────────────────
ax = axes[0]
ax.plot(t, df["P_sink"] / scale, color="steelblue", lw=0.8, label="P_sink (precip)")
ax.plot(t, -df["E_src"] / scale, color="darkorange", lw=0.8, label="|E_src| (evap, negated)")
ax.set_ylabel("×10¹² kg/s", fontsize=9)
ax.set_title("Precipitation sink vs Evaporation source", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Panel 2: dTWC/dt ─────────────────────────────────────────────────────
ax = axes[1]
ax.plot(t, df["dTWC_dt"] / scale, color="purple", lw=0.7)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("×10¹² kg/s", fontsize=9)
ax.set_title("dTWC/dt  (total column water tendency)", fontsize=10)
ax.grid(True, alpha=0.3)

# ── Panel 3: residual ────────────────────────────────────────────────────
ax = axes[2]
ax.plot(t, df["residual"] / scale, color="firebrick", lw=0.7)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.fill_between(t, 0, df["residual"] / scale,
                where=df["residual"] > 0, color="firebrick", alpha=0.25, label="positive (dry bias)")
ax.fill_between(t, 0, df["residual"] / scale,
                where=df["residual"] < 0, color="royalblue", alpha=0.25, label="negative (wet bias)")
ax.set_ylabel("×10¹² kg/s", fontsize=9)
ax.set_title("residual = −dTWC/dt − E_src − P_sink", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 4: drift_pct with stats ────────────────────────────────────────
ax = axes[3]
ax.plot(t, df["drift_pct"], color="darkgreen", lw=0.8, alpha=0.8)
ax.axhline(0,              color="k",      lw=0.8, ls="--")
ax.axhline(df["drift_pct"].mean(),  color="red",    lw=1.0, ls=":",
           label=f"mean = {df['drift_pct'].mean():.3f}%")
ax.axhline(df["drift_pct"].mean() + df["drift_pct"].std(), color="salmon",
           lw=0.8, ls=":", label=f"±1σ = {df['drift_pct'].std():.3f}%")
ax.axhline(df["drift_pct"].mean() - df["drift_pct"].std(), color="salmon", lw=0.8, ls=":")
ax.fill_between(t, -5.5, df["drift_pct"],
                where=df["drift_pct"] > 0, color="firebrick", alpha=0.15)
ax.fill_between(t, df["drift_pct"], 5.5,
                where=df["drift_pct"] < 0, color="royalblue", alpha=0.15)
ax.set_ylabel("drift_pct (%)", fontsize=9)
ax.set_title("drift_pct = 100 × residual / P_sink  (WaterFixer correction needed)", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# x-axis formatting
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
fig.autofmt_xdate(rotation=0, ha="center")

# Annotation box
stats_text = (
    f"drift_pct:  mean={df['drift_pct'].mean():.3f}%  "
    f"std={df['drift_pct'].std():.3f}%  "
    f"min={df['drift_pct'].min():.2f}%  "
    f"max={df['drift_pct'].max():.2f}%"
)
fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9,
         bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
