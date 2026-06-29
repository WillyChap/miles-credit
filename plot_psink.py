"""
Extract P_sink / drift_pct from WaterFixer log lines across chained PBS jobs,
detect oscillations, and plot.

Usage
-----
# Auto-detect all camulator_extended_finetune.o* logs in script directory:
    python plot_psink.py

# Explicit files (sorted by job number automatically):
    python plot_psink.py camulator_extended_finetune.o3140017 camulator_extended_finetune.o3140021

# Full paths also accepted.
"""
import re, sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import uniform_filter1d
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Resolve input files ───────────────────────────────────────────────────────
if len(sys.argv) > 1:
    raw_paths = sys.argv[1:]
    files = []
    for p in raw_paths:
        if os.path.isabs(p):
            files.append(p)
        elif os.path.exists(p):
            files.append(os.path.abspath(p))
        else:
            files.append(os.path.join(SCRIPT_DIR, p))
else:
    files = glob.glob(os.path.join(SCRIPT_DIR, "camulator_extended_finetune.o*"))

def job_id(path):
    m = re.search(r"\.o(\d+)$", path)
    return int(m.group(1)) if m else 0

files = sorted(files, key=job_id)
print(f"Processing {len(files)} log file(s):")
for f in files:
    print(f"  {f}")

# ── Parse + deduplicate (4-GPU ranks write same step) ────────────────────────
re_epoch = re.compile(r"Beginning epoch\s+(\d+)")
re_water = re.compile(r"WaterFixer step\s+(\d+)\s+\|.*?P_sink=([-\d.eE+]+).*?drift_pct=([-\d.eE+]+)%")

STEPS_PER_EPOCH = 6101
current_epoch = 0
raw      = defaultdict(list)   # gx -> [(p_sink, drift, epoch, job_label)]
job_gx_first = {}

for path in files:
    jlabel = os.path.basename(path)
    with open(path) as fh:
        for line in fh:
            m = re_epoch.search(line)
            if m:
                current_epoch = int(m.group(1))
            m = re_water.search(line)
            if m:
                step = int(m.group(1))
                gx   = current_epoch * STEPS_PER_EPOCH + step
                raw[gx].append((float(m.group(2)), float(m.group(3)), current_epoch, jlabel))
                job_gx_first.setdefault(jlabel, gx)

records = []
epoch_first_gx = {}
for gx, vals in sorted(raw.items()):
    ps_mean = np.mean([v[0] for v in vals])
    dr_mean = np.mean([v[1] for v in vals])
    ep      = vals[0][2]
    jlabel  = vals[0][3]
    records.append((gx, ps_mean, dr_mean, ep, jlabel))
    epoch_first_gx.setdefault(ep, gx)

if not records:
    print("No WaterFixer lines found."); sys.exit(0)

gx     = np.array([r[0] for r in records])
ps     = np.array([r[1] for r in records])
dr     = np.array([r[2] for r in records])
ep     = np.array([r[3] for r in records])
jlabs  = [r[4] for r in records]
unique_jobs = list(dict.fromkeys(jlabs))
unique_ep   = np.unique(ep)

dt_median = float(np.median(np.diff(gx)))   # ~50 steps
print(f"\nDedup: {len(records)} points, dt≈{dt_median:.0f} steps/sample")
print(f"Epoch range : {ep.min()} – {ep.max()}")
print(f"P_sink      : {ps.min():.3e} – {ps.max():.3e}")
print(f"drift_pct   : {dr.min():.2f}% – {dr.max():.2f}%")

# ── Oscillation detection ─────────────────────────────────────────────────────
# 1) Per-epoch start values (epoch-alternation check)
ep_start = {e: dr[ep == e][0] for e in unique_ep}
ep_parity = {e: "even" if e % 2 == 0 else "odd" for e in unique_ep}
even_starts = [ep_start[e] for e in unique_ep if e % 2 == 0]
odd_starts  = [ep_start[e] for e in unique_ep if e % 2 == 1]
print(f"\nEven epoch start drift: {np.mean(even_starts):.1f}% ± {np.std(even_starts):.1f}%")
print(f"Odd  epoch start drift: {np.mean(odd_starts):.1f}%  ± {np.std(odd_starts):.1f}%")

# 2) Autocorrelation on per-epoch detrended drift
dr_dt = dr.copy()
for e in unique_ep:
    mask = ep == e
    x = np.arange(mask.sum())
    poly = np.polyfit(x, dr[mask], 2)
    dr_dt[mask] -= np.polyval(poly, x)

n = len(dr_dt)
ac_full = np.real(np.fft.ifft(np.abs(np.fft.fft(dr_dt))**2))
ac_full /= ac_full[0]
ac_rolled = np.roll(ac_full, n // 2)
lag_idx   = np.arange(n) - n // 2
lag_steps = lag_idx * dt_median

pos_mask = (lag_steps > dt_median * 1.5) & (lag_steps < lag_steps.max() / 2)
ac_pos  = ac_rolled[pos_mask]
lag_pos = lag_steps[pos_mask]
peaks, _ = signal.find_peaks(ac_pos, height=0.1, distance=3)
print("\nAutocorrelation peaks (detrended drift):")
for pk in peaks[:6]:
    print(f"  lag={lag_pos[pk]:.0f} steps  r={ac_pos[pk]:.3f}")

# 3) Dominant period from FFT on P_sink (globally detrended)
ps_dt = ps - np.polyval(np.polyfit(np.arange(len(ps)), ps, 1), np.arange(len(ps)))
fft_ps = np.abs(np.fft.rfft(ps_dt))
freqs  = np.fft.rfftfreq(len(ps_dt), d=dt_median)
top5   = np.argsort(fft_ps)[-6:][::-1]
print("\nDominant P_sink periods (globally detrended):")
for i in top5:
    if freqs[i] > 0:
        print(f"  period={1/freqs[i]:.0f} steps  power={fft_ps[i]:.3e}")

SMOOTH = max(6, len(records) // 40)
cmap   = plt.colormaps["tab20"]
jcolor = {j: cmap(i / max(len(unique_jobs)-1, 1)) for i, j in enumerate(unique_jobs)}
ep_cmap = plt.colormaps["RdYlGn_r"]

# ── Figure: 2×2 grid ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.suptitle("WaterFixer diagnostics — CAMulator extended fine-tune (chained jobs)", fontsize=13)
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.3)

ax_ps  = fig.add_subplot(gs[0, :])    # full width: P_sink
ax_dr  = fig.add_subplot(gs[1, :])    # full width: drift_pct
ax_fold= fig.add_subplot(gs[2, 0])    # epoch-folded drift
ax_ac  = fig.add_subplot(gs[2, 1])    # autocorrelation

def vlines_epochs(ax):
    ylim = ax.get_ylim()
    for e, gx_s in sorted(epoch_first_gx.items()):
        color = "#e06060" if e % 2 == 0 else "#6090e0"
        ax.axvline(gx_s, color=color, lw=0.8, ls="--", alpha=0.6)
        ax.text(gx_s + 30, ylim[1], f"e{e}", fontsize=6,
                color=color, va="top", clip_on=True)

def vlines_jobs(ax):
    for jl, gx_s in job_gx_first.items():
        ax.axvline(gx_s, color="black", lw=1.2, ls=":", alpha=0.7)

# -- P_sink panel --------------------------------------------------------------
for jlabel in unique_jobs:
    mask = np.array([j == jlabel for j in jlabs])
    ax_ps.scatter(gx[mask], ps[mask]/1e10, s=5, alpha=0.35, color=jcolor[jlabel], label=jlabel)
if len(ps) > SMOOTH:
    ax_ps.plot(gx, uniform_filter1d(ps, SMOOTH)/1e10, color="navy", lw=1.5, label=f"smooth w={SMOOTH}")
ax_ps.set_ylabel("P_sink  (×10¹⁰)", fontsize=10)
ax_ps.set_title("Global precipitation sink", fontsize=10)
ax_ps.legend(fontsize=7, ncol=min(8, len(unique_jobs)+1), loc="upper right")
ax_ps.grid(True, alpha=0.2)
ax_ps.relim(); ax_ps.autoscale_view()
vlines_epochs(ax_ps); vlines_jobs(ax_ps)

# -- drift_pct panel -----------------------------------------------------------
for jlabel in unique_jobs:
    mask = np.array([j == jlabel for j in jlabs])
    ax_dr.scatter(gx[mask], dr[mask], s=5, alpha=0.35, color=jcolor[jlabel])
if len(dr) > SMOOTH:
    ax_dr.plot(gx, uniform_filter1d(dr, SMOOTH), color="darkred", lw=1.5, label=f"smooth w={SMOOTH}")
ax_dr.axhline(0, color="k", lw=0.8, ls="--")
ax_dr.set_ylabel("drift_pct  (%)", fontsize=10)
ax_dr.set_xlabel("Global step  (epoch×6101 + WaterFixer step)", fontsize=10)
ax_dr.set_title("Water budget drift  — red=even epochs (high drift start), blue=odd epochs (low drift start)",
                fontsize=9)
ax_dr.legend(fontsize=7, loc="upper left")
ax_dr.grid(True, alpha=0.2)
ax_dr.relim(); ax_dr.autoscale_view()
vlines_epochs(ax_dr); vlines_jobs(ax_dr)
# Shade even epochs
for e in unique_ep:
    if e % 2 == 0:
        x0 = epoch_first_gx[e]
        x1 = epoch_first_gx.get(e+1, gx[-1]+100)
        ax_dr.axvspan(x0, x1, alpha=0.06, color="red")

# -- Epoch-folded drift --------------------------------------------------------
n_pts_per_epoch = int(round(STEPS_PER_EPOCH / dt_median))
for e in unique_ep:
    mask = ep == e
    d    = dr[mask]
    x    = np.linspace(0, 100, len(d))   # normalised 0-100%
    color= "#d04040" if e % 2 == 0 else "#3060c0"
    lw   = 1.4
    ax_fold.plot(x, d, color=color, lw=lw, alpha=0.7, label=f"e{e}")
ax_fold.axhline(0, color="k", lw=0.8, ls="--")
ax_fold.set_xlabel("Epoch progress  (%)", fontsize=10)
ax_fold.set_ylabel("drift_pct  (%)", fontsize=10)
ax_fold.set_title("Epoch-folded drift\n(red=even, blue=odd)", fontsize=9)
ax_fold.legend(fontsize=7, ncol=2)
ax_fold.grid(True, alpha=0.2)
# Annotation
ax_fold.text(0.98, 0.98,
    f"Even start: {np.mean(even_starts):.1f}%\nOdd  start: {np.mean(odd_starts):.1f}%",
    transform=ax_fold.transAxes, ha="right", va="top", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# -- Autocorrelation -----------------------------------------------------------
ax_ac.plot(lag_pos, ac_pos, color="purple", lw=1.2)
ax_ac.axhline(0, color="k", lw=0.7, ls="--")
for pk in peaks[:4]:
    ax_ac.axvline(lag_pos[pk], color="orange", lw=1.0, ls="--", alpha=0.8)
    ax_ac.text(lag_pos[pk], ac_pos[pk]+0.02, f"{lag_pos[pk]:.0f}",
               fontsize=7, ha="center", color="darkorange")
ax_ac.set_xlabel("Lag  (training steps)", fontsize=10)
ax_ac.set_ylabel("r", fontsize=10)
ax_ac.set_title("Autocorrelation of drift_pct\n(per-epoch polynomial detrended)", fontsize=9)
ax_ac.grid(True, alpha=0.2)

out = os.path.join(SCRIPT_DIR, "P_sink.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
