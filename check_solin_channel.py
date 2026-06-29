"""
check_solin_channel.py
----------------------
Load the x1tensor.pt saved by eval_2step_rollout.py and inspect channels
130-135 to verify channel 132 = SOLIN (day/night terminator pattern).

Expected layout (static_first=True, ConcatPreblock):
  0-127  : upper-air U,V,T,Qtot  (32 levels × 4 vars)
  128-129: surface PS, TREFHT
  130    : static   z_norm
  131    : static   LANDM_COSLAT
  132    : dyn_forcing SOLIN      <- should show sunlit hemisphere
  133    : dyn_forcing SST
  134    : dyn_forcing ICEFRAC
  135    : dyn_forcing co2vmr_3d

Usage:
    python check_solin_channel.py
"""

import numpy as np
import torch

TENSOR_PATH = "/glade/derecho/scratch/wchapman/x1tensor.pt"

print(f"Loading: {TENSOR_PATH}")
x = torch.load(TENSOR_PATH, map_location="cpu", weights_only=False)
print(f"  shape: {tuple(x.shape)}   dtype: {x.dtype}")
# x shape: [B, C, H, W]  (already flat after preblock)

# Take first sample
x0 = x[0]   # [C, H, W]
C, H, W = x0.shape
print(f"  Total channels: {C}")

channels = {
    128: "PS (surface pressure)",
    129: "TREFHT (2m temp)",
    130: "z_norm (static terrain)      <- expect terrain pattern",
    131: "LANDM_COSLAT (land mask)     <- expect land/ocean mask",
    132: "SOLIN (TOA solar)            <- expect day/night terminator",
    133: "SST                          <- expect ocean temp pattern",
    134: "ICEFRAC                      <- expect polar ice",
    135: "co2vmr_3d                    <- expect near-uniform",
}

print()
for idx, name in channels.items():
    if idx >= C:
        print(f"  ch {idx}: OUT OF RANGE")
        continue
    ch = x0[idx].numpy()
    frac_near_zero = (np.abs(ch) < 0.05).mean()
    print(f"  ch {idx}: min={ch.min():7.3f}  max={ch.max():7.3f}  "
          f"mean={ch.mean():7.3f}  std={ch.std():6.3f}  "
          f"~0: {frac_near_zero:.1%}   {name}")

# SOLIN diagnostic: expect ~50% near-zero (night side) and large positive max
solin = x0[132].numpy()
print(f"\nSOLIN (ch132) night-side fraction (< 0.05 normalized): {(solin < 0.05).mean():.1%}")
print("  -> if ~40-60%, this confirms ch132 is SOLIN (half the globe is in darkness)")

# ── plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()

    for ax, (idx, label) in zip(axes, channels.items()):
        if idx >= C:
            ax.axis("off")
            continue
        data = x0[idx].numpy()
        cmap = "hot" if idx == 132 else ("RdBu_r" if idx in (128, 129, 133) else "viridis")
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(f"ch{idx}: {label.split('<')[0].split('(')[0].strip()}", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.suptitle("x1tensor.pt — channels 128-135\n"
                 "ch132 (SOLIN) should show a day/night terminator", fontsize=11)
    fig.tight_layout()
    plot_path = "/tmp/input_channels_check.png"
    fig.savefig(plot_path, dpi=100)
    print(f"\nPlot saved to: {plot_path}")
    print("  -> scp <node>:/tmp/input_channels_check.png .")
except Exception as e:
    print(f"\nPlot failed ({e})")

print("\nDone.")
