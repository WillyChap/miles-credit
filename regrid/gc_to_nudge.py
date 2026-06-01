"""Stand-alone CLI: GraphCast prediction -> CAM6 nudge file.

This is the "graphcast-only" mode end-point. Input is a GraphCast prediction
netCDF (from regrid.run_graphcast_inference). Output is a sumo_cam6_nudge.*.nc
file in the run directory that the CAM6 nudging toolbox reads.

For supermodel mode (CAMulator + GraphCast blended), this is one of two
sources fed to regrid.blender.blend before write_nudge_file. The coordinator
chains the steps; this CLI is for direct testing.

Run:
    python -m regrid.gc_to_nudge \\
        --prediction /scratch/.../gc_forecast_1980-01-06-06000.nc \\
        --reference-h1 /scratch/.../h1.1980-01-06-00000.nc \\
        --out /scratch/.../run/sumo_cam6_nudge.1980-01-06-21600.nc

The reference-h1 supplies:
    - CAM6 grid + hybrid coord (one-time stencil precompute)
    - PS at coupling time T (used for hybrid pressure at T+6h target)
    - Structural template for the nudge file netCDF layout
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from .blender import NUDGED_VARS, blend, default_blend_config
from .reverse_translate import ReverseTranslator, write_nudge_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", type=Path, required=True,
                    help="GraphCast prediction netCDF (regrid.run_graphcast_inference output)")
    ap.add_argument("--reference-h1", type=Path, required=True,
                    help="CAM6 h1 file at coupling time T (for PS + grid + template)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output nudge file path (atomic-rename target)")
    args = ap.parse_args()

    # Build reverse translator (stencil precompute).
    tr = ReverseTranslator.build(args.reference_h1)

    # PS at coupling time T from h1.
    h1 = xr.open_dataset(args.reference_h1)
    try:
        ps_cam = np.asarray(h1["PS"].isel(time=0).values, dtype=np.float64)
    finally:
        h1.close()

    # GC prediction needs decode_times=False to avoid the xarray time-attr conflict.
    pred = xr.open_dataset(args.prediction, decode_times=False)
    try:
        gc_targets = tr.translate_prediction(pred, ps_cam, time_index=0, batch_index=0)
    finally:
        pred.close()

    # Single-source blend (pass-through) so the nudge file gets the
    # non-negativity clip for Q applied consistently.
    nudge = blend(cam_pred=None, gc_pred=gc_targets, cfg_blend=default_blend_config())

    # Diagnostic
    print("== GC-only nudge target ==")
    for v in NUDGED_VARS:
        arr = nudge[v]
        print(f"  {v:>3s}  min={arr.min():+10.4g}  max={arr.max():+10.4g}  "
              f"mean={arr.mean():+10.4g}  std={arr.std():10.4g}")

    write_nudge_file(nudge, args.reference_h1, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
