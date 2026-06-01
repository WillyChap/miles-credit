"""Inspect a GraphCast prediction file and check it against ERA5 ground truth.

Two things this answers:
  1. "Does the GC output look physically plausible?"
       -> Global statistics per variable, per level
  2. "Does GC predict +6h close to ERA5 +6h?"
       -> bias / RMSE / max|err| against the ERA5 fields at the matching UTC

Reusable for the reverse-direction (translates back to CAM6 hybrid and
reports the nudge target stats too, so we can sanity-check the round-trip
before any actual coupling).

Run:
    python -m regrid.inspect_gc_prediction \\
        --prediction /scratch/.../gc_forecast_1980-01-06-06000.nc \\
        [--reference-h1 /scratch/.../h1.1980-01-06-00000.nc]
        [--check-reverse]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from . import config as cfg
from .blender import (
    NUDGED_VARS,
    blend,
    default_blend_config,
    report_blend,
    stratosphere_lean_blend_config,
)
from .reverse_translate import ReverseTranslator
from .validate_against_era5 import (
    ERA5_VARS,
    TERRAIN_BOXES,
    bias_rmse,
    load_era5_pl,
)


def report_global_stats(ds: xr.Dataset, name_field: str = "prediction") -> None:
    print(f"\n== Global stats: {name_field} ==")
    for n, da in ds.data_vars.items():
        v = np.asarray(da.values)
        if not np.issubdtype(v.dtype, np.floating):
            continue
        print(f"  {n:30s} shape={v.shape}  "
              f"min={float(v.min()):+11.4g}  "
              f"max={float(v.max()):+11.4g}  "
              f"mean={float(v.mean()):+11.4g}  "
              f"std={float(v.std()):11.4g}  "
              f"NaN={int(np.isnan(v).sum())}")


def report_pressure_columns(pred_atmo: dict, tropics_only: bool = True) -> None:
    """Print a few representative columns to confirm vertical structure looks
    like real atmosphere (T decreasing with height, U/V finite, etc.)."""
    if tropics_only:
        # Tropical sample: lat=0, lon=180 in GC grid.
        j = int(np.argmin(np.abs(np.linspace(90, -90, cfg.GC_NLAT) - 0.0)))
        i = int(np.argmin(np.abs(np.linspace(0, 359.75, cfg.GC_NLON) - 180.0)))
        label = "tropics (0 N, 180 E)"
    else:
        j, i = 200, 200
        label = f"sample (lat_idx={j}, lon_idx={i})"
    print(f"\n== Pressure column at {label} ==")
    levels = cfg.PRESSURE_LEVELS_HPA
    keys = [k for k in ("temperature", "u_component_of_wind",
                        "v_component_of_wind", "specific_humidity",
                        "geopotential") if k in pred_atmo]
    header = f"{'plev':>6}  " + "  ".join(f"{k.split('_')[0][:6]:>7}" for k in keys)
    print(header)
    for li, L in enumerate(levels):
        row = f"{L:6.0f}  "
        for k in keys:
            v = pred_atmo[k][li, j, i]
            row += f"  {v:7.2g}"
        print(row)


def report_surface_summary(pred_surf: dict) -> None:
    print("\n== Surface variables ==")
    for k, arr in pred_surf.items():
        print(f"  {k:30s}  "
              f"min={float(arr.min()):+11.4g}  "
              f"max={float(arr.max()):+11.4g}  "
              f"mean={float(arr.mean()):+11.4g}  "
              f"std={float(arr.std()):11.4g}")


def compare_to_era5(
    pred: xr.Dataset,
    prediction_time: datetime,
    vars_to_check: tuple[str, ...] = (
        "temperature", "u_component_of_wind", "v_component_of_wind",
        "specific_humidity", "geopotential", "vertical_velocity",
    ),
) -> None:
    """Per-variable comparison. Loads one ERA5 file at a time and frees it
    before the next, so peak RAM stays at ~4 GB instead of 6x that.
    """
    print(f"\n== GraphCast +6h vs ERA5 at {prediction_time.isoformat()} ==")
    print("    bias = GC - ERA5     (negative = GC lower)")
    print()
    for gv in vars_to_check:
        if gv not in pred:
            continue
        try:
            era5 = load_era5_pl(prediction_time, gv)
        except FileNotFoundError as exc:
            print(f"  {gv}: ERA5 file not found ({exc.filename})")
            continue
        pa = np.asarray(pred[gv].isel(batch=0, time=0).values, dtype=np.float64)
        b, r, m = bias_rmse(pa, era5)
        print(f"  {gv:30s}  bias={b:+.4g}  rmse={r:.4g}  max|err|={m:.4g}")
        # Per-level RMSE at a few key altitudes
        per_lvl = np.sqrt(((pa - era5) ** 2).reshape(pa.shape[0], -1).mean(axis=1))
        compact = " ".join(
            f"{cfg.PRESSURE_LEVELS_HPA[i]}hPa={per_lvl[i]:.3g}"
            for i in (0, 10, 17, 25, 30, 33, 36)
        )
        print(f"        per-level RMSE: {compact}")
        # Free both arrays before the next variable so we don't accumulate.
        del era5, pa, per_lvl
        import gc as _gc; _gc.collect()


def check_reverse_roundtrip(pred: xr.Dataset, ref_h1: Path) -> None:
    """Reverse-translate GC prediction -> CAM6 hybrid; report nudge stats."""
    print("\n== Reverse-direction check (GC -> CAM6 hybrid nudge target) ==")
    tr = ReverseTranslator.build(ref_h1)
    h1 = xr.open_dataset(ref_h1)
    try:
        ps_cam = np.asarray(h1["PS"].isel(time=0).values, dtype=np.float64)
    finally:
        h1.close()
    targets = tr.translate_prediction(pred, ps_cam, time_index=0, batch_index=0)
    for k, arr in targets.items():
        print(f"  CAM6 nudge {k:>3s}  shape={arr.shape}  "
              f"min={float(arr.min()):+10.4g}  "
              f"max={float(arr.max()):+10.4g}  "
              f"mean={float(arr.mean()):+10.4g}  "
              f"NaN={int(np.isnan(arr).sum())}")
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", type=Path, required=True,
                    help="Output netCDF from run_graphcast_inference.py")
    ap.add_argument("--reference-h1", type=Path, default=None,
                    help="Any CAM6 h1 (for reverse-translator stencil) -- "
                         "required if --check-reverse is set")
    ap.add_argument("--check-reverse", action="store_true",
                    help="Also reverse-translate to CAM6 hybrid nudge target.")
    ap.add_argument("--no-era5", action="store_true",
                    help="Skip ERA5 comparison (e.g., for synthetic dates).")
    ap.add_argument("--camulator-prediction", type=Path, default=None,
                    help="If provided (a CAMulator prediction file on CAM6 grid), "
                         "blend with the reverse-translated GC and report. "
                         "Requires --check-reverse.")
    ap.add_argument("--strat-lean", action="store_true",
                    help="Use stratosphere-lean blend weights (50/50 troposphere, "
                         "80-90% CAMulator above 50 hPa). Default is 50/50.")
    args = ap.parse_args()

    # decode_times=False: the file's time dim is a timedelta with a `dtype`
    # attr that conflicts with xarray's CF decoder. We don't need time decoded.
    pred = xr.open_dataset(args.prediction, decode_times=False)
    print(f"Loaded prediction: {args.prediction}")
    print(f"  dims: {dict(pred.sizes)}")
    print(f"  vars: {list(pred.data_vars)}")
    if "datetime" in pred.coords:
        print(f"  datetime: {pred.datetime.values}")

    # Stats: walk one variable at a time so we never hold more than one
    # full atmospheric array (~150 MB f32) in memory.
    report_global_stats(pred, "GraphCast prediction")

    # For the column and surface summaries, materialize only the small
    # subsets we actually print.
    pred_atmo_subset = {}
    pred_surf_subset = {}
    for n, da in pred.data_vars.items():
        sliced = da.isel(batch=0, time=0)
        if "level" in da.dims:
            # Only keep the one tropical column we'll print + the per-level
            # slice; do not retain the full 3D array.
            arr = np.asarray(sliced.values)
            pred_atmo_subset[n] = arr  # small enough to keep through the next call
        else:
            pred_surf_subset[n] = np.asarray(sliced.values)
    report_pressure_columns(pred_atmo_subset, tropics_only=True)
    report_surface_summary(pred_surf_subset)
    del pred_atmo_subset, pred_surf_subset
    import gc as _gc; _gc.collect()

    if not args.no_era5:
        # Pred's prediction time = bundle's last datetime (T+6h).
        if "datetime" in pred.coords:
            when_ns = pred.datetime.isel(batch=0, time=0).values.astype("int64")
            when = datetime.utcfromtimestamp(when_ns // 1_000_000_000)
        else:
            # Fall back: derive from --reference-h1 + 6h.
            if args.reference_h1 is None:
                print("\n(skipping ERA5 comparison: no datetime + no --reference-h1)")
                when = None
            else:
                from .validate_against_era5 import datetime_from_h1_name
                when = datetime_from_h1_name(args.reference_h1) + timedelta(hours=6)
        if when is not None:
            compare_to_era5(pred, when)

    if args.check_reverse:
        if args.reference_h1 is None:
            raise SystemExit("--check-reverse requires --reference-h1")
        gc_targets = check_reverse_roundtrip(pred, args.reference_h1)

        if args.camulator_prediction is not None:
            print(f"\nLoading CAMulator prediction from {args.camulator_prediction}")
            cam_ds = xr.open_dataset(args.camulator_prediction)
            # Pull U, V, T, Q on CAM6 hybrid grid. Exact var-name layout will
            # depend on which CAMulator output we wire in; for now expect
            # CAM-style names.
            cam_pred = {v: np.asarray(cam_ds[v].squeeze().values, dtype=np.float64)
                        for v in NUDGED_VARS}
            cam_ds.close()
            cfg_b = (
                stratosphere_lean_blend_config()
                if args.strat_lean else default_blend_config()
            )
            blended = blend(cam_pred, gc_targets, cfg_b)
            report_blend(cam_pred, gc_targets, blended)
            print("\nBlended nudge target ready (CAM6 grid). "
                  "Pass to regrid.reverse_translate.write_nudge_file to ship.")


if __name__ == "__main__":
    main()
