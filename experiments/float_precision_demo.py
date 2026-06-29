#!/usr/bin/env python3
"""E2 — float32 vs float64 global-sum rounding demo.

Reproduces, on real CAM6 data, the accumulation error that motivated the float64
change in `credit/physics_core.py::weighted_sum`. The global water-budget terms are
area-weighted sums over ~55,000 grid cells (192 x 288). Summing those contributions in
float32 introduces a fractional error large enough to masquerade as a real budget drift;
accumulating in float64 removes it.

This script computes the global area-weighted precipitation sink and evaporation source
for a few timesteps of one training year, once with float32 accumulation and once with
float64, and reports the fractional difference. It reuses the exact area weights and flux
definitions from `compute_water_budget_truth.py` so the comparison matches training.

CPU only, runs in seconds. Output is printed and written to --out.
"""

import argparse
import numpy as np
import torch
import xarray as xr

from compute_water_budget_truth import build_physics_core, RHO_WATER, N_SECONDS, ZARR_TMPL


def frac_pct(a: float, b: float) -> float:
    """100 * (a - b) / b, guarding against zero."""
    return 100.0 * (a - b) / b if b != 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", default="1980", help="training year to sample")
    p.add_argument("--nsteps", type=int, default=10, help="timesteps to average over")
    p.add_argument("--out", default="experiments/out/float_precision_demo.txt")
    args = p.parse_args()

    core = build_physics_core()
    area = core.area  # (lat, lon) area weights, same object used by weighted_sum

    zarr_path = ZARR_TMPL.format(year=args.year)
    ds = xr.open_zarr(zarr_path)
    nt = min(args.nsteps, ds.dims["time"])

    rows = []
    for t in range(nt):
        prect = torch.from_numpy(ds["PRECT"].isel(time=t).values.astype(np.float32))
        qflx = torch.from_numpy(ds["QFLX"].isel(time=t).values.astype(np.float32))
        p_flux = prect * RHO_WATER / N_SECONDS
        e_flux = qflx * RHO_WATER / N_SECONDS

        for name, fld in (("P_sink", p_flux), ("E_src", e_flux)):
            # float32 accumulation (the old behavior) vs float64 (the fix)
            s32 = torch.sum(fld.float() * area.float()).item()
            s64 = torch.sum(fld.double() * area.double()).item()
            rows.append((t, name, s32, s64, frac_pct(s32, s64)))

    # summarize
    import statistics

    lines = ["step  term     sum_f32          sum_f64          err_pct"]
    for t, name, s32, s64, err in rows:
        lines.append(f"{t:<5d} {name:<7s} {s32: .6e}   {s64: .6e}   {err:+.4f}")
    p_err = [r[4] for r in rows if r[1] == "P_sink"]
    e_err = [r[4] for r in rows if r[1] == "E_src"]
    lines.append("")
    lines.append(f"mean |err| P_sink: {statistics.mean(map(abs, p_err)):.4f}%  "
                 f"E_src: {statistics.mean(map(abs, e_err)):.4f}%  "
                 f"(over {nt} steps of {args.year})")

    report = "\n".join(lines)
    print(report)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
