#!/usr/bin/env python3
"""E2 — budget-accumulation precision demo (drift-level).

Motivates the float64 accumulation in `credit/physics_core.py::weighted_sum`. The global
water-budget drift is a small residual of three large area-weighted sums, so it is
sensitive to the precision the sums are accumulated in. This script computes, for several
timesteps of one CAM6 year, the drift with the area-weighted global sums accumulated in
float64 (reference), float32, and bfloat16, and reports the drift error each precision
introduces relative to float64.

Key finding (see paper Section 4): float32 accumulation is already accurate -- torch.sum
uses pairwise summation, so summing ~55,000 single-precision cells loses nothing. The
error appears at the bfloat16 precision used for mixed-precision (AMP) training, where the
drift error reaches a few tenths of a percent, comparable to the natural per-step spread.
Casting to float64 before the sum removes it regardless of the input dtype.

CPU only, runs in seconds. Output printed and written to --out.
"""

import argparse
import os
import sys

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_water_budget_truth import build_physics_core, RHO_WATER, N_SECONDS, ZARR_TMPL  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", default="1980")
    p.add_argument("--nsteps", type=int, default=30)
    p.add_argument("--out", default="experiments/out/float_precision_demo.txt")
    args = p.parse_args()

    core = build_physics_core()
    area64 = core.area.double()
    ds = xr.open_zarr(ZARR_TMPL.format(year=args.year))
    nt = min(args.nsteps + 1, ds.dims["time"])

    def fields_f64(t):
        q0 = torch.from_numpy(ds["Qtot"].isel(time=t - 1).values.astype(np.float64)).unsqueeze(0)
        q1 = torch.from_numpy(ds["Qtot"].isel(time=t).values.astype(np.float64)).unsqueeze(0)
        sp0 = torch.from_numpy(ds["PS"].isel(time=t - 1).values.astype(np.float64)).unsqueeze(0)
        sp1 = torch.from_numpy(ds["PS"].isel(time=t).values.astype(np.float64)).unsqueeze(0)
        dtwc = (core.total_column_water(q1, sp1) - core.total_column_water(q0, sp0)) / N_SECONDS
        pr = torch.from_numpy(ds["PRECT"].isel(time=t).values.astype(np.float64)).unsqueeze(0)
        qf = torch.from_numpy(ds["QFLX"].isel(time=t).values.astype(np.float64)).unsqueeze(0)
        return dtwc, pr * RHO_WATER / N_SECONDS, qf * RHO_WATER / N_SECONDS

    def drift(dtwc, pflux, eflux, dtype):
        # accumulate the area-weighted global sums in `dtype`; float64 is the fixed path
        s = lambda x: torch.sum(x.to(dtype) * area64.to(dtype)).double()
        T, E, P = s(dtwc), s(eflux), s(pflux)
        return (100.0 * (-T - E - P) / P).item()

    ref, e32, ebf = [], [], []
    for t in range(1, nt):
        d, pf, ef = fields_f64(t)
        r = drift(d, pf, ef, torch.float64)
        ref.append(r)
        e32.append(abs(drift(d, pf, ef, torch.float32) - r))
        ebf.append(abs(drift(d, pf, ef, torch.bfloat16) - r))

    lines = [
        f"Budget-accumulation precision demo ({nt - 1} steps of {args.year})",
        f"natural drift (float64 reference): mean {np.mean(ref):+.3f}%  std {np.std(ref):.3f}%",
        "",
        f"{'accumulation dtype':>20}   mean|drift err vs f64|   max",
        f"{'float32':>20}   {np.mean(e32):>18.4f}%   {np.max(e32):.4f}%",
        f"{'bfloat16 (AMP)':>20}   {np.mean(ebf):>18.4f}%   {np.max(ebf):.4f}%",
        "",
        "float32 is accurate (pairwise torch.sum); bfloat16 (the AMP compute dtype) is the",
        "source of the budget rounding error. Casting to float64 before the sum removes it.",
    ]
    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
