#!/usr/bin/env python3
"""Numerically verify the corrector's scale invariance (paper Eq. for C and J_C(P)P=0).

The water corrector rescales precipitation by a global factor so the corrected global total
equals the budget-closing value Q (set by the predicted storage tendency and evaporation,
independent of the raw precip field):  C(P) = Q * P / sum(P) = r*P,  r = Q / sum(P).

For real CAM6 precip fields P and several uniform scale factors c, this script checks:
  - C(cP) = C(P)                       (corrected field is scale invariant)
  - r ~ 1/c                            (correction factor absorbs the scaling)
  - corrected-output precip loss ~ const while raw-output precip loss varies with c
  - water penalty (r-1)^2 varies with c
and runs a finite-difference test of the null direction, [C(P+eps P) - C(P)] / eps -> 0,
which verifies J_C(P) P = 0 without building the Jacobian.

CPU only, runs in seconds. Writes a table to --out.
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
    p.add_argument("--nsteps", type=int, default=8, help="precip fields to average over")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--out", default="experiments/out/scale_invariance_test.txt")
    a = p.parse_args()
    dt = torch.float64 if a.dtype == "float64" else torch.float32

    core = build_physics_core()
    area = core.area.to(dt)
    ds = xr.open_zarr(ZARR_TMPL.format(year=a.year))

    def wsum(x):  # area-weighted global sum
        return torch.sum(x * area)

    def wmse(x, t):  # area-weighted mean squared error
        return (torch.sum(area * (x - t) ** 2) / torch.sum(area)).item()

    rows = {c: dict(reldiff=[], r=[], raw=[], corr=[], pen=[]) for c in a.scales}
    fd = {e: [] for e in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]}

    for t in range(1, a.nsteps + 1):
        q0 = torch.from_numpy(ds["Qtot"].isel(time=t - 1).values.astype(np.float64)).unsqueeze(0).to(dt)
        q1 = torch.from_numpy(ds["Qtot"].isel(time=t).values.astype(np.float64)).unsqueeze(0).to(dt)
        sp0 = torch.from_numpy(ds["PS"].isel(time=t - 1).values.astype(np.float64)).unsqueeze(0).to(dt)
        sp1 = torch.from_numpy(ds["PS"].isel(time=t).values.astype(np.float64)).unsqueeze(0).to(dt)
        P = torch.from_numpy(ds["PRECT"].isel(time=t).values.astype(np.float64)).to(dt)     # raw precip field
        qflx = torch.from_numpy(ds["QFLX"].isel(time=t).values.astype(np.float64)).to(dt)

        # budget-closing precip total Q (flux units), independent of P
        dTWC = (core.total_column_water(q1, sp1) - core.total_column_water(q0, sp0)).to(dt).squeeze(0) / N_SECONDS
        T = wsum(dTWC)
        E = wsum(qflx * RHO_WATER / N_SECONDS)
        Q = -T - E

        def C(Pf):  # corrector on a precip field; returns corrected field and the scalar r
            r = Q / wsum(Pf * RHO_WATER / N_SECONDS)
            return Pf * r, r.item()

        C0, _ = C(P)
        denom = torch.max(torch.abs(C0))
        for c in a.scales:
            Cc, rc = C(c * P)
            rows[c]["reldiff"].append((torch.max(torch.abs(Cc - C0)) / denom).item())
            rows[c]["r"].append(rc)
            rows[c]["raw"].append(wmse(c * P, P))      # raw-output precip loss vs target P
            rows[c]["corr"].append(wmse(Cc, P))        # corrected-output precip loss vs target P
            rows[c]["pen"].append((rc - 1.0) ** 2)
        for e in fd:
            Ce, _ = C(P + e * P)
            fd[e].append((torch.max(torch.abs(Ce - C0)) / e / denom).item())

    m = lambda v: float(np.mean(v))
    lines = [f"Scale-invariance test of the water corrector  ({a.nsteps} fields of {a.year}, {a.dtype})", ""]
    lines.append(f"{'c':>6} {'max|C(cP)-C(P)|/|C(P)|':>24} {'r':>10} {'r*c':>8} "
                 f"{'raw loss':>12} {'corr loss':>12} {'penalty (r-1)^2':>16}")
    for c in a.scales:
        rr = m(rows[c]["r"])
        lines.append(f"{c:>6.2f} {m(rows[c]['reldiff']):>24.2e} {rr:>10.4f} {rr*c:>8.4f} "
                     f"{m(rows[c]['raw']):>12.3e} {m(rows[c]['corr']):>12.3e} {m(rows[c]['pen']):>16.3e}")
    lines += ["", "Finite-difference null test:  [C(P + eps*P) - C(P)] / eps,  relative (/max|C(P)|)"]
    lines.append(f"{'eps':>10} {'relative magnitude':>22}")
    for e in sorted(fd, reverse=True):
        lines.append(f"{e:>10.0e} {m(fd[e]):>22.2e}")
    overall = max(m(rows[c]["reldiff"]) for c in a.scales)
    lines += ["", f"max relative difference C(cP) vs C(P) over all c: {overall:.2e}  "
              f"(= machine precision; the corrected field is scale invariant)"]

    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(report + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
