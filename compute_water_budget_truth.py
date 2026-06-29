#!/usr/bin/env python3
"""
compute_water_budget_truth.py

Compute dTWC/dt, E_src, P_sink, residual, and drift_pct at every 6-hourly
timestep in one (or more) years of training data, using the exact same physics
as GlobalWaterFixer in the training loop.

Formula (identical to _postblock.py GlobalWaterFixer.forward):
    precip_flux = PRECT * RHO_WATER / N_seconds
    evapor_flux = QFLX  * RHO_WATER / N_seconds
    dTWC_dt     = (TWC[t+1] - TWC[t]) / N_seconds          # TWC via sigma-level integral
    TWC_sum     = weighted_sum(dTWC_dt)                     # global area-weighted sum
    E_sum       = weighted_sum(evapor_flux)
    P_sum       = weighted_sum(precip_flux)
    residual    = -TWC_sum - E_sum - P_sum
    drift_pct   = 100 * residual / P_sum

Usage:
    python compute_water_budget_truth.py --years 1980              # single year
    python compute_water_budget_truth.py --years 1980 1981 1982    # multiple years
    python compute_water_budget_truth.py --years 1980-2012         # range
    python compute_water_budget_truth.py --out water_budget.csv    # custom output path
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

# ── repo path ────────────────────────────────────────────────────────────────
REPO = "/glade/work/wchapman/Roman_Coupling/train_johns"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from credit.physics_core import physics_hybrid_sigma_level  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────
RHO_WATER = 1000.0          # kg/m³
N_SECONDS = 6 * 3600        # 21600 s per 6-h step (lead_time_periods=6)
CHUNK = 32                  # timesteps per batch (memory limit for Qtot)

STATICS = (
    "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/"
    "b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
)
ZARR_TMPL = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "b.e21.CREDIT_climate_branch_1980_{year}_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)


def build_physics_core() -> physics_hybrid_sigma_level:
    """Initialize the same physics core as GlobalWaterFixer in training."""
    ds = xr.open_dataset(STATICS)
    lat2d = torch.from_numpy(ds["lat2d"].values).float()
    lon2d = torch.from_numpy(ds["lon2d"].values).float()
    hyai  = torch.from_numpy(ds["hyai"].values).float()
    hybi  = torch.from_numpy(ds["hybi"].values).float()
    # midpoint=True matches training config global_water_fixer.midpoint: True
    return physics_hybrid_sigma_level(lon2d, lat2d, hyai, hybi, midpoint=True)


def process_year(year: int, core: physics_hybrid_sigma_level) -> pd.DataFrame:
    zarr_path = ZARR_TMPL.format(year=year)
    if not os.path.exists(zarr_path):
        print(f"  [skip] zarr not found: {zarr_path}")
        return pd.DataFrame()

    print(f"  opening {zarr_path}")
    ds = xr.open_zarr(zarr_path)
    ntimes = ds.dims["time"]
    times  = ds["time"].values          # cftime objects

    # Pre-load 2-D fields into numpy (cheap: 1460×192×288 ≈ 0.3 GB each)
    prect_np = ds["PRECT"].values       # (T, lat, lon)
    qflx_np  = ds["QFLX"].values       # (T, lat, lon)
    ps_np    = ds["PS"].values          # (T, lat, lon)

    records = []
    # Step t computes: Qtot[t-1]→Qtot[t], PS[t-1]→PS[t], PRECT[t], QFLX[t]
    # So loop starts at t=1
    for t in range(1, ntimes):
        # Load Qtot for t-1 and t (lazy zarr read per step keeps memory low)
        qt_prev = torch.from_numpy(
            ds["Qtot"].isel(time=t - 1).values.astype(np.float32)
        ).unsqueeze(0)   # (1, level, lat, lon)
        qt_curr = torch.from_numpy(
            ds["Qtot"].isel(time=t).values.astype(np.float32)
        ).unsqueeze(0)

        sp_prev = torch.from_numpy(ps_np[t - 1].astype(np.float32)).unsqueeze(0)  # (1, lat, lon)
        sp_curr = torch.from_numpy(ps_np[t    ].astype(np.float32)).unsqueeze(0)

        # Total column water [kg/m²] — same call as postblock
        with torch.no_grad():
            TWC_prev = core.total_column_water(qt_prev, sp_prev)  # (1, lat, lon)
            TWC_curr = core.total_column_water(qt_curr, sp_curr)

            dTWC_dt = (TWC_curr - TWC_prev) / N_SECONDS
            TWC_sum = core.weighted_sum(dTWC_dt, axis=(-2, -1))   # (1,)

            prect_t = torch.from_numpy(prect_np[t].astype(np.float32)).unsqueeze(0)
            qflx_t  = torch.from_numpy(qflx_np[t].astype(np.float32)).unsqueeze(0)

            precip_flux = prect_t * RHO_WATER / N_SECONDS
            evapor_flux = qflx_t  * RHO_WATER / N_SECONDS

            P_sum    = core.weighted_sum(precip_flux, axis=(-2, -1))
            E_sum    = core.weighted_sum(evapor_flux, axis=(-2, -1))
            residual = -TWC_sum - E_sum - P_sum

        P_val   = P_sum.item()
        E_val   = E_sum.item()
        R_val   = residual.item()
        TWC_val = TWC_sum.item()
        drift   = 100.0 * R_val / P_val if abs(P_val) > 0 else float("nan")

        records.append(
            {
                "time":      str(times[t]),
                "dTWC_dt":   TWC_val,
                "E_src":     E_val,
                "P_sink":    P_val,
                "residual":  R_val,
                "drift_pct": drift,
            }
        )

        if t % 200 == 0:
            print(f"    t={t}/{ntimes-1}  P_sink={P_val:.4e}  E_src={E_val:.4e}"
                  f"  drift={drift:.2f}%")

    return pd.DataFrame(records)


def parse_years(tokens: list[str]) -> list[int]:
    """Accept '1980', '1980-2012', or multiple individual years."""
    years = []
    for tok in tokens:
        if "-" in tok and len(tok) > 5:        # range like 1980-2012
            a, b = tok.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(tok))
    return sorted(set(years))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--years", nargs="+", default=["1980"],
        help="Year(s) to process: '1980', '1980 1981', or '1980-2012'",
    )
    parser.add_argument(
        "--out", default="water_budget_truth.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    print(f"Years to process: {years}")
    print("Building physics core (sigma-level, midpoint=True) ...")
    core = build_physics_core()
    print("  area shape:", core.area.shape, "  dtype:", core.area.dtype)

    all_frames = []
    for yr in years:
        print(f"\nYear {yr}:")
        df = process_year(yr, core)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No data processed — check zarr paths.")
        return

    result = pd.concat(all_frames, ignore_index=True)
    result.to_csv(args.out, index=False)
    print(f"\nSaved {len(result)} rows → {args.out}")

    # Summary stats
    print("\nSummary:")
    print(result[["P_sink", "E_src", "residual", "drift_pct"]].describe().to_string())
    print(f"\nMean drift_pct : {result.drift_pct.mean():.3f}%")
    print(f"Std  drift_pct : {result.drift_pct.std():.3f}%")
    print(f"P_sink > 0 frac: {(result.P_sink > 0).mean():.3f}")
    print(f"E_src  < 0 frac: {(result.E_src  < 0).mean():.3f}  (negative = upward evap)")


if __name__ == "__main__":
    main()
