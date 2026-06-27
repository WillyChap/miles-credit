#!/usr/bin/env python3
"""
slim_progressive_forcing.py  (maintainer tool)
----------------------------------------------
Shrink the progressive/transient CAMulator forcing for hosting.

The raw 1980-2014 forcing is ~85 GB: four dynamic fields (SOLIN, SST, ICEFRAC,
co2vmr_3d) stored as float64, plus two unused 2-D fields (PHIS, LANDFRAC). This
keeps only the fields the rollout reads, casts the dynamic fields to float32 (the
model runs in float32), and zlib-compresses -> ~10 GB, a single file under
HuggingFace's per-file limit and a drop-in replacement for inference.

co2vmr_3d is kept as a full (time, lat, lon) field (it is spatially uniform but
the model expects the field shape); compression makes it tiny on disk.

    python slim_progressive_forcing.py \
        --src /glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate_branch_1980_2014.nc \
        --out /glade/derecho/scratch/wchapman/CAMULATOR_FORCING/b.e21.CREDIT_climate_branch_1980_2014_slim.nc
"""
import argparse
import os
import time

import xarray as xr

KEEP = ["SOLIN", "SST", "ICEFRAC", "co2vmr_3d", "z_norm", "LANDM_COSLAT"]  # drop PHIS, LANDFRAC
DYN = ["SOLIN", "SST", "ICEFRAC", "co2vmr_3d"]

DEF_SRC = "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate_branch_1980_2014.nc"
DEF_OUT = "/glade/derecho/scratch/wchapman/CAMULATOR_FORCING/b.e21.CREDIT_climate_branch_1980_2014_slim.nc"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEF_SRC)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--complevel", type=int, default=4)
    args = ap.parse_args()

    ds = xr.open_dataset(args.src, chunks={"time": 200})[KEEP]
    for v in DYN:
        ds[v] = ds[v].astype("float32")
    enc = {v: {"zlib": True, "complevel": args.complevel, "dtype": "float32"} for v in DYN}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"writing {args.out} ...", flush=True)
    t0 = time.time()
    ds.to_netcdf(args.out, encoding=enc)
    print(f"done in {time.time() - t0:.0f}s, size {os.path.getsize(args.out) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
