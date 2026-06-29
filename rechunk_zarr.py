"""
Rechunk CAMulator zarr stores: float64, chunk=(4,...) → float32, chunk=(1,...)
Writes new stores to DST_DIR without modifying originals.

Usage:
    python rechunk_zarr.py --year 1980
    python rechunk_zarr.py --year all
"""

import argparse
import glob
import os
import time

import numpy as np
import zarr
from numcodecs import Blosc

SRC_DIR = "/glade/derecho/scratch/wchapman/b_credit_runs"
DST_DIR = "/glade/derecho/scratch/wchapman/b_credit_runs_f32_02"

COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
BATCH_T = 10  # timesteps read at once to keep RAM reasonable


def rechunk_one(src_path: str, dst_path: str):
    fname = os.path.basename(src_path)

    if os.path.exists(dst_path):
        print(f"[SKIP] {fname} already exists at destination", flush=True)
        return

    print(f"[START] {fname}", flush=True)
    t0 = time.time()

    src = zarr.open(src_path, mode="r", zarr_format=2)
    dst = zarr.open(dst_path, mode="w", zarr_format=2)

    # Copy group-level attributes (coordinate metadata etc.)
    dst.attrs.update(dict(src.attrs))

    for varname in sorted(src.keys()):
        arr = src[varname]
        shape = arr.shape
        ndim = arr.ndim

        # Preserve non-float dtypes (e.g. int coords); cast float64 → float32
        if np.issubdtype(arr.dtype, np.floating):
            out_dtype = np.float32
        else:
            out_dtype = arr.dtype

        # New chunks: time-dim=1, everything else full
        if ndim == 1:
            new_chunks = shape          # 1-D coords/scalars — keep as-is
        else:
            new_chunks = (1,) + shape[1:]  # (1, lev, lat, lon) or (1, lat, lon)

        # Preserve fill_value from source (orig is nan for floats).
        # Without this xarray masks every zero value as NaN on open_zarr.
        src_fill = arr.fill_value

        dst_arr = dst.zeros(
            name=varname,
            shape=shape,
            chunks=new_chunks,
            dtype=out_dtype,
            fill_value=src_fill,
            compressor=COMPRESSOR,
            zarr_format=2,
        )

        # Copy array attributes (critically includes _ARRAY_DIMENSIONS for xarray)
        dst_arr.attrs.update(dict(arr.attrs))

        # Copy in temporal batches to keep memory bounded
        if ndim > 1:
            nt = shape[0]
            for i in range(0, nt, BATCH_T):
                end = min(i + BATCH_T, nt)
                dst_arr[i:end] = arr[i:end].astype(out_dtype)
        else:
            dst_arr[:] = arr[:].astype(out_dtype)

        print(
            f"  {varname:20s}  {str(shape):30s} → chunks={str(new_chunks):30s}  {arr.dtype}→{np.dtype(out_dtype).name}",
            flush=True,
        )

    zarr.consolidate_metadata(dst_path)
    elapsed = time.time() - t0
    print(f"[DONE] {fname}  ({elapsed/60:.1f} min)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        required=True,
        help="4-digit year to rechunk, or 'all' to process every year found",
    )
    args = parser.parse_args()

    os.makedirs(DST_DIR, exist_ok=True)

    pattern = os.path.join(SRC_DIR, "b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr")
    all_src = sorted(glob.glob(pattern))

    if not all_src:
        raise FileNotFoundError(f"No zarr files found matching: {pattern}")

    if args.year == "all":
        targets = all_src
    else:
        targets = [p for p in all_src if f"_{args.year}_" in os.path.basename(p)]
        if not targets:
            raise ValueError(f"No zarr file found for year {args.year}")

    for src_path in targets:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(DST_DIR, fname)
        rechunk_one(src_path, dst_path)


if __name__ == "__main__":
    main()
