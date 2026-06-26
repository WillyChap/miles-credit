"""
netcdf_to_zarr.py
-----------------
Consolidate per-step CAMulator NetCDF output (one file per 6-hourly step,
`pred_<init>_<hour>.nc`) into per-year zarr stores that match the CREDIT
training-data layout:

    <out_dir>/<prefix>_<year>.zarr     # 1460 steps/yr, dims [time, level, lat, lon]
                                       # 3D vars chunked time=1, float32

Robustness:
  * Files are ordered by the integer forecast-hour parsed from the name, so
    it works whether or not the names are zero-padded (the early runs were
    not; `:06d` runs are). The actual zarr `time` axis is taken from each
    file's own `time` coordinate -- never from the filename.
  * Year bucketing is computed from the init time + forecast hour (no file
    reads needed for grouping), so a 35-yr / 51,100-file run is processed one
    cheap year at a time instead of opening everything at once.
  * Resumable: a year whose .zarr already exists is skipped unless --overwrite.

Example:
    python netcdf_to_zarr.py \
        --input_dir /glade/derecho/scratch/wchapman/CREDIT/AR_study_1980_2014/chkpt058_6hr_1980_2014/1980-01-01T00Z \
        --out_dir   /glade/derecho/scratch/wchapman/CREDIT/AR_study_1980_2014/zarr \
        --prefix    camulator_chkpt058_AR
"""

import os
import re
import glob
import shutil
import argparse
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

# 3D vars get one time slice per chunk (matches training data); coords are kept whole.
CHUNK_3D = {"time": 1, "level": -1, "ilev": -1, "latitude": -1, "longitude": -1}

_HOUR_RE = re.compile(r"_(\d+)\.nc$")

# CAM reference pressure (Pa). hyai is already in Pa here, but P0 is carried for
# the dimensionless midpoint coeffs: p_mid = hyam*P0 + hybm*PS.
P0_PA = 100000.0

# Default statics file holding the hybrid-sigma coefficients (same one the
# conservation post-blocks use). These are NOT in the per-step model output, so
# we attach them to each year's zarr to match the training data and to make the
# store self-sufficient for vertical integration (e.g. IVT):
#   p_interface = hyai + hybi*PS         (Pa; hyai already in Pa)
#   dp_k        = diff(p_interface)      (per-layer thickness for the q*u integral)
DEFAULT_STATICS = (
    "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/"
    "b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
)


def load_vertical_coefs(statics_path):
    """Return {name: (dim, np.float32 values)} for hyai/hybi/hyam/hybm (+ scalar P0)."""
    from credit.data import get_forward_data

    sd = get_forward_data(statics_path)
    coefs = {
        "hyai": ("ilev", np.asarray(sd["hyai"].values, dtype="float32")),   # Pa
        "hybi": ("ilev", np.asarray(sd["hybi"].values, dtype="float32")),   # dimensionless
        "hyam": ("level", np.asarray(sd["hyam"].values, dtype="float32")),  # dimensionless
        "hybm": ("level", np.asarray(sd["hybm"].values, dtype="float32")),  # dimensionless
    }
    return coefs


def parse_init_dt(input_dir: str) -> datetime:
    """Init datetime from the folder name, e.g. '1980-01-01T00Z'."""
    base = os.path.basename(os.path.normpath(input_dir))
    return datetime.strptime(base, "%Y-%m-%dT%HZ")


def hour_of(path: str) -> int:
    """Integer forecast hour parsed from 'pred_<init>_<hour>.nc'."""
    m = _HOUR_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse forecast hour from {path!r}")
    return int(m.group(1))


def year_of(init_dt: datetime, hour: int) -> int:
    """Valid-time year for a file. time = init + (hour - lead) ; lead=6 (first step==init)."""
    return (init_dt + timedelta(hours=hour - 6)).year


def attach_vertical_coefs(ds, coefs):
    """Add hyai/hybi/hyam/hybm + P0 to a dataset, matching available dims."""
    if coefs is None:
        return ds
    for name, (dim, vals) in coefs.items():
        if dim in ds.dims and ds.sizes[dim] == vals.size:
            ds[name] = (dim, vals)
    ds["P0"] = np.float32(P0_PA)
    return ds


def year_is_complete(store, n_time):
    """True iff `store` exists and is FULLY written.

    Existence / .zmetadata are NOT reliable (xarray writes consolidated metadata
    before all chunk data lands, and a killed job leaves a partial store). The
    authoritative check: every time-dimensioned data variable must have all its
    per-timestep chunk files on disk (time chunk = 1 -> exactly n_time chunks),
    and the store must carry the no-leap time axis of the right length.
    """
    if not os.path.isdir(store):
        return False
    try:
        ds = xr.open_zarr(store)
    except Exception:
        return False
    try:
        if ds.sizes.get("time") != n_time:
            return False
        for v in ds.data_vars:
            if "time" not in ds[v].dims:
                continue  # coefficients (hyai/hybi/...) are time-independent
            vdir = os.path.join(store, v)
            if not os.path.isdir(vdir):
                return False
            n_chunks = sum(1 for f in os.listdir(vdir) if not f.startswith("."))
            if n_chunks != n_time:  # missing per-timestep chunks -> partial write
                return False
        return True
    finally:
        ds.close()


def consolidate(input_dir, out_dir, prefix, chunk_time, overwrite, years_filter, statics_path, lazy):
    init_dt = parse_init_dt(input_dir)
    files = glob.glob(os.path.join(input_dir, "pred_*.nc"))
    if not files:
        raise SystemExit(f"No pred_*.nc files found in {input_dir}")

    # Build the authoritative NO-LEAP (365-day) time axis and relabel onto it.
    # The model stepped through forcing index i to produce output step i, and the run
    # started at forcing index 0, so step i == forcing time[i] == cftime-noleap(init + i*6h).
    # The raw files were stamped with a real (leap-aware) datetime clock instead, which
    # inserts Feb 29 and drifts the dates; we replace that with the no-leap axis so the
    # zarr matches the CESM / training-data calendar exactly (1460 steps/yr, no Feb 29).
    recs = sorted((hour_of(f), f) for f in files)  # chronological by step
    # size the axis to the largest step index so partial/sparse sets still map correctly
    max_si = max(h // 6 - 1 for h, _ in recs)
    noleap = xr.cftime_range(
        start=f"{init_dt:%Y-%m-%d %H:%M:%S}", periods=max_si + 1, freq="6h", calendar="noleap"
    )
    by_year = {}
    for h, f in recs:
        si = h // 6 - 1  # step index: hour 6 -> step 0
        t = noleap[si]
        by_year.setdefault(t.year, []).append((f, t))

    os.makedirs(out_dir, exist_ok=True)

    coefs = None
    if statics_path:
        if os.path.exists(statics_path):
            coefs = load_vertical_coefs(statics_path)
            print(f"Vertical coefs: hyai/hybi/hyam/hybm + P0 from {os.path.basename(statics_path)}")
        else:
            print(f"WARNING: statics file not found ({statics_path}); skipping hybrid coeffs")

    print(f"Init time     : {init_dt:%Y-%m-%dT%HZ}")
    print(f"Files found   : {len(files)}")
    print(f"Years present : {min(by_year)}-{max(by_year)}  ({len(by_year)} stores)")

    CHUNK_3D["time"] = chunk_time
    for yr in sorted(by_year):
        if years_filter and yr not in years_filter:
            continue
        items = by_year[yr]  # [(file, noleap_time)], step-ordered
        yfiles = [f for f, _t in items]
        ytimes = [t for _f, t in items]
        n_time = len(yfiles)
        store = os.path.join(out_dir, f"{prefix}_{yr}.zarr")

        # Resume safely: skip only stores verified FULL; a partial store (e.g. from a
        # killed/timed-out job) fails the check and is overwritten below.
        if not overwrite and year_is_complete(store, n_time):
            print(f"  {yr}: SKIP (verified complete, {n_time} steps)")
            continue
        if os.path.isdir(store):
            print(f"  {yr}: incomplete store found -> rewriting")

        print(f"  {yr}: {n_time} steps -> {store}", flush=True)
        ds = xr.open_mfdataset(
            yfiles,
            combine="nested",
            concat_dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            parallel=False,
            engine="netcdf4",
        )
        ds = ds.drop_vars("forecast_hour", errors="ignore")
        # Relabel the time axis to the no-leap calendar (files are in step order,
        # so the override aligns positionally with the data).
        ds = ds.assign_coords(time=("time", xr.CFTimeIndex(ytimes)))
        ds = attach_vertical_coefs(ds, coefs)  # hybrid-sigma coeffs for vertical integration
        # Read the whole year into memory ONCE, so each source file is opened a single
        # time instead of being re-read per output variable (much less I/O on the 2.9 TB).
        if not lazy:
            ds = ds.load()
        chunks = {k: v for k, v in CHUNK_3D.items() if k in ds.dims}
        ds = ds.chunk(chunks)
        # Write to a temp store then atomically rename, so an interrupted write never
        # leaves a half-finished store at the final path that looks "done".
        tmp_store = store + ".tmp"
        if os.path.isdir(tmp_store):
            shutil.rmtree(tmp_store)
        ds.to_zarr(tmp_store, mode="w", consolidated=True)
        nt = ds.sizes.get("time")
        ds.close()
        if os.path.isdir(store):
            shutil.rmtree(store)
        os.rename(tmp_store, store)
        print(f"  {yr}: done  ({nt} steps)", flush=True)

    print("All requested years consolidated.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="Folder of pred_*.nc (the <init>T..Z directory)")
    ap.add_argument("--out_dir", required=True, help="Where to write <prefix>_<year>.zarr stores")
    ap.add_argument("--prefix", default="camulator_AR", help="Output zarr filename prefix")
    ap.add_argument("--chunk_time", type=int, default=1, help="time chunk size for 3D vars (default 1, like training)")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite year stores that already exist")
    ap.add_argument("--years", type=int, nargs="+", default=None, help="Only process these calendar years")
    ap.add_argument("--statics", default=DEFAULT_STATICS,
                    help="Statics file with hyai/hybi/hyam/hybm (set to '' to skip attaching them)")
    ap.add_argument("--lazy", action="store_true",
                    help="Stream from disk instead of loading each year into RAM "
                         "(lower memory, but re-reads source files per variable)")
    args = ap.parse_args()
    consolidate(args.input_dir, args.out_dir, args.prefix, args.chunk_time, args.overwrite,
                set(args.years) if args.years else None, args.statics or None, args.lazy)


if __name__ == "__main__":
    main()
