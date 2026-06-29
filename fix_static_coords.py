"""
fix_static_coords.py
--------------------
Re-save static, mean, and std netCDF files with float32 latitude/longitude
coordinates that EXACTLY match those in the b_credit_runs_f32_02 zarr.

The zarr stores lat/lon as float32; the STAGING static/mean/std files store
them as float64.  xarray treats float32(-89.057594) and float64(-89.05759162)
as DIFFERENT coordinate values and produces a union instead of an alignment,
inflating both the channel and latitude dimensions in the resulting tensor.

Usage (on interactive node with credit-casper-mar2026 activated):
    python fix_static_coords.py

Outputs
-------
  /glade/derecho/scratch/wchapman/b_credit_runs/
      statics_b_credit_runs_f32_02.nc
      mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal_f32coords.nc
      std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal_f32coords.nc

Update camulator_config_new.yml to point to these new files:
    save_loc_static:  ...statics_b_credit_runs_f32_02.nc
    save_loc_physics: ...statics_b_credit_runs_f32_02.nc
    mean_path: ...mean_..._f32coords.nc
    std_path:  ...std_..._f32coords.nc
"""

import glob
import numpy as np
import xarray as xr

# ── reference zarr (has the float32 coords we want to match) ─────────────────
ZARR_GLOB = (
    "/glade/derecho/scratch/wchapman/b_credit_runs_f32_02/"
    "b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr"
)
zarr_paths = sorted(glob.glob(ZARR_GLOB))
assert zarr_paths, f"No zarr found at: {ZARR_GLOB}"
zarr_path = zarr_paths[0]
print(f"Reference zarr: {zarr_path}")

ds_ref = xr.open_zarr(zarr_path)
# find whichever lat/lon coord name the zarr uses
lat_name = next(c for c in ds_ref.coords if c in ("lat", "latitude"))
lon_name = next(c for c in ds_ref.coords if c in ("lon", "longitude"))
print(f"  zarr lat coord: '{lat_name}' dtype={ds_ref[lat_name].dtype}, "
      f"lon coord: '{lon_name}' dtype={ds_ref[lon_name].dtype}")
print(f"  zarr lat[:3]: {ds_ref[lat_name].values[:3]}")
ref_lat = ds_ref[lat_name].values   # float32 array
ref_lon = ds_ref[lon_name].values   # float32 array

# collect ALL numeric 1-D coords from the zarr for matching
ref_coords = {}
for c in ds_ref.coords:
    v = ds_ref[c].values
    if v.ndim == 1 and np.issubdtype(v.dtype, np.number):
        ref_coords[c] = v
print(f"  zarr numeric 1-D coords: {list(ref_coords.keys())}")

# ── helper ───────────────────────────────────────────────────────────────────

OUT_DIR = "/glade/derecho/scratch/wchapman/b_credit_runs"


def _fix_coords(ds: xr.Dataset, ref_coords: dict) -> xr.Dataset:
    """
    Replace every numeric 1-D coordinate in ds whose values are close to a
    reference coordinate (from the zarr) with the zarr's float32 values.
    This covers lat, lon, lev, ilev — anything that might cause an .equals()
    failure due to float32 vs float64 precision differences.
    """
    coords_new = {}
    for c in list(ds.coords):
        v = ds.coords[c].values
        if v.ndim == 0 or not np.issubdtype(v.dtype, np.number):
            continue
        # find a matching ref coord by shape + allclose
        for ref_name, ref_v in ref_coords.items():
            if v.shape == ref_v.shape and np.allclose(
                v.astype(np.float64), ref_v.astype(np.float64), rtol=1e-3, atol=1e-3
            ):
                coords_new[c] = ref_v
                print(f"  replacing '{c}' ({v.dtype}) with zarr '{ref_name}' ({ref_v.dtype})")
                break
    if coords_new:
        ds = ds.assign_coords(coords_new)
    return ds


# ── 1. Static file ────────────────────────────────────────────────────────────
STATIC_IN = (
    "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/"
    "b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
)
STATIC_OUT = f"{OUT_DIR}/statics_b_credit_runs_f32_02.nc"

print(f"\nLoading static: {STATIC_IN}")
ds_st = xr.open_dataset(STATIC_IN)
print(f"  vars: {list(ds_st.data_vars)}")
print(f"  dims: {dict(ds_st.dims)}")
ds_st_fixed = _fix_coords(ds_st, ref_coords)
ds_st_fixed.to_netcdf(STATIC_OUT)
print(f"  Saved → {STATIC_OUT}")

# ── 2. Mean file ──────────────────────────────────────────────────────────────
MEAN_IN = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc"
)
MEAN_OUT = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal_f32coords.nc"
)

print(f"\nLoading mean: {MEAN_IN}")
ds_mean = xr.open_dataset(MEAN_IN)
print(f"  dims: {dict(ds_mean.dims)}")
ds_mean_fixed = _fix_coords(ds_mean, ref_coords)
ds_mean_fixed.to_netcdf(MEAN_OUT)
print(f"  Saved → {MEAN_OUT}")

# ── 3. Std file ───────────────────────────────────────────────────────────────
STD_IN = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc"
)
STD_OUT = (
    "/glade/derecho/scratch/wchapman/b_credit_runs/"
    "std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal_f32coords.nc"
)

print(f"\nLoading std: {STD_IN}")
ds_std = xr.open_dataset(STD_IN)
print(f"  dims: {dict(ds_std.dims)}")
ds_std_fixed = _fix_coords(ds_std, ref_coords)
ds_std_fixed.to_netcdf(STD_OUT)
print(f"  Saved → {STD_OUT}")

# ── 4. Cyclic forcing file ───────────────────────────────────────────────────
FORCING_IN  = "/glade/derecho/scratch/wchapman/CAMULATOR_FORCING/b.e21.CREDIT_climate_cyclic_1yr.nc"
FORCING_OUT = "/glade/derecho/scratch/wchapman/CAMULATOR_FORCING/b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc"

print(f"\nLoading cyclic forcing: {FORCING_IN}")
ds_forcing = xr.open_dataset(FORCING_IN, chunks={"time": 32})
print(f"  dims: {dict(ds_forcing.dims)}")
ds_forcing_fixed = _fix_coords(ds_forcing.load(), ref_coords)
ds_forcing_fixed.to_netcdf(FORCING_OUT)
print(f"  Saved → {FORCING_OUT}")

# ── 5. Quick verification ─────────────────────────────────────────────────────
print("\n── Verification ──────────────────────────────────────────────────────")
for path, label in [(STATIC_OUT, "static"), (MEAN_OUT, "mean"), (STD_OUT, "std"), (FORCING_OUT, "forcing")]:
    ds = xr.open_dataset(path)
    for c in list(ds.coords):
        v = ds.coords[c].values
        if v.ndim == 0 or not np.issubdtype(v.dtype, np.number):
            continue
        for ref_name, ref_v in ref_coords.items():
            if v.shape == ref_v.shape:
                ok = np.array_equal(v, ref_v)
                print(f"  {label} '{c}': dtype={v.dtype}, exact_match_with_zarr_{ref_name}={ok}")
                break

print("""
Done.  Update camulator_config_new.yml:

    save_loc_static:  /glade/derecho/scratch/wchapman/b_credit_runs/statics_b_credit_runs_f32_02.nc
    save_loc_physics: /glade/derecho/scratch/wchapman/b_credit_runs/statics_b_credit_runs_f32_02.nc
    mean_path: /glade/derecho/scratch/wchapman/b_credit_runs/mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal_f32coords.nc
    std_path:  /glade/derecho/scratch/wchapman/b_credit_runs/std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal_f32coords.nc
""")
