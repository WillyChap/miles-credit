"""Compare reference translator output to actual ERA5 37-level fields.

This is Phase A2: take a recent CAM6 h1, translate it to GraphCast format
with `ReferenceTranslator`, and diff against ERA5 for the same UTC stamp.

What we report:
  - per-variable global RMSE, bias, max abs error
  - per-level RMSE for atmospheric vars
  - terrain-stratified errors (Himalayas, Andes, Rockies, Greenland, Antarctica,
    high-pressure lowlands) to expose below-ground extrapolation bias separately
    from above-ground drift
  - same metrics restricted to "above-ground only" so we can isolate the
    legitimate atmospheric difference (CAM6 weather is not ERA5 weather --
    they will differ even with a perfect translator) from any bug.

Run:
    python -m regrid.validate_against_era5 <CAM6_h1_path>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from . import config as cfg
from .reference_translate import ReferenceTranslator


# -----------------------------------------------------------------------------
# ERA5 file lookup
# -----------------------------------------------------------------------------

# RDA ds633 / d633000 mirror on glade.
ERA5_PL_BASE = Path(
    "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl"
)

# var -> (param-code-tag, ll025 variant, netcdf var name)
ERA5_VARS = {
    "temperature":          ("128_130_t",  "ll025sc", "T"),
    "u_component_of_wind":  ("128_131_u",  "ll025uv", "U"),
    "v_component_of_wind":  ("128_132_v",  "ll025uv", "V"),
    "specific_humidity":    ("128_133_q",  "ll025sc", "Q"),
    "vertical_velocity":    ("128_135_w",  "ll025sc", "W"),
    "geopotential":         ("128_129_z",  "ll025sc", "Z"),
}


def era5_pl_file(date: datetime, gc_var: str) -> Path:
    """Return the daily ERA5 PL file for `date` (UTC) covering `gc_var`."""
    yymm = date.strftime("%Y%m")
    yymmdd = date.strftime("%Y%m%d")
    tag, kind, _ = ERA5_VARS[gc_var]
    fn = f"e5.oper.an.pl.{tag}.{kind}.{yymmdd}00_{yymmdd}23.nc"
    return ERA5_PL_BASE / yymm / fn


def load_era5_pl(date: datetime, gc_var: str) -> np.ndarray:
    """Return ERA5 field (37, 721, 1440) for the requested UTC hour.

    Sub-daily date.hour selects the time record (ERA5 PL is hourly).
    """
    path = era5_pl_file(date, gc_var)
    if not path.exists():
        raise FileNotFoundError(path)
    _, _, vname = ERA5_VARS[gc_var]
    ds = xr.open_dataset(path)
    try:
        # ERA5 daily files are 24 hourly records; pick the matching hour.
        arr = ds[vname].isel(time=date.hour).values
    finally:
        ds.close()
    assert arr.shape == (cfg.NUM_PLEV, cfg.GC_NLAT, cfg.GC_NLON), arr.shape
    return arr.astype(np.float64)


# -----------------------------------------------------------------------------
# Terrain masks (built once per validation run)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainBox:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float        # 0..360
    lon_max: float

    def mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Boolean (nlat, nlon) mask."""
        lat_ok = (lat >= self.lat_min) & (lat <= self.lat_max)
        if self.lon_min < self.lon_max:
            lon_ok = (lon >= self.lon_min) & (lon <= self.lon_max)
        else:
            # wraps the dateline
            lon_ok = (lon >= self.lon_min) | (lon <= self.lon_max)
        return lat_ok[:, None] & lon_ok[None, :]


TERRAIN_BOXES = (
    TerrainBox("himalayas",     25.0,  40.0,   70.0,  100.0),
    TerrainBox("andes",        -45.0, -15.0,  280.0,  300.0),
    TerrainBox("rockies",       35.0,  50.0,  235.0,  255.0),
    TerrainBox("greenland",     60.0,  82.0,  300.0,  345.0),
    TerrainBox("antarctica",   -90.0, -65.0,    0.0,  360.0),
    TerrainBox("amazon_lowland", -10.0,   5.0,  290.0,  315.0),  # high PS
    TerrainBox("global",       -90.0,  90.0,    0.0,  360.0),
)


# -----------------------------------------------------------------------------
# Metric routines
# -----------------------------------------------------------------------------


def bias_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """Return (bias, RMSE, max_abs_error) of (a - b)."""
    diff = a - b
    if mask is not None:
        diff = diff[..., mask]
    return (
        float(np.mean(diff)),
        float(np.sqrt(np.mean(diff * diff))),
        float(np.max(np.abs(diff))),
    )


def per_level_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """RMSE at each pressure level."""
    diff = a - b
    if mask is not None:
        diff = diff[:, mask]
        return np.sqrt((diff * diff).mean(axis=1))
    return np.sqrt((diff * diff).reshape(diff.shape[0], -1).mean(axis=1))


# -----------------------------------------------------------------------------
# Above-ground vs below-ground separation
# -----------------------------------------------------------------------------


def make_above_ground_mask(ps_cam_on_gc_grid: np.ndarray) -> np.ndarray:
    """Boolean (nlev_gc, nlat_gc, nlon_gc): True where p_gc < ps (above ground)."""
    p_gc = np.asarray(cfg.PRESSURE_LEVELS_PA, dtype=np.float64).reshape(-1, 1, 1)
    return p_gc < ps_cam_on_gc_grid[None, :, :]


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def datetime_from_h1_name(h1_path: Path) -> datetime:
    """Return the simulation time of a CAM h1 file.

    Tries the filename first (the canonical CAM naming convention encodes
    the time as `.cam.h1.YYYY-MM-DD-SSSSS.nc`). Falls back to reading the
    `date` (YYYYMMDD) and `datesec` variables from the file -- which is
    what we use for files that have been renamed for archival or test
    reasons (e.g., the persistent reference_h1_*.nc pair).
    """
    stem = h1_path.name
    # Standard pattern: ".cam.h1.1980-01-06-00000.nc"
    if ".cam.h1." in stem:
        try:
            parts = stem.split(".cam.h1.")[1].rstrip(".nc")
            date_part, sec_part = parts.rsplit("-", 1)
            secs = int(sec_part)
            hh = secs // 3600
            mm = (secs % 3600) // 60
            return datetime.strptime(
                f"{date_part} {hh:02d}:{mm:02d}", "%Y-%m-%d %H:%M",
            )
        except (IndexError, ValueError):
            pass

    # Fallback: read `date` (YYYYMMDD) + `datesec` (seconds-of-day) from the file.
    ds = xr.open_dataset(h1_path, decode_times=False)
    try:
        ymd = int(ds["date"].isel(time=0).values)
        sec = int(ds["datesec"].isel(time=0).values)
    finally:
        ds.close()
    y, rest = divmod(ymd, 10000)
    m, d = divmod(rest, 100)
    hh = sec // 3600
    mm = (sec % 3600) // 60
    return datetime(y, m, d, hh, mm)


def validate(h1_path: Path, vars_to_check: tuple[str, ...] | None = None) -> None:
    when = datetime_from_h1_name(h1_path)
    print(f"\n== CAM6 h1 -> ERA5 comparison ==")
    print(f"   h1 file: {h1_path.name}")
    print(f"   UTC    : {when.isoformat()}")

    print("\nBuilding translator (cached stencil)...")
    tr = ReferenceTranslator.build(h1_path)
    print("Translating CAM6 h1 -> GraphCast format...")
    ds_cam = tr.translate_h1_file(
        h1_path,
        time_index=0,
        which=("T", "U", "V", "Q", "Z3", "OMEGA"),
    )

    # PS on GC grid for the above-ground / below-ground split.
    h1 = xr.open_dataset(h1_path)
    try:
        ps_cam = h1["PS"].isel(time=0).values.astype(np.float64)
    finally:
        h1.close()
    from .horizontal_weights import apply_bilinear
    ps_gc = apply_bilinear(ps_cam, tr.cam_to_gc)
    above_gc = make_above_ground_mask(ps_gc)

    if vars_to_check is None:
        vars_to_check = tuple(ds_cam.data_vars)
    print(f"   Variables: {vars_to_check}\n")

    gc_grid = tr.gc_grid

    # Load each ERA5 var, compute metrics globally + per terrain box.
    for gv in vars_to_check:
        if gv not in ds_cam:
            print(f"  ... skipping {gv} (not in translated ds)")
            continue
        try:
            era5 = load_era5_pl(when, gv)
        except FileNotFoundError as exc:
            print(f"  ... ERA5 missing for {gv}: {exc}")
            continue

        cam = ds_cam[gv].values
        print(f"  {gv}")
        # Global, all levels, all points
        b, r, m = bias_rmse(cam, era5)
        print(f"     global all-points:   bias={b:+.3e}  rmse={r:.3e}  max|err|={m:.3e}")
        # Global, above-ground only
        b, r, m = bias_rmse(cam, era5, mask=None)  # placeholder, masked variant below
        diff = cam - era5
        b_ag = float(diff[above_gc].mean())
        r_ag = float(np.sqrt((diff[above_gc] ** 2).mean()))
        m_ag = float(np.abs(diff[above_gc]).max())
        print(f"     global above-ground: bias={b_ag:+.3e}  rmse={r_ag:.3e}  max|err|={m_ag:.3e}")
        # Below-ground only
        below = ~above_gc
        if below.any():
            b_bg = float(diff[below].mean())
            r_bg = float(np.sqrt((diff[below] ** 2).mean()))
            m_bg = float(np.abs(diff[below]).max())
            print(f"     global below-ground: bias={b_bg:+.3e}  rmse={r_bg:.3e}  max|err|={m_bg:.3e}")

        # Per terrain box (above ground only -- isolates regional weather bias
        # from extrapolation bias).
        for box in TERRAIN_BOXES[:-1]:  # skip "global", already shown
            m2d = box.mask(gc_grid.lat, gc_grid.lon)
            m3d = m2d[None, :, :] & above_gc
            if not m3d.any():
                continue
            d = diff[m3d]
            print(f"     {box.name:>16s} (above): "
                  f"bias={d.mean():+.3e}  rmse={np.sqrt((d*d).mean()):.3e}  "
                  f"max|err|={np.abs(d).max():.3e}")

        # Per-level RMSE (above ground only), printed compactly.
        per_lvl = []
        for li in range(cam.shape[0]):
            sel = above_gc[li]
            if sel.any():
                d = diff[li][sel]
                per_lvl.append(np.sqrt((d * d).mean()))
            else:
                per_lvl.append(np.nan)
        print(f"     per-level RMSE (above-ground): " +
              " ".join(f"{cfg.PRESSURE_LEVELS_HPA[i]}={per_lvl[i]:.2g}"
                       for i in (0, 10, 20, 25, 30, 33, 36)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h1_path", type=Path, help="CAM6 h1 file to translate and compare")
    ap.add_argument("--vars", nargs="+", default=None, help="GraphCast var names to validate (default: all atmospheric)")
    args = ap.parse_args()
    validate(args.h1_path, vars_to_check=tuple(args.vars) if args.vars else None)


if __name__ == "__main__":
    main()
