"""Assemble a 3-timestep GraphCast input bundle from two CAM6 h1 files.

GraphCast's `extract_inputs_targets_forcings(...)` (see graphcast_demo.ipynb)
asserts `example_batch.dims["time"] >= 3`: 2 input frames (T-6h, T) plus
>=1 target frame (T+6h) that GraphCast fills via prediction.

For inference, the target frame is a structural placeholder -- it carries
the right shape and a `datetime` coord but the values can be NaN. They will
be overwritten by the predicted state.

The bundle therefore contains:

  - time dim of length 3 (T-6h, T, T+6h)
  - level dim of length 37 for atmospheric vars
  - lat, lon dims for the 0.25 deg grid
  - `datetime` coordinate on (batch, time) (used to compute TISR + sin/cos
    day/year forcings; GraphCast adds these itself if absent)
  - atmospheric vars shaped (batch, time, level, lat, lon)
  - surface vars shaped (batch, time, lat, lon)
  - static vars shaped (lat, lon)   -- NO time dim, NO batch
  - a `batch` dim, size 1

GraphCast computes the forcing vars from `datetime` -- we do not write them.

The "time" coordinate is in `pd.Timedelta`s relative to the forecast
reference (final input timestep T). For 12h input duration with 6h steps,
the three timedeltas are `-6h`, `0h`, `+6h`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from . import config as cfg
from .reference_translate import ReferenceTranslator
from .surface_and_forcing import (
    translate_static_from_h1,
    translate_surface_from_h1,
)
from .validate_against_era5 import datetime_from_h1_name


# Atmospheric vars we always translate.
_ATMO = ("T", "U", "V", "Q", "Z3", "OMEGA")


def build_graphcast_input_bundle(
    h1_paths: tuple[Path, Path],
    translator: ReferenceTranslator | None = None,
    static_h1: Path | None = None,
    batch_size: int = 1,
) -> xr.Dataset:
    """Build a GraphCast-ready (batch, time=2, ...) input Dataset.

    Parameters
    ----------
    h1_paths : (Path, Path)
        Two CAM6 h1 files at T-6h and T (in order). Atmospheric + surface
        vars are read from both.
    translator : ReferenceTranslator, optional
        Reuse a prebuilt translator (its stencil + hybrid coord). If None,
        we build one from h1_paths[0].
    static_h1 : Path, optional
        Source for PHIS / LANDFRAC (which are time-invariant per case).
        Defaults to h1_paths[0].
    batch_size : int
        Currently only 1 is supported; included for forward compatibility.

    Returns
    -------
    xarray.Dataset
        Ready for `extract_inputs_targets_forcings(..., input_duration='12h', ...)`.
    """
    if batch_size != 1:
        raise NotImplementedError("batch_size > 1 not implemented")

    h1_minus6, h1_t = (Path(p) for p in h1_paths)
    if translator is None:
        translator = ReferenceTranslator.build(h1_minus6)
    if static_h1 is None:
        static_h1 = h1_minus6

    # Absolute datetimes for the two real inputs (T-6h, T) and the placeholder
    # target slot (T+6h). The target time is what GraphCast will fill.
    dt_minus6 = pd.Timestamp(datetime_from_h1_name(h1_minus6))
    dt_t = pd.Timestamp(datetime_from_h1_name(h1_t))
    dt_plus6 = dt_t + pd.Timedelta("6h")
    assert dt_t - dt_minus6 == pd.Timedelta("6h"), (
        f"h1 spacing must be 6h, got {dt_t - dt_minus6}"
    )

    # Lead-time coordinate (final input -> 0; previous -> -6h; target -> +6h).
    time_coord = pd.TimedeltaIndex(["-6h", "0h", "6h"], name="time")
    datetime_coord = np.array([dt_minus6, dt_t, dt_plus6], dtype="datetime64[ns]")
    n_time = 3

    # ---- Atmospheric vars (time=3, level=37, lat, lon) ----
    # Real values at slots 0 and 1; NaN at slot 2 (target placeholder).
    atmo_stacks: dict[str, np.ndarray] = {}
    for ti, h1 in enumerate((h1_minus6, h1_t)):
        ds_atmo = translator.translate_h1_file(h1, time_index=0, which=_ATMO)
        for gv, da in ds_atmo.data_vars.items():
            arr = atmo_stacks.setdefault(
                gv,
                np.full((n_time, cfg.NUM_PLEV, cfg.GC_NLAT, cfg.GC_NLON),
                        np.nan, dtype=np.float64),
            )
            arr[ti] = da.values

    # ---- Surface vars (time=3, lat, lon) ----
    surf_stacks: dict[str, np.ndarray] = {}
    for ti, h1 in enumerate((h1_minus6, h1_t)):
        surf = translate_surface_from_h1(h1, translator.cam_to_gc, time_index=0)
        for gv, arr2d in surf.items():
            arr = surf_stacks.setdefault(
                gv,
                np.full((n_time, cfg.GC_NLAT, cfg.GC_NLON), np.nan, dtype=np.float64),
            )
            arr[ti] = arr2d

    # ---- Static vars (lat, lon) ----
    statics = translate_static_from_h1(static_h1, translator.cam_to_gc, time_index=0)

    # ---- Coordinates ----
    coords = {
        "time": ("time", time_coord),
        "datetime": (("batch", "time"), datetime_coord[None, :]),
        "level": ("level", np.array(cfg.PRESSURE_LEVELS_HPA, dtype=np.int32)),
        "lat": ("lat", translator.gc_grid.lat),
        "lon": ("lon", translator.gc_grid.lon),
    }

    # ---- Build Dataset ----
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    # batch dim: prepend size-1 axis.
    for name, arr in atmo_stacks.items():
        data_vars[name] = (("batch", "time", "level", "lat", "lon"), arr[None, ...])
    for name, arr in surf_stacks.items():
        data_vars[name] = (("batch", "time", "lat", "lon"), arr[None, ...])
    for name, arr in statics.items():
        # statics carry no batch dim per GraphCast convention.
        data_vars[name] = (("lat", "lon"), arr)

    attrs = {
        "translator": "regrid.reference_translate.ReferenceTranslator",
        "source_h1_t_minus_6h": str(h1_minus6),
        "source_h1_t":          str(h1_t),
        "source_static":        str(static_h1),
        "cam_grid":             translator.cam_grid.name,
        "gc_grid":              translator.gc_grid.name,
        "input_duration":       "12h",
    }
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def consecutive_h1_pairs(h1_dir: Path, pattern: str = "*.cam.h1.*.nc") -> Iterable[tuple[Path, Path]]:
    """Yield consecutive (T-6h, T) h1 file pairs, sorted by filename."""
    h1_files = sorted(Path(h1_dir).glob(pattern))
    for a, b in zip(h1_files, h1_files[1:]):
        yield a, b


__all__ = [
    "build_graphcast_input_bundle",
    "consecutive_h1_pairs",
]
