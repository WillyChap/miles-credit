"""Reverse translator: GraphCast prediction -> CAM6 hybrid nudge target.

Forward direction (already done in reference_translate.py):
  CAM6 hybrid (32 lev, 192x288 FV) -> GraphCast (37 plev, 721x1440 ERA5)

This module does the inverse for the four nudged prognostic fields:
  GraphCast (37 plev, 721x1440) -> CAM6 hybrid (32 lev, 192x288)
  for {T, U, V, Q}

Output is a netCDF file matching the existing SUMO nudge file format
(sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc), produced atomically via copy-template +
rename. The CAM6 nudging toolbox reads these files at each 6h coupling step
and applies the standard Newtonian relaxation
  dU/dt = Ucoef * window * (Target - Model) / Utau
(with Nudge_Do=0, see user_nl_cam).

Edge cases handled:

  - GraphCast top is 1 hPa; CAM6 top is 3.64 hPa. So CAM6's top model level
    (and all interior levels) sit inside the GraphCast pressure range -- no
    above-top extrapolation is ever needed.
  - In high-PS storm centers (PS > 1000 hPa, e.g. the Aleutian Low at 1040
    hPa), CAM6's lowest hybrid level can exceed GraphCast's 1000 hPa bottom
    by ~40 hPa. We "hold lowest" -- repeat the GC 1000 hPa value -- which
    matches the same convention we use forward for below-ground filler.

Surface pressure policy:

  PS is NOT written. CAM6 PS is dynamically determined by the FV dycore
  from mass conservation, and naive replacement with GraphCast MSLP would
  break that balance. We only nudge U, V, T, Q. The SUMO_CAM6_v03 case
  has `Nudge_PSprof=0` for exactly this reason.

The PS field used to compute CAM6 hybrid pressures at the target time is
CAM6's current PS (i.e., PS at coupling time T, not T+6h). CAM6 PS changes
~1-2% over 6h, which is well below the variability we're nudging against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import os
import tempfile

import numpy as np
import xarray as xr

from . import config as cfg
from .grids import (
    HybridCoord,
    LatLonGrid,
    cam6_grid_from_h1,
    graphcast_grid,
    hybrid_coord_from_h1,
)
from .horizontal_weights import (
    BilinearStencil,
    apply_bilinear,
    build_bilinear_stencil,
)
from .hybrid_pressure import midpoint_pressures_np
from .vertical_interp import interp_log_p_to_hybrid


# GraphCast var name -> CAM6 var name (only the four nudged fields).
GC_TO_CAM_ATMO = {
    "temperature":         "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "specific_humidity":   "Q",
}


@dataclass
class ReverseTranslator:
    """Holds the precomputed GC->CAM6 stencil + hybrid coord; reused per call."""

    hybrid: HybridCoord
    cam_grid: LatLonGrid
    gc_grid: LatLonGrid
    gc_to_cam: BilinearStencil

    @classmethod
    def build(cls, ref_h1_path: str | Path) -> "ReverseTranslator":
        cam = cam6_grid_from_h1(ref_h1_path)
        hyb = hybrid_coord_from_h1(ref_h1_path)
        gc = graphcast_grid()
        gc_to_cam = build_bilinear_stencil(gc, cam)
        return cls(hybrid=hyb, cam_grid=cam, gc_grid=gc, gc_to_cam=gc_to_cam)

    # ------------------------------------------------------------------ #
    # Single-variable translation                                         #
    # ------------------------------------------------------------------ #

    def translate_one_var(
        self,
        gc_field_plev: np.ndarray,    # (37, 721, 1440)
        ps_cam: np.ndarray,           # (192, 288) Pa
        clip_nonneg: bool = False,    # True for Q
    ) -> np.ndarray:
        """Translate one atmospheric var: 37 plev x GC grid -> 32 hybrid x CAM grid."""
        assert gc_field_plev.shape == (cfg.NUM_PLEV, cfg.GC_NLAT, cfg.GC_NLON), gc_field_plev.shape
        assert ps_cam.shape == (cfg.CAM6_NLAT, cfg.CAM6_NLON), ps_cam.shape

        # 1. Horizontal regrid GC -> CAM6, level by level (apply_bilinear handles
        #    the leading level axis automatically).
        field_cam_plev = apply_bilinear(gc_field_plev.astype(np.float64), self.gc_to_cam)
        # shape: (37, 192, 288)

        # 2. Build CAM6 hybrid midpoint pressures from CAM6 PS.
        p_cam_hybrid = midpoint_pressures_np(self.hybrid.hyam, self.hybrid.hybm, ps_cam)
        # shape: (32, 192, 288), Pa, strictly increasing in level (positive=down).

        # 3. Vertical interp from fixed pressure levels to per-column hybrid.
        vi = interp_log_p_to_hybrid(
            field_plev=field_cam_plev,
            p_src_plev=np.asarray(cfg.PRESSURE_LEVELS_PA, dtype=np.float64),
            p_tgt_hybrid=p_cam_hybrid,
        )
        out = vi.field   # (32, 192, 288)

        # 4. Below-ground (CAM6 hybrid p > GC max p): hold the lowest GC value.
        #    This only triggers for PS > 1000 hPa columns at their lowest hybrid
        #    levels. above_top will never trigger (CAM6 top > GC top in pressure).
        if vi.below_ground.any():
            gc_bottom = field_cam_plev[-1:, ...]   # (1, 192, 288), 1000 hPa value
            out = np.where(vi.below_ground, gc_bottom, out)

        if clip_nonneg:
            out = np.maximum(out, 0.0)

        assert np.isfinite(out).all(), "non-finite values in reverse-translated field"
        return out

    # ------------------------------------------------------------------ #
    # Multi-variable nudge bundle                                         #
    # ------------------------------------------------------------------ #

    def translate_prediction(
        self,
        gc_predictions: xr.Dataset,
        cam_ps: np.ndarray,
        time_index: int = 0,
        batch_index: int = 0,
    ) -> dict[str, np.ndarray]:
        """Translate the four nudged atmospheric vars in a GC prediction Dataset.

        Parameters
        ----------
        gc_predictions : xr.Dataset
            Output of rollout.chunked_prediction. Expected dims include
            (batch, time, level, lat, lon).
        cam_ps : (192, 288) Pa
            CAM6 surface pressure at the COUPLING time T (used to build hybrid
            pressures for the target time T+6h; small error since PS varies
            <2% per 6h).
        time_index : int
            Which prediction step to translate. For SUMO use 0 (the +6h step).
        batch_index : int

        Returns
        -------
        dict[str, np.ndarray]
            Keys: "U", "V", "T", "Q". Each value shape (32, 192, 288), float64.
        """
        out: dict[str, np.ndarray] = {}
        for gc_name, cam_name in GC_TO_CAM_ATMO.items():
            if gc_name not in gc_predictions:
                raise KeyError(f"GC prediction missing variable: {gc_name}")
            da = gc_predictions[gc_name].isel(batch=batch_index, time=time_index)
            arr = np.asarray(da.values, dtype=np.float64)
            assert arr.shape == (cfg.NUM_PLEV, cfg.GC_NLAT, cfg.GC_NLON), (
                f"{gc_name}: expected (37,721,1440), got {arr.shape}"
            )
            out[cam_name] = self.translate_one_var(
                arr, cam_ps, clip_nonneg=(cam_name == "Q"),
            )
        return out


# ----------------------------------------------------------------------------- #
# Nudge file writer                                                              #
# ----------------------------------------------------------------------------- #


def write_nudge_file(
    nudge_targets: dict[str, np.ndarray],
    template_h1_path: str | Path,
    out_path: str | Path,
) -> None:
    """Write a SUMO-format nudge file by copying an h1 template and overwriting.

    The CAM6 nudging toolbox in this case is strict about netCDF layout (var
    dim order, fill values, time unlimited, etc.); using a real h1 as the
    template guarantees structural compatibility. We then:

      1. Copy template -> out_path.tmp
      2. Open out_path.tmp in append mode, overwrite U/V/T/Q with our targets
      3. Drop all other variables (PS, TS, SST, ICEFRAC, OMEGA, Z3, ...) so the
         file is small and unambiguous about what is being nudged
      4. os.replace(out_path.tmp, out_path) -- atomic, prevents partial reads

    Parameters
    ----------
    nudge_targets : dict
        {"U", "V", "T", "Q"} -> (32, 192, 288) arrays
    template_h1_path : Path
        Any well-formed CAM6 h1 file with U, V, T, Q in standard layout.
    out_path : Path
        Final atomic destination.
    """
    template_h1_path = Path(template_h1_path)
    out_path = Path(out_path)

    # Vars to KEEP in the output. Everything else is dropped.
    keep = {"U", "V", "T", "Q",
            "PS",                          # left as-is; CAM nudging needs it for hybrid coord recompute
            "lat", "lon", "lev", "ilev",
            "hyam", "hybm", "hyai", "hybi", "P0",
            "time", "time_bnds", "date", "datesec",
            "nbnd", "chars"}

    # Step 1+2: load template, overwrite values, drop excess.
    ds = xr.open_dataset(template_h1_path, decode_times=False)

    # Sanity-check input shapes match the template.
    for v in ("U", "V", "T", "Q"):
        tgt_shape = ds[v].isel(time=0).shape  # (lev, lat, lon) order in CAM6 h1
        if nudge_targets[v].shape != tgt_shape:
            raise ValueError(
                f"{v} shape mismatch: target {nudge_targets[v].shape} "
                f"vs template {tgt_shape}"
            )

    # Replace U, V, T, Q values; preserve dims and attrs from the template.
    for v in ("U", "V", "T", "Q"):
        ds[v] = (
            ds[v].dims,
            nudge_targets[v][None, ...].astype(np.float32),  # add time dim back
            ds[v].attrs,
        )

    # Drop everything not in `keep`.
    drop_vars = [name for name in list(ds.data_vars) if name not in keep]
    ds = ds.drop_vars(drop_vars)

    # Suppress xarray's default `_FillValue = NaN` on coord vars — the CAM6
    # nudging toolbox is strict about netCDF layout and the source h1 files
    # do NOT carry fill values on coords. The previous SUMO crash was traced
    # to this exact mismatch.
    encoding = {}
    for name in list(ds.coords) + list(ds.data_vars):
        encoding[name] = {"_FillValue": None}

    # Step 3+4: atomic rename.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".nc.tmp", prefix=out_path.name + ".", dir=str(out_path.parent),
    )
    os.close(fd)
    try:
        ds.to_netcdf(tmp_path, format="NETCDF4", encoding=encoding)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        ds.close()


__all__ = [
    "ReverseTranslator",
    "write_nudge_file",
    "GC_TO_CAM_ATMO",
]
