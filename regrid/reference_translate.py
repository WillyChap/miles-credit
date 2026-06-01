"""Reference two-stage CAM6 -> GraphCast translator (NumPy, slow, clear).

This is the correctness baseline. The fused JAX runtime version in
`fused_translate.py` must agree with this output to within a small tolerance
(see config.TranslatorConfig.reference_rtol / reference_atol).

Pipeline for each atmospheric variable on each call:

    CAM6 native (32 hybrid, 192 lat, 288 lon)
      |  1. compute hybrid p_mid from PS + (hyam, hybm) per column
      |  2. log-p interpolate to 37 ERA5 levels  (column-local)
      |  3. variable-specific below-ground extrapolation (config.BG_POLICY)
      v
    CAM6 grid at 37 pressure levels
      |  4. apply precomputed bilinear stencil  (37 * 192 * 288 -> 37 * 721 * 1440)
      v
    GraphCast grid at 37 pressure levels  (dense, no NaNs)

Surface and forcing variables (2m T, 10m U/V, MSLP, precip, TOA solar,
day/year sin/cos) are added in a follow-on step once the new h1 stream lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from . import config as cfg
from .below_ground_era5 import apply_below_ground
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
from .vertical_interp import interp_log_p


# Atmospheric variables we know are in the current h1 stream.
# Z3 / OMEGA are appended once the new fincl2 fields land.
_AVAILABLE_ATMO_TODAY = ("T", "U", "V", "Q")


@dataclass
class ReferenceTranslator:
    """Holds the precomputed horizontal stencil + hybrid coord; reused per file."""

    hybrid: HybridCoord
    cam_grid: LatLonGrid
    gc_grid: LatLonGrid
    cam_to_gc: BilinearStencil
    translator_cfg: cfg.TranslatorConfig = cfg.DEFAULT_CONFIG

    @classmethod
    def build(cls, ref_h1_path: str | Path, translator_cfg=cfg.DEFAULT_CONFIG):
        cam_grid = cam6_grid_from_h1(ref_h1_path)
        hybrid = hybrid_coord_from_h1(ref_h1_path)
        gc_grid = graphcast_grid()
        cam_to_gc = build_bilinear_stencil(cam_grid, gc_grid)
        return cls(
            hybrid=hybrid,
            cam_grid=cam_grid,
            gc_grid=gc_grid,
            cam_to_gc=cam_to_gc,
            translator_cfg=translator_cfg,
        )

    # ------------------------------------------------------------------ #
    # Single-variable translation                                         #
    # ------------------------------------------------------------------ #

    def translate_atmo_var(
        self,
        cam_field: np.ndarray,    # (nlev, nlat_cam, nlon_cam)
        ps: np.ndarray,           # (nlat_cam, nlon_cam) Pa
        gc_var_name: str,
        *,
        z_lowest_m: np.ndarray | None = None,    # required for Z bridge
        t_lowest: np.ndarray | None = None,      # required for Z bridge
    ) -> np.ndarray:
        """Translate one CAM6 hybrid-level field to GraphCast (37 plev, 721, 1440).

        Steps 1-3 happen on CAM6 horizontal grid (192*288 columns).
        Step 4 is a single fused gather to the GC grid.
        """
        assert cam_field.shape == (cfg.CAM6_NLEV, self.cam_grid.nlat, self.cam_grid.nlon)
        assert ps.shape == (self.cam_grid.nlat, self.cam_grid.nlon)

        # 1. Hybrid -> pressure per column.
        p_cam_mid = midpoint_pressures_np(self.hybrid.hyam, self.hybrid.hybm, ps)
        # p_cam_mid shape (nlev, nlat, nlon), monotonically increasing along axis 0.

        # 2. Log-p vertical interpolation to the 37 ERA5 levels.
        p_tgt = np.asarray(cfg.PRESSURE_LEVELS_PA, dtype=np.float64)
        vi = interp_log_p(cam_field.astype(np.float64), p_cam_mid, p_tgt)

        # 3. Below-ground policy.
        ctx = {
            "field_interp": vi.field,
            "field_src": cam_field,
            "below_ground": vi.below_ground,
            "p_lowest": p_cam_mid[-1, ...],
            "p_tgt": p_tgt,
            "taper_lapse": self.translator_cfg.t_bridge_taper_lapse,
        }
        if gc_var_name == "geopotential":
            assert z_lowest_m is not None and t_lowest is not None, (
                "geopotential below-ground requires z_lowest_m and t_lowest"
            )
            ctx["z_lowest_m"] = z_lowest_m
            ctx["t_lowest"] = t_lowest
        field_cam_plev = apply_below_ground(gc_var_name, ctx)

        # 4. Horizontal regrid CAM -> GC.
        field_gc_plev = apply_bilinear(field_cam_plev, self.cam_to_gc)

        # Final check: no NaNs in the GraphCast-bound output.
        if not np.isfinite(field_gc_plev).all():
            bad = (~np.isfinite(field_gc_plev)).sum()
            raise AssertionError(
                f"non-finite values in translated {gc_var_name}: {bad} points"
            )
        return field_gc_plev

    # ------------------------------------------------------------------ #
    # Multi-variable convenience                                          #
    # ------------------------------------------------------------------ #

    def translate_h1_file(
        self,
        h1_path: str | Path,
        time_index: int = 0,
        which: tuple[str, ...] = _AVAILABLE_ATMO_TODAY,
    ) -> xr.Dataset:
        """Translate the atmospheric subset of variables we know are present.

        Returns an xarray Dataset on the GraphCast grid with one variable
        per CAM6 source name (renamed to the GraphCast convention via
        config.ATMO_VAR_MAP).

        Parameters
        ----------
        h1_path : str or Path
        time_index : int
            Which time record in the h1 file (h1 files write one per file with
            mfilt=1, so this is almost always 0).
        which : tuple of str
            CAM6 variable names to translate. Must be a subset of
            ATMO_VAR_MAP keys.
        """
        h1_path = Path(h1_path)
        ds = xr.open_dataset(h1_path)
        try:
            ps = np.asarray(ds["PS"].isel(time=time_index).values, dtype=np.float64)
            # Pull Z3 + T at lowest level if Z3 is available (needed for Z bridge).
            t_lowest = None
            z_lowest_m = None
            if "T" in ds.variables:
                t_lowest = np.asarray(
                    ds["T"].isel(time=time_index, lev=cfg.CAM6_NLEV - 1).values,
                    dtype=np.float64,
                )
            if "Z3" in ds.variables:
                z_lowest_m = np.asarray(
                    ds["Z3"].isel(time=time_index, lev=cfg.CAM6_NLEV - 1).values,
                    dtype=np.float64,
                )

            out = {}
            for cam_name in which:
                if cam_name not in ds.variables:
                    # Skip silently for now; once new h1 lands these will all be present.
                    continue
                gc_name = cfg.ATMO_VAR_MAP[cam_name]
                field = np.asarray(
                    ds[cam_name].isel(time=time_index).values, dtype=np.float64
                )
                # Z3 unit handling: CAM6 stores height in m. GraphCast expects
                # geopotential in m^2/s^2. Multiply by g once; then the bridge
                # works in m^2/s^2 throughout.
                if cam_name == "Z3":
                    field = field * cfg.GRAVITY
                arr = self.translate_atmo_var(
                    field,
                    ps,
                    gc_name,
                    z_lowest_m=z_lowest_m,
                    t_lowest=t_lowest,
                )
                out[gc_name] = arr
        finally:
            ds.close()

        # Wrap in xr.Dataset using GraphCast lat/lon conventions.
        coords = {
            "level": ("level", np.array(cfg.PRESSURE_LEVELS_HPA, dtype=np.int32)),
            "lat":   ("lat", self.gc_grid.lat),
            "lon":   ("lon", self.gc_grid.lon),
        }
        data_vars = {
            name: (("level", "lat", "lon"), arr) for name, arr in out.items()
        }
        attrs = {
            "source_h1": str(h1_path),
            "translator": "reference_translate.ReferenceTranslator",
            "cam_grid": self.cam_grid.name,
            "gc_grid": self.gc_grid.name,
            "below_ground_policy": "config.BG_POLICY",
        }
        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


__all__ = ["ReferenceTranslator"]
