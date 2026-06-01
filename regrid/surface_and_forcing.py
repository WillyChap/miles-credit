"""Surface, static, and forcing variable translation for CAM6 -> GraphCast.

Surface and static variables are 2D fields -- only horizontal regrid, no
vertical interpolation. They reuse the same precomputed bilinear stencil
as the atmospheric translator.

Forcing variables (TOA incident solar radiation, day/year progress sin/cos)
are computed by GraphCast itself from the `datetime` coordinate, so this
module does NOT compute them -- it just makes sure the bundle carries the
right `datetime` coord so GraphCast can derive them.

CAM6 -> GraphCast conventions implemented here:
  TREFHT   -> 2m_temperature            (K, pass-through)
  PSL      -> mean_sea_level_pressure   (Pa, pass-through)
  PRECT    -> total_precipitation_6hr   (m, = m/s * 21600)
  U10 + UBOT + VBOT
           -> 10m_u_component_of_wind   (m/s, direction from {UBOT,VBOT})
           -> 10m_v_component_of_wind   (m/s, magnitude from U10)
  PHIS     -> geopotential_at_surface   (m^2/s^2, pass-through, static)
  LANDFRAC -> land_sea_mask             (0..1, pass-through, static)

CAM6 lacks separate 10m U/V wind components; only the 10m wind SPEED (`U10`)
is diagnosed. We reconstruct components by rotating U10 along the lowest-
model-level wind direction. The lowest model level sits ~70 m above the
surface, so the direction is a good approximation for the 10 m wind under
near-neutral conditions and is the best signal we have without modifying
CAM6's surface scheme.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from . import config as cfg
from .horizontal_weights import BilinearStencil, apply_bilinear

DT_SEC = 21600.0          # 6 hours, seconds (GraphCast time step)


# -----------------------------------------------------------------------------
# 10 m wind reconstruction
# -----------------------------------------------------------------------------


def reconstruct_10m_components(
    u10_speed: np.ndarray, ubot: np.ndarray, vbot: np.ndarray,
    eps_wind_speed: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize 10m_u, 10m_v from CAM6's U10 magnitude + lowest-level direction.

        u10 = U10 * UBOT / |UBOT, VBOT|
        v10 = U10 * VBOT / |UBOT, VBOT|

    When the lowest-level wind speed is below `eps_wind_speed` (essentially
    calm), we set both components to zero rather than divide by a tiny vector.
    This is rare on the CAM6 grid and matches the physical expectation that
    the 10 m wind is also essentially zero there.
    """
    mag = np.sqrt(ubot * ubot + vbot * vbot)
    safe = mag > eps_wind_speed
    u10 = np.zeros_like(u10_speed)
    v10 = np.zeros_like(u10_speed)
    np.divide(u10_speed * ubot, mag, out=u10, where=safe)
    np.divide(u10_speed * vbot, mag, out=v10, where=safe)
    return u10, v10


# -----------------------------------------------------------------------------
# Per-h1 surface translation
# -----------------------------------------------------------------------------


def translate_surface_from_h1(
    h1_path: str | Path, stencil: BilinearStencil, time_index: int = 0,
) -> dict[str, np.ndarray]:
    """Return dict of GraphCast-named surface fields on the destination grid.

    Output arrays are 2D (nlat_gc, nlon_gc), float64.
    """
    h1_path = Path(h1_path)
    ds = xr.open_dataset(h1_path)
    try:
        ti = time_index

        trefht = np.asarray(ds["TREFHT"].isel(time=ti).values, dtype=np.float64)
        psl    = np.asarray(ds["PSL"   ].isel(time=ti).values, dtype=np.float64)
        prect  = np.asarray(ds["PRECT" ].isel(time=ti).values, dtype=np.float64)
        u10    = np.asarray(ds["U10"   ].isel(time=ti).values, dtype=np.float64)
        ubot   = np.asarray(ds["UBOT"  ].isel(time=ti).values, dtype=np.float64)
        vbot   = np.asarray(ds["VBOT"  ].isel(time=ti).values, dtype=np.float64)
    finally:
        ds.close()

    # Reconstruct 10m components on the CAM6 grid first (preserves accuracy of
    # the U10 magnitude and the UBOT/VBOT direction). Then horizontal-regrid.
    u10_u_cam, u10_v_cam = reconstruct_10m_components(u10, ubot, vbot)

    out = {
        "2m_temperature":          apply_bilinear(trefht, stencil),
        "mean_sea_level_pressure": apply_bilinear(psl,    stencil),
        # PRECT is m/s averaged over the 6h interval -> multiply to get m / 6h.
        "total_precipitation_6hr": apply_bilinear(prect * DT_SEC, stencil),
        "10m_u_component_of_wind": apply_bilinear(u10_u_cam, stencil),
        "10m_v_component_of_wind": apply_bilinear(u10_v_cam, stencil),
    }
    # GraphCast requires non-negative precip. CAM6 PRECT is always >=0 but
    # floating-point arithmetic in the bilinear gather can produce small
    # negatives at the edges of vanishing-precip regions.
    out["total_precipitation_6hr"] = np.maximum(out["total_precipitation_6hr"], 0.0)
    return out


# -----------------------------------------------------------------------------
# Static fields (once per case)
# -----------------------------------------------------------------------------


def translate_static_from_h1(
    h1_path: str | Path, stencil: BilinearStencil, time_index: int = 0,
) -> dict[str, np.ndarray]:
    """Return dict of GraphCast-named static fields on the destination grid.

    PHIS and LANDFRAC are time-invariant per case (PHIS exactly, LANDFRAC
    nearly), so this is typically called once and the result cached.
    """
    h1_path = Path(h1_path)
    ds = xr.open_dataset(h1_path)
    try:
        ti = time_index
        phis = np.asarray(ds["PHIS"    ].isel(time=ti).values, dtype=np.float64)
        lndf = np.asarray(ds["LANDFRAC"].isel(time=ti).values, dtype=np.float64)
    finally:
        ds.close()

    return {
        "geopotential_at_surface": apply_bilinear(phis, stencil),
        # land_sea_mask must lie in [0, 1] even after bilinear smoothing.
        "land_sea_mask": np.clip(apply_bilinear(lndf, stencil), 0.0, 1.0),
    }


__all__ = [
    "reconstruct_10m_components",
    "translate_surface_from_h1",
    "translate_static_from_h1",
    "DT_SEC",
]
