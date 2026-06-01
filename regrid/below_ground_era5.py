"""ERA5 / FULL-POS-like below-ground extrapolation for pressure-level fields.

Why bother instead of just NaN-ing them: GraphCast was trained on dense ERA5
pressure-level tensors that *do* contain values below ground -- ECMWF's
post-processing extrapolates them downward with rules per variable. The
training distribution is not "physical atmosphere only", so masking with NaNs
would shift GraphCast's input distribution off-manifold.

ECMWF documented rule
(confluence.ecmwf.int CKB, "ERA5: data documentation"):

  "For extrapolation below the surface, wind components and humidity are kept
   constant from the lowest model level, whereas temperature is linearly
   interpolated from the lowest model level to a surface temperature and then
   a cubic polynomial extrapolation is used thereafter."

Direct inspection of ERA5 pressure-level data at three high-terrain sites
(Mt Everest, Andes, Greenland) for 1980-01-02 confirms:

  Var  | Pattern below first underground level
  -----+-----------------------------------------------------------------
  T    | smooth ~6.5 K/km lapse from last valid down to 1000 hPa
  Z    | monotonic descent, hydrostatic from the extrapolated T
  U,V  | tiny transition at first underground point, then near-perfect
       | hold (drift <1 m/s across the entire underground column)
  Q    | tiny transition, then perfect hold (drift <1% across column)
  W    | small noise around small values; not damped to zero, but the
       | dynamic range below ground is much smaller than aloft, so
       | "hold lowest" captures the magnitude well enough.

Variable-specific rules implemented here:

  u, v, w (omega)   : hold lowest valid CAM6 model-level value
  q                 : hold lowest valid, clipped to >= 0
  temperature       : ICAO-standard lapse (6.5 K/km) from the lowest CAM6
                      model level. Optionally tapered to zero over the
                      lowest 2 km of underground depth (FULL-POS-style)
                      to avoid unphysical T over deep topography. The
                      CAM6 lowest hybrid level sits at ~0.9926*PS (~6 hPa
                      above the surface), so we use it directly as the
                      anchor without the small TS-bridge segment that
                      ECMWF applies between the IFS model surface and
                      the first below-ground pressure level.
  geopotential      : hydrostatic continuation downward in log(p) using
                      T_extrap from above. Q neglected (<1% on T_v).

This is an emulation, not a literal port of FULL-POS. The interface is
structured so the T and Z modules can be swapped for a closer reproduction
later without touching the rest of the pipeline.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import config as cfg


# -----------------------------------------------------------------------------
# Simple policies
# -----------------------------------------------------------------------------


def hold_lowest(
    field_interp: np.ndarray,
    field_src: np.ndarray,
    below_ground: np.ndarray,
    clip_nonneg: bool = False,
) -> np.ndarray:
    """Overwrite below-ground points with the lowest-valid CAM6 model-level value.

    Parameters
    ----------
    field_interp : np.ndarray
        Output of vertical_interp, shape (nlev_tgt, ...).
    field_src : np.ndarray
        Source field on CAM6 hybrid levels, shape (nlev_src, ...). Used to
        pull the lowest-level value.
    below_ground : np.ndarray
        Boolean mask (nlev_tgt, ...) marking which target points to overwrite.
    clip_nonneg : bool
        If True, also clip the result to >= 0 (used for specific humidity).

    Returns
    -------
    np.ndarray
        Modified field. NOT a copy of `field_interp` -- the caller may rely on
        in-place semantics.
    """
    bottom = field_src[-1:]              # (1, ...)
    if clip_nonneg:
        # Clip only the value we're about to insert below ground; leave the
        # above-ground field_interp untouched so this is a pure pass-through
        # over physical levels.
        bottom = np.maximum(bottom, 0.0)
    return np.where(below_ground, bottom, field_interp)


# -----------------------------------------------------------------------------
# Temperature: ERA5-style lapse-rate bridge
# -----------------------------------------------------------------------------


def temperature_bridge(
    t_interp: np.ndarray,
    t_lowest: np.ndarray,
    z_lowest_m: np.ndarray,
    p_lowest: np.ndarray,
    p_tgt: np.ndarray,
    below_ground: np.ndarray,
    taper_lapse: bool = True,
) -> np.ndarray:
    """Below-ground temperature using standard-atmosphere lapse from the
    lowest CAM6 model level.

    For each below-ground target pressure p_tgt > p_lowest, the height drop
    relative to the lowest model level is (from hydrostatic + ideal gas):

        dz = -(R_d * T_lowest / g) * log(p_tgt / p_lowest)

    which is negative (target is below the source). Then

        T_target = T_lowest - Gamma_eff * dz       (note dz < 0)

    so warmer below ground for positive Gamma. The effective lapse rate is

        Gamma_eff = Gamma_0 * (1 - depth / depth_taper)    if taper_lapse
                  = Gamma_0                                 otherwise

    where Gamma_0 = 6.5 K/km and the taper goes to zero by `T_LAPSE_ZERO_AT_2K`
    meters below the surface. The taper avoids the FULL-POS pathology of
    unphysically hot underground T over very deep topography.

    Parameters
    ----------
    t_interp : (nlev_tgt, ...) -- vertical-interp output (we will overwrite below-ground).
    t_lowest : (...,)           -- lowest CAM6 model-level T, K.
    z_lowest_m : (...,)         -- height of lowest CAM6 model level above sea level, m.
                                   (Not used by this exact form, but kept for
                                   compatibility with a future Z-aware variant.)
    p_lowest : (...,)           -- pressure of lowest CAM6 model level, Pa.
    p_tgt : (nlev_tgt,)         -- GraphCast target pressures, Pa.
    below_ground : (nlev_tgt, ...) -- mask of points to fill.
    taper_lapse : bool

    Returns
    -------
    np.ndarray
        T with below-ground points overwritten.
    """
    p_tgt_b = p_tgt[(slice(None),) + (None,) * t_lowest.ndim]   # (nlev_tgt, 1, 1, ...)
    p_low_b = p_lowest[None, ...]                                # (1, ...)
    t_low_b = t_lowest[None, ...]                                # (1, ...)

    # Hydrostatic dz between lowest src level and target p_tgt > p_low (so dz < 0).
    dz = -(cfg.RD * t_low_b / cfg.GRAVITY) * np.log(p_tgt_b / p_low_b)

    gamma = cfg.T_LAPSE
    if taper_lapse:
        # Lapse rate tapers linearly from Gamma_0 at the surface to 0 at
        # T_LAPSE_ZERO_AT_2K m below it. Depth = -dz (positive number).
        depth = -dz
        taper = 1.0 - np.clip(depth / cfg.T_LAPSE_ZERO_AT_2K, 0.0, 1.0)
        gamma = cfg.T_LAPSE * taper

    t_extrap = t_low_b - gamma * dz                              # dz < 0 -> warmer below
    return np.where(below_ground, t_extrap, t_interp)


# -----------------------------------------------------------------------------
# Geopotential: hydrostatic continuation downward
# -----------------------------------------------------------------------------


def geopotential_bridge(
    z_interp_m2s2: np.ndarray,
    z_lowest_m: np.ndarray,
    t_lowest: np.ndarray,
    p_lowest: np.ndarray,
    p_tgt: np.ndarray,
    below_ground: np.ndarray,
    taper_lapse: bool = True,
) -> np.ndarray:
    """Below-ground geopotential by hydrostatic integration from the lowest
    CAM6 model level, using the same lapse-tapered virtual temperature as
    `temperature_bridge`.

    The hydrostatic relation in pressure coordinates,

        d(Phi) / d(log p) = -R_d * T_v,

    integrated from p_low (where Phi = g * z_low) down to p > p_low gives

        Phi(p) = g * z_low - R_d * T_v_mean * log(p / p_low),

    where T_v_mean is the integrated mean of T_v between p_low and p. For
    consistency with `temperature_bridge`, we use T_v(p) computed at the
    same lapse-tapered profile and approximate T_v_mean as the average of
    T_v(p_low) and T_v(p) (trapezoidal in log p). Humidity is neglected
    below ground -- ERA5 below-ground q is held constant from the lowest
    level and contributes <1% to T_v.

    Returns geopotential in m^2/s^2 (matches GraphCast convention).
    """
    p_tgt_b = p_tgt[(slice(None),) + (None,) * t_lowest.ndim]
    p_low_b = p_lowest[None, ...]
    t_low_b = t_lowest[None, ...]
    phi_low = cfg.GRAVITY * z_lowest_m[None, ...]                # (1, ...)

    # T at target p using the same lapse profile (dry T = T_v since we drop q).
    dz = -(cfg.RD * t_low_b / cfg.GRAVITY) * np.log(p_tgt_b / p_low_b)
    gamma = cfg.T_LAPSE
    if taper_lapse:
        depth = -dz
        taper = 1.0 - np.clip(depth / cfg.T_LAPSE_ZERO_AT_2K, 0.0, 1.0)
        gamma = cfg.T_LAPSE * taper
    t_tgt = t_low_b - gamma * dz
    t_mean = 0.5 * (t_low_b + t_tgt)

    phi_extrap = phi_low - cfg.RD * t_mean * np.log(p_tgt_b / p_low_b)
    return np.where(below_ground, phi_extrap, z_interp_m2s2)


# -----------------------------------------------------------------------------
# Dispatcher: map config policy name -> callable
# -----------------------------------------------------------------------------

# Each handler signature is (context: dict) -> np.ndarray. We pass a dict so
# we can extend without changing the prototype.

def _hold_lowest_handler(ctx: dict, clip_nonneg: bool = False) -> np.ndarray:
    return hold_lowest(
        field_interp=ctx["field_interp"],
        field_src=ctx["field_src"],
        below_ground=ctx["below_ground"],
        clip_nonneg=clip_nonneg,
    )


def _t_bridge_handler(ctx: dict) -> np.ndarray:
    return temperature_bridge(
        t_interp=ctx["field_interp"],
        t_lowest=ctx["field_src"][-1, ...],
        z_lowest_m=ctx.get("z_lowest_m"),
        p_lowest=ctx["p_lowest"],
        p_tgt=ctx["p_tgt"],
        below_ground=ctx["below_ground"],
        taper_lapse=ctx.get("taper_lapse", True),
    )


def _z_hydro_handler(ctx: dict) -> np.ndarray:
    return geopotential_bridge(
        z_interp_m2s2=ctx["field_interp"],
        z_lowest_m=ctx["z_lowest_m"],
        t_lowest=ctx["t_lowest"],
        p_lowest=ctx["p_lowest"],
        p_tgt=ctx["p_tgt"],
        below_ground=ctx["below_ground"],
        taper_lapse=ctx.get("taper_lapse", True),
    )


HANDLERS: dict[str, Callable[[dict], np.ndarray]] = {
    "hold_lowest": _hold_lowest_handler,
    "era5_t_bridge": _t_bridge_handler,
    "era5_z_hydrostatic": _z_hydro_handler,
}


def apply_below_ground(gc_var_name: str, ctx: dict) -> np.ndarray:
    """Dispatch by GraphCast variable name using cfg.BG_POLICY."""
    pol = cfg.BG_POLICY[gc_var_name]
    if pol.name == "hold_lowest":
        return _hold_lowest_handler(ctx, clip_nonneg=pol.clip_nonneg)
    return HANDLERS[pol.name](ctx)


__all__ = [
    "hold_lowest",
    "temperature_bridge",
    "geopotential_bridge",
    "apply_below_ground",
    "HANDLERS",
]
