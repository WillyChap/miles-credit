"""Blend GraphCast + CAMulator predictions into a single CAM6 nudge target.

Architectural role
==================

This is the heart of the supermodel: CAM6 is the dynamical core we trust to
conserve mass and energy, but we *steer* it via 6-hourly Newtonian relaxation
toward the consensus of two surrogate forecasters:

  - CAMulator: PyTorch neural surrogate trained on CAM6 climate (native CAM6
    hybrid grid, sees the full hybrid column including the stratosphere).
  - GraphCast: JAX graph-NN trained on ERA5 (native 0.25 deg / 37 ERA5 plev,
    sees full physical-atmosphere structure but only down to 1000 hPa).

Both produce a +6h prediction. After the reverse translator brings GraphCast
back onto the CAM6 hybrid grid, BOTH live in the same (32 lev, 192 lat,
288 lon) space, and we can compute a per-(var, level, lat, lon) weighted
blend:

    nudge_target[var, k, j, i] = w_cam[var, k, j, i] * cam_pred[var, k, j, i]
                                + w_gc[var, k, j, i]  * gc_pred[var, k, j, i]

The weights are normalized to sum to 1 at every point so the blend is a
proper convex combination (no spurious amplification).

Default policy
==============

For first runs we use a single scalar weight per variable, the same at every
level and lat/lon, equal 50/50. This is the simplest defensible choice; we
do not yet have empirical evidence that one surrogate dominates in any
particular regime, and a uniform blend keeps the system easy to debug.

More sophisticated policies are designed as drop-in replacements that take
(var, level_idx, lat_array) and return (w_cam, w_gc), each shape (32, 192,
288), sum to 1 at every point. Examples:

  - Level taper: 50/50 in the troposphere, ramp to 90/10 favoring CAMulator
    above 50 hPa where GraphCast loses its training resolution.
  - Latitude taper: weight GC slightly more in mid-latitudes where ERA5 has
    rich observations, slightly less near the poles where ERA5 is interpolated.
  - Online consensus weights: track the running RMSE of each surrogate vs
    CAM6 and adjust toward the better performer.

PS policy
=========

PS is NOT nudged. CAM6's FV dycore determines PS from mass conservation;
naively replacing it with a surrogate's MSLP would break dynamical balance.
The blended nudge file therefore carries only U, V, T, Q.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import xarray as xr

from . import config as cfg


# Nudged vars + their non-negativity rule.
NUDGED_VARS: tuple[str, ...] = ("U", "V", "T", "Q")
CLIP_NONNEG: dict[str, bool] = {"U": False, "V": False, "T": False, "Q": True}


# ----------------------------------------------------------------------------
# Weight policies
# ----------------------------------------------------------------------------

WeightFn = Callable[[str, int, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
"""Signature: (var, level_idx, lat, lon) -> (w_cam, w_gc), shapes broadcastable
to (nlat_cam, nlon_cam). Caller normalizes."""


def equal_weights() -> Mapping[str, float]:
    """Simplest policy: 50/50 every variable."""
    return {v: 0.5 for v in NUDGED_VARS}


def cam_favored_in_stratosphere(
    cfg_obj=cfg, transition_hpa: float = 50.0, transition_width_hpa: float = 30.0,
):
    """Return a per-level w_cam array that ramps from 0.5 (troposphere) to
    0.9 (above 50 hPa), with a smooth transition.

    The CAM6 hybrid grid sits in ascending k = TOA-to-surface. Levels 0..~8
    are above 50 hPa where GraphCast lacks training resolution; we lean
    toward CAMulator there.

    Returns a 1-D array of length nlev=32 with values in [0.5, 0.9].
    """
    # Use a typical (PS = 1013 hPa) midpoint pressure curve for this
    # weighting; we don't need PS-dependent weights since this is a
    # broad guidance, not exact mapping.
    p_typical = np.array([
        0.00364, 0.00759, 0.01435, 0.02461, 0.03592, 0.04319, 0.05167,
        0.06152, 0.07375, 0.08782, 0.10331, 0.12154, 0.14299, 0.16822,
        0.19790, 0.23282, 0.27391, 0.32224, 0.37910, 0.44599, 0.52468,
        0.60977, 0.69138, 0.76340, 0.82085, 0.85953, 0.88702, 0.91264,
        0.93619, 0.95748, 0.97632, 0.99255,
    ]) * 1013.25  # hPa, approximate ascending TOA->surface
    # smoothstep from low-pressure side
    x = (transition_hpa - p_typical) / transition_width_hpa
    s = np.clip(0.5 * (1 + np.tanh(x)), 0.0, 1.0)   # 1 above stratosphere, 0 below
    w_cam = 0.5 + 0.4 * s   # ranges 0.5 (trop) -> 0.9 (high stratosphere)
    return w_cam


# ----------------------------------------------------------------------------
# Blender
# ----------------------------------------------------------------------------

@dataclass
class BlendConfig:
    """Weights to use when combining CAMulator and GraphCast predictions."""

    w_cam_per_var: dict[str, float] | None = None       # scalar fallback
    w_cam_per_var_per_lev: dict[str, np.ndarray] | None = None  # (32,) override

    def w_cam(self, var: str) -> np.ndarray:
        """Return shape-(nlev=32,) weight array for CAMulator on `var`."""
        if self.w_cam_per_var_per_lev and var in self.w_cam_per_var_per_lev:
            arr = np.asarray(self.w_cam_per_var_per_lev[var], dtype=np.float64)
            assert arr.shape == (cfg.CAM6_NLEV,), arr.shape
            return arr
        if self.w_cam_per_var and var in self.w_cam_per_var:
            return np.full(cfg.CAM6_NLEV, float(self.w_cam_per_var[var]))
        # Default
        return np.full(cfg.CAM6_NLEV, 0.5)


def default_blend_config() -> BlendConfig:
    return BlendConfig(w_cam_per_var={v: 0.5 for v in NUDGED_VARS})


def stratosphere_lean_blend_config() -> BlendConfig:
    """50/50 in troposphere, 80-90/10-20 favoring CAMulator above ~50 hPa."""
    w_strat = cam_favored_in_stratosphere()
    return BlendConfig(
        w_cam_per_var_per_lev={v: w_strat for v in NUDGED_VARS},
    )


def blend(
    cam_pred: Mapping[str, np.ndarray] | None,
    gc_pred:  Mapping[str, np.ndarray] | None,
    cfg_blend: BlendConfig | None = None,
) -> dict[str, np.ndarray]:
    """Blend two equal-shape prediction bundles on the CAM6 hybrid grid.

    Parameters
    ----------
    cam_pred, gc_pred : dict[str, (nlev=32, nlat=192, nlon=288)] or None
        CAMulator and reverse-translated GraphCast predictions on CAM6 grid.
        If one is None, the other is passed through unchanged (single-source
        nudge mode). At least one must be provided.
    cfg_blend : BlendConfig, optional
        Weighting policy. Defaults to 50/50 every variable, every level.

    Returns
    -------
    dict[str, np.ndarray]
        Blended targets in the same layout, ready for `write_nudge_file`.
    """
    # Single-source pass-through: caller is running in camulator-only or
    # graphcast-only mode. Just validate shapes and copy through.
    if cam_pred is None and gc_pred is None:
        raise ValueError("blend(): at least one of cam_pred, gc_pred must be provided")
    if cam_pred is None or gc_pred is None:
        sole = cam_pred if gc_pred is None else gc_pred
        expected_shape = (cfg.CAM6_NLEV, cfg.CAM6_NLAT, cfg.CAM6_NLON)
        out: dict[str, np.ndarray] = {}
        for v in NUDGED_VARS:
            if v not in sole:
                raise KeyError(f"Single-source bundle missing {v}")
            arr = np.asarray(sole[v], dtype=np.float64)
            if arr.shape != expected_shape:
                raise ValueError(f"{v} shape {arr.shape} != {expected_shape}")
            if CLIP_NONNEG[v]:
                arr = np.maximum(arr, 0.0)
            if not np.isfinite(arr).all():
                raise AssertionError(f"non-finite values in single-source {v}")
            out[v] = arr
        return out

    cfg_blend = cfg_blend or default_blend_config()
    out: dict[str, np.ndarray] = {}
    expected_shape = (cfg.CAM6_NLEV, cfg.CAM6_NLAT, cfg.CAM6_NLON)
    for v in NUDGED_VARS:
        if v not in cam_pred or v not in gc_pred:
            raise KeyError(f"Both bundles must contain {v}")
        a = np.asarray(cam_pred[v], dtype=np.float64)
        b = np.asarray(gc_pred[v],  dtype=np.float64)
        if a.shape != expected_shape:
            raise ValueError(f"cam {v} shape {a.shape} != {expected_shape}")
        if b.shape != expected_shape:
            raise ValueError(f"gc  {v} shape {b.shape} != {expected_shape}")

        w_cam_k = cfg_blend.w_cam(v)            # (32,)
        w_cam = w_cam_k[:, None, None]          # (32, 1, 1) -> broadcast
        w_gc  = 1.0 - w_cam

        merged = w_cam * a + w_gc * b
        if CLIP_NONNEG[v]:
            merged = np.maximum(merged, 0.0)
        if not np.isfinite(merged).all():
            raise AssertionError(f"non-finite values in blended {v}")
        out[v] = merged
    return out


# ----------------------------------------------------------------------------
# Diagnostic
# ----------------------------------------------------------------------------

def report_blend(
    cam_pred: Mapping[str, np.ndarray],
    gc_pred:  Mapping[str, np.ndarray],
    blended:  Mapping[str, np.ndarray],
) -> None:
    """Print per-var summary of how much each surrogate pulled the result."""
    print("\n== Blend summary ==")
    print(f"  {'var':>3}  "
          f"{'CAM mean':>10}  {'GC mean':>10}  {'blend mean':>10}  "
          f"{'CAM-GC RMSD':>12}")
    for v in NUDGED_VARS:
        a = np.asarray(cam_pred[v]); b = np.asarray(gc_pred[v])
        m = np.asarray(blended[v])
        rmsd = float(np.sqrt(((a - b) ** 2).mean()))
        print(f"  {v:>3}  "
              f"{float(a.mean()):10.4g}  {float(b.mean()):10.4g}  "
              f"{float(m.mean()):10.4g}  {rmsd:12.4g}")


__all__ = [
    "NUDGED_VARS",
    "CLIP_NONNEG",
    "BlendConfig",
    "default_blend_config",
    "stratosphere_lean_blend_config",
    "blend",
    "report_blend",
]
