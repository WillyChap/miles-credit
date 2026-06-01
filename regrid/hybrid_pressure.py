"""CAM6 hybrid-sigma pressure formulas.

Two layers:
  - NumPy versions used by the reference / offline pipeline (rich shape handling).
  - Bare-array vectorized versions safe to import into JAX (no xarray, no Path).

The pressure formulas are:
    p_mid[k, col]       = hyam[k] * P0 + hybm[k] * ps[col]
    p_interface[k, col] = hyai[k] * P0 + hybi[k] * ps[col]

CAM6 stores `lev` with `positive='down'`, so index 0 is TOA and index nlev-1
is the lowest model level (closest to the surface). These functions preserve
that ordering -- they do not flip.
"""

from __future__ import annotations

import numpy as np

from . import config as cfg


def midpoint_pressures_np(
    hyam: np.ndarray, hybm: np.ndarray, ps: np.ndarray, p0: float = cfg.CAM6_P0_PA
) -> np.ndarray:
    """Midpoint pressures in Pa.

    Parameters
    ----------
    hyam, hybm : (nlev,)
        Hybrid-A and hybrid-B midpoint coefficients (unitless).
    ps : array_like
        Surface pressure in Pa. Any shape; output gets nlev prepended.
    p0 : float
        Reference pressure in Pa (default 100000).

    Returns
    -------
    np.ndarray
        Shape (nlev,) + ps.shape.
    """
    ps = np.asarray(ps, dtype=np.float64)
    nlev = hyam.size
    assert hybm.size == nlev, "hyam and hybm must have same length"
    # Broadcast hyam, hybm to (nlev,) + (1,)*ps.ndim
    slc = (slice(None),) + (None,) * ps.ndim
    return hyam[slc] * p0 + hybm[slc] * ps[None, ...]


def interface_pressures_np(
    hyai: np.ndarray, hybi: np.ndarray, ps: np.ndarray, p0: float = cfg.CAM6_P0_PA
) -> np.ndarray:
    """Interface pressures in Pa. Shape (nilev,) + ps.shape."""
    ps = np.asarray(ps, dtype=np.float64)
    nilev = hyai.size
    assert hybi.size == nilev, "hyai and hybi must have same length"
    slc = (slice(None),) + (None,) * ps.ndim
    return hyai[slc] * p0 + hybi[slc] * ps[None, ...]


def layer_thickness_np(
    hyai: np.ndarray, hybi: np.ndarray, ps: np.ndarray, p0: float = cfg.CAM6_P0_PA
) -> np.ndarray:
    """delta_p across each midpoint layer in Pa. Shape (nlev,) + ps.shape.

    CAM6 lev is positive=down: p_interface[k+1] > p_interface[k], so the layer
    delta_p for midpoint k is p_interface[k+1] - p_interface[k].
    """
    pi = interface_pressures_np(hyai, hybi, ps, p0=p0)  # (nilev,) + ps.shape
    return pi[1:, ...] - pi[:-1, ...]


# -----------------------------------------------------------------------------
# JAX-safe duplicates (same math, jnp-friendly)
# -----------------------------------------------------------------------------
# Keep these in a separate import-guarded block so importing this module does
# not pay the JAX startup cost. Callers that want JAX should import the `_jnp`
# helpers explicitly.

def midpoint_pressures_jnp(hyam, hybm, ps, p0: float = cfg.CAM6_P0_PA):
    """JAX equivalent of midpoint_pressures_np. Caller imports jnp arrays."""
    import jax.numpy as jnp
    slc = (slice(None),) + (None,) * ps.ndim
    return hyam[slc] * p0 + hybm[slc] * ps[None, ...]


def interface_pressures_jnp(hyai, hybi, ps, p0: float = cfg.CAM6_P0_PA):
    import jax.numpy as jnp
    slc = (slice(None),) + (None,) * ps.ndim
    return hyai[slc] * p0 + hybi[slc] * ps[None, ...]


__all__ = [
    "midpoint_pressures_np",
    "interface_pressures_np",
    "layer_thickness_np",
    "midpoint_pressures_jnp",
    "interface_pressures_jnp",
]
