"""Precomputed bilinear stencils between two regular lat-lon grids.

Build once, reuse forever:

    stencil = build_bilinear_stencil(src_grid, dst_grid)
    out_field = apply_bilinear(src_field, stencil)

Stencil layout (fixed 4-point bilinear over lat-lon):
    stencil.jj : (nlat_dst, nlon_dst, 4) int    -- row indices into src
    stencil.ii : (nlat_dst, nlon_dst, 4) int    -- col indices into src
    stencil.ww : (nlat_dst, nlon_dst, 4) float  -- bilinear weights, sum to 1

Conventions:
  * Longitude is periodic. ii wraps mod nlon_src.
  * Latitude is bracketed without wrap. Beyond either pole the stencil collapses
    to the nearest pole row (degenerate weights still sum to 1).
  * Source grid may be ascending or descending in lat; we sort to ascending
    internally for searchsort, then map back. Caller does not need to know.

This is the slowest path the translator has to call, so keep it offline.
The apply step is a pure gather+multiply+sum and goes in fused_translate.py
for the JAX runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .grids import LatLonGrid

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class BilinearStencil:
    """Fixed 4-point bilinear interpolation weights between two regular lat-lon grids."""

    jj: np.ndarray   # (nlat_dst, nlon_dst, 4) int
    ii: np.ndarray   # (nlat_dst, nlon_dst, 4) int
    ww: np.ndarray   # (nlat_dst, nlon_dst, 4) float
    src_shape: tuple[int, int]
    dst_shape: tuple[int, int]
    src_name: str
    dst_name: str

    def validate(self, rtol: float = 1e-12) -> None:
        nlat_dst, nlon_dst = self.dst_shape
        nlat_src, nlon_src = self.src_shape
        assert self.jj.shape == (nlat_dst, nlon_dst, 4), self.jj.shape
        assert self.ii.shape == (nlat_dst, nlon_dst, 4), self.ii.shape
        assert self.ww.shape == (nlat_dst, nlon_dst, 4), self.ww.shape
        assert self.jj.min() >= 0 and self.jj.max() < nlat_src
        assert self.ii.min() >= 0 and self.ii.max() < nlon_src
        # Weights sum to 1 (up to floating point).
        w_sum = self.ww.sum(axis=-1)
        np.testing.assert_allclose(w_sum, 1.0, atol=rtol)
        # Weights non-negative.
        assert self.ww.min() >= -1e-12, f"negative weight {self.ww.min()}"

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            jj=self.jj, ii=self.ii, ww=self.ww,
            src_shape=np.array(self.src_shape, dtype=np.int64),
            dst_shape=np.array(self.dst_shape, dtype=np.int64),
            src_name=np.array(self.src_name),
            dst_name=np.array(self.dst_name),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BilinearStencil":
        d = np.load(path, allow_pickle=False)
        return cls(
            jj=d["jj"], ii=d["ii"], ww=d["ww"],
            src_shape=tuple(int(x) for x in d["src_shape"]),
            dst_shape=tuple(int(x) for x in d["dst_shape"]),
            src_name=str(d["src_name"]),
            dst_name=str(d["dst_name"]),
        )


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


def build_bilinear_stencil(src: LatLonGrid, dst: LatLonGrid) -> BilinearStencil:
    """Build a bilinear stencil from `src` lat-lon to `dst` lat-lon.

    Handles:
      - source ascending or descending in lat
      - longitude periodicity (wrap mod 360 and mod nlon_src)
      - both poles inclusive
      - destination beyond pole row: collapses to nearest pole (weights still sum to 1)
    """
    src.validate()
    dst.validate()

    # Convert source to ascending lat ordering for searchsort, remember the map.
    if src.lat_direction == "ascending":
        src_lat_asc = src.lat
        src_lat_to_orig = np.arange(src.nlat)
    else:
        src_lat_asc = src.lat[::-1]
        src_lat_to_orig = np.arange(src.nlat)[::-1]

    # ---- Latitude bracketing (no wrap, just clamp at edges) ----
    # For each destination lat, find j0 such that src_lat_asc[j0] <= dst_lat < src_lat_asc[j0+1].
    dst_lat = dst.lat
    j0_asc = np.searchsorted(src_lat_asc, dst_lat, side="right") - 1
    j0_asc = np.clip(j0_asc, 0, src.nlat - 2)
    j1_asc = j0_asc + 1

    lat_below = src_lat_asc[j0_asc]
    lat_above = src_lat_asc[j1_asc]
    wlat = (dst_lat - lat_below) / (lat_above - lat_below)
    # Clamp dst lats outside the source range (shouldn't happen with both grids
    # spanning [-90, 90], but defensive: at exactly -90 / +90, wlat is 0 or 1).
    wlat = np.clip(wlat, 0.0, 1.0)

    # Map back to original (possibly descending) source row indices.
    j0 = src_lat_to_orig[j0_asc]
    j1 = src_lat_to_orig[j1_asc]

    # ---- Longitude bracketing with periodicity ----
    # src lons are ascending in [0, 360). Spacing assumed uniform (validated).
    dlon = (src.lon[-1] - src.lon[0]) / (src.nlon - 1)
    # Treat src as cyclic with period 360. Distance of dst lon past src.lon[0]:
    dst_lon = np.asarray(dst.lon)
    rel = (dst_lon - src.lon[0]) % 360.0
    fi = rel / dlon
    i0 = np.floor(fi).astype(np.int64) % src.nlon
    i1 = (i0 + 1) % src.nlon
    wlon = fi - np.floor(fi)
    wlon = np.clip(wlon, 0.0, 1.0)

    # ---- Assemble the 4-point stencil ----
    # Stencil order:
    #   k=0: (j0, i0)   weight = (1-wlat)*(1-wlon)
    #   k=1: (j0, i1)   weight = (1-wlat)*wlon
    #   k=2: (j1, i0)   weight = wlat*(1-wlon)
    #   k=3: (j1, i1)   weight = wlat*wlon
    nlat_dst, nlon_dst = dst.nlat, dst.nlon

    jj = np.empty((nlat_dst, nlon_dst, 4), dtype=np.int64)
    ii = np.empty((nlat_dst, nlon_dst, 4), dtype=np.int64)
    ww = np.empty((nlat_dst, nlon_dst, 4), dtype=np.float64)

    # Broadcast j0/j1 across lon, i0/i1 across lat.
    j0b = j0[:, None]
    j1b = j1[:, None]
    i0b = i0[None, :]
    i1b = i1[None, :]
    wlatb = wlat[:, None]
    wlonb = wlon[None, :]

    jj[..., 0] = j0b
    jj[..., 1] = j0b
    jj[..., 2] = j1b
    jj[..., 3] = j1b

    ii[..., 0] = i0b
    ii[..., 1] = i1b
    ii[..., 2] = i0b
    ii[..., 3] = i1b

    ww[..., 0] = (1 - wlatb) * (1 - wlonb)
    ww[..., 1] = (1 - wlatb) * wlonb
    ww[..., 2] = wlatb * (1 - wlonb)
    ww[..., 3] = wlatb * wlonb

    stencil = BilinearStencil(
        jj=jj, ii=ii, ww=ww,
        src_shape=src.shape,
        dst_shape=dst.shape,
        src_name=src.name,
        dst_name=dst.name,
    )
    stencil.validate()
    return stencil


# -----------------------------------------------------------------------------
# Apply (NumPy reference; JAX version lives in fused_translate.py)
# -----------------------------------------------------------------------------


def apply_bilinear(field: np.ndarray, stencil: BilinearStencil) -> np.ndarray:
    """Apply the precomputed stencil to a source field.

    Parameters
    ----------
    field : np.ndarray
        Shape (..., nlat_src, nlon_src). Leading axes (level, var, time) are
        preserved.
    stencil : BilinearStencil

    Returns
    -------
    np.ndarray
        Shape (..., nlat_dst, nlon_dst).
    """
    assert field.shape[-2:] == stencil.src_shape, (
        f"field horizontal shape {field.shape[-2:]} != stencil src {stencil.src_shape}"
    )
    # Fancy index: field[..., jj, ii] has shape (..., nlat_dst, nlon_dst, 4)
    gathered = field[..., stencil.jj, stencil.ii]
    return (gathered * stencil.ww).sum(axis=-1)


__all__ = [
    "BilinearStencil",
    "build_bilinear_stencil",
    "apply_bilinear",
]
