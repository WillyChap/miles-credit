"""Grid definitions and validation for CAM6 (FV f09) and GraphCast (0.25 deg).

Both grids are regular lat-lon, but with different conventions:

  CAM6 FV f09         GraphCast 0.25
  ---------------     -----------------
  192 lat x 288 lon   721 lat x 1440 lon
  lat: -90 .. +90     lat: 90 .. -90 (DESCENDING by ERA5 convention)
  lon: 0 .. 358.75    lon: 0 .. 359.75
  Both poles included Both poles included (lat[0] = 90, lat[-1] = -90)

Read native arrays from the files rather than reconstructing them: we want
the validator to fail loudly if our assumption ever drifts from reality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from . import config as cfg


@dataclass(frozen=True)
class LatLonGrid:
    """A regular lat-lon grid. lat/lon are 1-D, sorted in their stored direction."""

    lat: np.ndarray          # shape (nlat,), degrees north
    lon: np.ndarray          # shape (nlon,), degrees east
    lat_direction: Literal["ascending", "descending"]
    lon_origin: float        # lon[0]; used for periodic wrapping
    name: str

    @property
    def nlat(self) -> int:
        return int(self.lat.size)

    @property
    def nlon(self) -> int:
        return int(self.lon.size)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nlat, self.nlon

    def validate(self) -> None:
        """Cheap structural assertions. Raises AssertionError on mismatch."""
        assert self.lat.ndim == 1, f"{self.name}: lat must be 1D"
        assert self.lon.ndim == 1, f"{self.name}: lon must be 1D"

        # Monotonicity in declared direction.
        if self.lat_direction == "ascending":
            assert np.all(np.diff(self.lat) > 0), f"{self.name}: lat not strictly ascending"
        else:
            assert np.all(np.diff(self.lat) < 0), f"{self.name}: lat not strictly descending"

        # Latitude bounds.
        assert -90.0 - 1e-9 <= self.lat.min() <= -89.0, (
            f"{self.name}: lat min {self.lat.min()} not near -90"
        )
        assert 89.0 <= self.lat.max() <= 90.0 + 1e-9, (
            f"{self.name}: lat max {self.lat.max()} not near +90"
        )

        # Longitude: ascending, in [0, 360).
        assert np.all(np.diff(self.lon) > 0), f"{self.name}: lon not strictly ascending"
        assert 0.0 <= self.lon.min() < 360.0, f"{self.name}: lon min out of [0,360)"
        assert self.lon.max() < 360.0, f"{self.name}: lon max must be < 360"

        # Periodicity check: lon spacing roughly uniform.
        dlon = np.diff(self.lon)
        assert np.allclose(dlon, dlon[0], atol=1e-6), (
            f"{self.name}: lon spacing non-uniform"
        )


# -----------------------------------------------------------------------------
# Factory constructors
# -----------------------------------------------------------------------------


def cam6_grid_from_h1(h1_path: str | Path) -> LatLonGrid:
    """Load CAM6 lat/lon from an h1 file. Verifies shape against config."""
    h1_path = Path(h1_path)
    ds = xr.open_dataset(h1_path)
    try:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
    finally:
        ds.close()

    assert lat.size == cfg.CAM6_NLAT, (
        f"CAM6 nlat from {h1_path.name} is {lat.size}, expected {cfg.CAM6_NLAT}"
    )
    assert lon.size == cfg.CAM6_NLON, (
        f"CAM6 nlon from {h1_path.name} is {lon.size}, expected {cfg.CAM6_NLON}"
    )

    direction: Literal["ascending", "descending"] = (
        "ascending" if lat[1] > lat[0] else "descending"
    )
    g = LatLonGrid(
        lat=lat, lon=lon,
        lat_direction=direction, lon_origin=float(lon[0]),
        name="CAM6_FV_f09",
    )
    g.validate()
    return g


def graphcast_grid() -> LatLonGrid:
    """Construct the GraphCast 0.25 deg ERA5-convention grid.

    ERA5 stores latitudes DESCENDING (90 -> -90). GraphCast follows that.
    """
    lat = np.linspace(90.0, -90.0, cfg.GC_NLAT, dtype=np.float64)
    lon = np.linspace(0.0, 360.0 - 0.25, cfg.GC_NLON, dtype=np.float64)
    g = LatLonGrid(
        lat=lat, lon=lon,
        lat_direction="descending", lon_origin=0.0,
        name="GraphCast_ERA5_0p25",
    )
    g.validate()
    return g


# -----------------------------------------------------------------------------
# Hybrid vertical metadata
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridCoord:
    """CAM6 hybrid-sigma coordinate. Stored exactly as in the file (positive=down)."""

    hyam: np.ndarray   # (nlev,)
    hybm: np.ndarray   # (nlev,)
    hyai: np.ndarray   # (nilev,)
    hybi: np.ndarray   # (nilev,)
    p0: float          # Pa
    positive_down: bool

    def validate(self) -> None:
        assert self.hyam.shape == (cfg.CAM6_NLEV,), self.hyam.shape
        assert self.hybm.shape == (cfg.CAM6_NLEV,), self.hybm.shape
        assert self.hyai.shape == (cfg.CAM6_NILEV,), self.hyai.shape
        assert self.hybi.shape == (cfg.CAM6_NILEV,), self.hybi.shape
        assert abs(self.p0 - cfg.CAM6_P0_PA) < 1.0, f"P0 = {self.p0}, expected {cfg.CAM6_P0_PA}"
        # Bottom interface: hyai+hybi = 1 (pure sigma); top: hyai*P0 should be small.
        bottom_idx = -1 if self.positive_down else 0
        top_idx = 0 if self.positive_down else -1
        assert abs((self.hyai[bottom_idx] + self.hybi[bottom_idx]) - 1.0) < 1e-6, (
            f"bottom interface a+b should be ~1, got {self.hyai[bottom_idx] + self.hybi[bottom_idx]}"
        )
        assert self.hybi[top_idx] == 0.0, (
            f"top interface hybi should be 0 (pure pressure), got {self.hybi[top_idx]}"
        )

    def midpoint_pressures(self, ps: np.ndarray) -> np.ndarray:
        """p_mid[k, ...] = hyam[k] * P0 + hybm[k] * ps[...]; Pa.

        ps may be 1-D (col) or 2-D (lat, lon) or 3-D (time, lat, lon).
        Output has shape (nlev,) + ps.shape.
        """
        ps = np.asarray(ps, dtype=np.float64)
        return (
            self.hyam[(slice(None),) + (None,) * ps.ndim] * self.p0
            + self.hybm[(slice(None),) + (None,) * ps.ndim] * ps[None, ...]
        )

    def interface_pressures(self, ps: np.ndarray) -> np.ndarray:
        """Same as midpoint_pressures but at interfaces (nilev levels)."""
        ps = np.asarray(ps, dtype=np.float64)
        return (
            self.hyai[(slice(None),) + (None,) * ps.ndim] * self.p0
            + self.hybi[(slice(None),) + (None,) * ps.ndim] * ps[None, ...]
        )


def hybrid_coord_from_h1(h1_path: str | Path) -> HybridCoord:
    """Load hyam/hybm/hyai/hybi/P0 from an h1 file."""
    h1_path = Path(h1_path)
    ds = xr.open_dataset(h1_path)
    try:
        hyam = np.asarray(ds["hyam"].values, dtype=np.float64)
        hybm = np.asarray(ds["hybm"].values, dtype=np.float64)
        hyai = np.asarray(ds["hyai"].values, dtype=np.float64)
        hybi = np.asarray(ds["hybi"].values, dtype=np.float64)
        p0 = float(ds["P0"].values)
        # Use lev attribute to determine vertical orientation.
        positive = str(ds["lev"].attrs.get("positive", "down"))
    finally:
        ds.close()

    hc = HybridCoord(
        hyam=hyam, hybm=hybm, hyai=hyai, hybi=hybi,
        p0=p0, positive_down=(positive == "down"),
    )
    hc.validate()
    return hc


__all__ = [
    "LatLonGrid",
    "HybridCoord",
    "cam6_grid_from_h1",
    "graphcast_grid",
    "hybrid_coord_from_h1",
]
