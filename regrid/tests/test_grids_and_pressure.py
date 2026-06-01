"""Smoke tests that run against a real CAM6 h1 file. No fixtures, no mocks.

Run from camulator_sumo/:
    python -m pytest regrid/tests/test_grids_and_pressure.py -v

Or directly:
    python -m regrid.tests.test_grids_and_pressure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from regrid import config as cfg
from regrid.grids import (
    LatLonGrid,
    cam6_grid_from_h1,
    graphcast_grid,
    hybrid_coord_from_h1,
)
from regrid.hybrid_pressure import (
    interface_pressures_np,
    layer_thickness_np,
    midpoint_pressures_np,
)

# Use the most recent SUMO h1 we know is well-formed.
H1_REF = Path(
    "/glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run/"
    "g.e21.SUMO_CAM6_v03.cam.h1.1980-01-01-21600.nc"
)


# -----------------------------------------------------------------------------
# Grids
# -----------------------------------------------------------------------------


def test_cam6_grid_shape_and_orientation():
    g = cam6_grid_from_h1(H1_REF)
    assert g.shape == (cfg.CAM6_NLAT, cfg.CAM6_NLON)
    # CAM6 FV f09 is ASCENDING in lat.
    assert g.lat_direction == "ascending"
    # FV lats include the poles.
    assert g.lat[0] == -90.0
    assert g.lat[-1] == 90.0
    # FV lons start at 0, regularly spaced by 1.25 deg.
    assert g.lon[0] == 0.0
    assert np.isclose(g.lon[1] - g.lon[0], 360.0 / cfg.CAM6_NLON)


def test_graphcast_grid_matches_era5_convention():
    g = graphcast_grid()
    assert g.shape == (cfg.GC_NLAT, cfg.GC_NLON)
    assert g.lat_direction == "descending"
    assert g.lat[0] == 90.0
    assert g.lat[-1] == -90.0
    assert g.lon[0] == 0.0
    assert np.isclose(g.lon[1] - g.lon[0], 0.25)
    assert np.isclose(g.lon[-1], 359.75)


def test_grids_validate_pass():
    cam = cam6_grid_from_h1(H1_REF)
    gc = graphcast_grid()
    cam.validate()
    gc.validate()


# -----------------------------------------------------------------------------
# Hybrid coordinate
# -----------------------------------------------------------------------------


def test_hybrid_coord_load_and_validate():
    hc = hybrid_coord_from_h1(H1_REF)
    assert hc.positive_down is True
    assert hc.hyam.shape == (cfg.CAM6_NLEV,)
    assert hc.hybm.shape == (cfg.CAM6_NLEV,)
    # CAM6 top: pure pressure (hybm[0] = 0).
    assert hc.hybm[0] == 0.0
    # CAM6 bottom: nearly pure sigma.
    assert hc.hybm[-1] > 0.99
    # P0 sanity.
    assert np.isclose(hc.p0, cfg.CAM6_P0_PA)


def test_midpoint_pressure_formula_known_value():
    """Midpoint pressure at the bottom level for PS=1 atm should be ~p992."""
    hc = hybrid_coord_from_h1(H1_REF)
    ps = np.array(101325.0)  # 1 atm, scalar
    p_mid = midpoint_pressures_np(hc.hyam, hc.hybm, ps)
    assert p_mid.shape == (cfg.CAM6_NLEV,)
    # Bottom CAM6 midpoint should be near surface for sigma~1.
    p_bot = p_mid[-1]
    # hybm[-1] = 0.9926, hyam[-1] = 0 -> p_bot = 0.9926 * 101325 = ~100571
    assert 1.0e5 < p_bot < 1.02e5, f"bottom p_mid = {p_bot}"
    # Top must be small and pressure-only.
    p_top = p_mid[0]
    assert 100.0 < p_top < 1000.0, f"top p_mid = {p_top}"


def test_pressure_monotonic_top_to_bottom():
    """For positive=down ordering, p_mid should be strictly increasing with k."""
    hc = hybrid_coord_from_h1(H1_REF)
    # Use a realistic PS field rather than a scalar.
    ds = xr.open_dataset(H1_REF)
    try:
        ps = ds["PS"].isel(time=0).values
    finally:
        ds.close()
    p_mid = midpoint_pressures_np(hc.hyam, hc.hybm, ps)  # (nlev, nlat, nlon)
    dp = np.diff(p_mid, axis=0)
    assert np.all(dp > 0), "p_mid should be strictly increasing top->bottom"


def test_layer_thickness_sums_to_ps_minus_ptop():
    """sum_k delta_p_k = ps - p_interface_top."""
    hc = hybrid_coord_from_h1(H1_REF)
    ds = xr.open_dataset(H1_REF)
    try:
        ps = ds["PS"].isel(time=0).values
    finally:
        ds.close()
    dp = layer_thickness_np(hc.hyai, hc.hybi, ps)  # (nlev, nlat, nlon)
    pi = interface_pressures_np(hc.hyai, hc.hybi, ps)
    # sum across levels equals ps minus top-of-atmosphere pressure.
    summed = dp.sum(axis=0)
    expected = pi[-1] - pi[0]
    np.testing.assert_allclose(summed, expected, rtol=1e-12)
    # And pi[-1] should equal ps (bottom interface is pure sigma).
    np.testing.assert_allclose(pi[-1], ps, rtol=1e-12)


# Allow running directly: `python regrid/tests/test_grids_and_pressure.py`
if __name__ == "__main__":
    tests = [
        test_cam6_grid_shape_and_orientation,
        test_graphcast_grid_matches_era5_convention,
        test_grids_validate_pass,
        test_hybrid_coord_load_and_validate,
        test_midpoint_pressure_formula_known_value,
        test_pressure_monotonic_top_to_bottom,
        test_layer_thickness_sums_to_ps_minus_ptop,
    ]
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            raise
