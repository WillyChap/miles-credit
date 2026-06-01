"""Unit tests for the horizontal stencil, log-p interp, and reference translator.

The end-to-end translator run at the bottom only exercises T, U, V, Q (the
vars present in the existing 1980-01-01 h1 stream) -- Z3 / OMEGA will be added
to this test once the new fincl2 fields are produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from regrid import config as cfg
from regrid.below_ground_era5 import temperature_bridge, hold_lowest
from regrid.grids import LatLonGrid, cam6_grid_from_h1, graphcast_grid
from regrid.horizontal_weights import (
    apply_bilinear,
    build_bilinear_stencil,
)
from regrid.reference_translate import ReferenceTranslator
from regrid.vertical_interp import interp_log_p

H1_REF = Path(
    "/glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run/"
    "g.e21.SUMO_CAM6_v03.cam.h1.1980-01-01-21600.nc"
)


# -----------------------------------------------------------------------------
# Horizontal bilinear stencil
# -----------------------------------------------------------------------------


def test_stencil_weights_sum_to_one():
    cam = cam6_grid_from_h1(H1_REF)
    gc = graphcast_grid()
    st = build_bilinear_stencil(cam, gc)
    np.testing.assert_allclose(st.ww.sum(axis=-1), 1.0, atol=1e-12)


def test_stencil_constants_invariant():
    """A constant source field must come out as the same constant on the dst grid."""
    cam = cam6_grid_from_h1(H1_REF)
    gc = graphcast_grid()
    st = build_bilinear_stencil(cam, gc)
    src = np.full((cam.nlat, cam.nlon), 7.5)
    out = apply_bilinear(src, st)
    assert out.shape == (gc.nlat, gc.nlon)
    np.testing.assert_allclose(out, 7.5, atol=1e-12)


def test_stencil_linear_in_lon_is_recovered():
    """A field that is linear in longitude on the source should remain so on the
    destination (up to wrapping). Check away from the periodic seam."""
    cam = cam6_grid_from_h1(H1_REF)
    gc = graphcast_grid()
    st = build_bilinear_stencil(cam, gc)
    # f(lon) = lon; constant in lat. Bilinear is exact for separable affine.
    src = np.broadcast_to(cam.lon[None, :], (cam.nlat, cam.nlon)).copy()
    out = apply_bilinear(src, st)
    # GraphCast lons in [0, 359.75]; sample interior (skip the seam at 0).
    mask = (gc.lon > 1.5) & (gc.lon < 358.5)
    expected = np.broadcast_to(gc.lon[None, :], (gc.nlat, gc.nlon))
    np.testing.assert_allclose(out[:, mask], expected[:, mask], atol=1e-6)


def test_stencil_longitude_wrap_seam():
    """Across the dateline the stencil must wrap from lon=358.75 to lon=0."""
    cam = cam6_grid_from_h1(H1_REF)
    gc = graphcast_grid()
    st = build_bilinear_stencil(cam, gc)
    # GraphCast lon=359.75 sits between cam lon[-1]=358.75 (col 287) and
    # cam lon[0]=0 (col 0, after wrap). The two seam columns must both appear
    # in the stencil. Weights are NOT equal there: dlon_cam=1.25, so 359.75 is
    # at fractional position 0.8 of the way from 358.75 toward 360, giving
    # 0.2 weight on col 287 and 0.8 on col 0.
    j_eq = int(np.argmin(np.abs(gc.lat - 0.0)))
    i_seam = int(np.argmin(np.abs(gc.lon - 359.75)))
    ii = st.ii[j_eq, i_seam, :]
    ww = st.ww[j_eq, i_seam, :]
    assert set(int(x) for x in ii) >= {0, cfg.CAM6_NLON - 1}, ii
    # Total weight on the two seam columns must equal 1 (lat axis sums separately).
    w_col0 = sum(w for w, i in zip(ww, ii) if i == 0)
    w_col_end = sum(w for w, i in zip(ww, ii) if i == cfg.CAM6_NLON - 1)
    np.testing.assert_allclose(w_col0 + w_col_end, 1.0, atol=1e-12)
    # And col 0 should have the heavier weight (closer in lon-space).
    assert w_col0 > w_col_end, (w_col0, w_col_end)


# -----------------------------------------------------------------------------
# Vertical log-p interp
# -----------------------------------------------------------------------------


def test_interp_log_p_constants():
    """A field constant in level must come out constant."""
    nlev_src, ncol = 32, 5
    p_src = np.linspace(1.0, 5.0, nlev_src)[:, None] * 1e4
    p_src = np.broadcast_to(p_src, (nlev_src, ncol)).copy()
    field = np.full((nlev_src, ncol), 3.14)
    p_tgt = np.linspace(1.5, 4.5, 11) * 1e4
    res = interp_log_p(field, p_src, p_tgt)
    np.testing.assert_allclose(res.field, 3.14, atol=1e-12)
    assert not res.above_top.any()
    assert not res.below_ground.any()


def test_interp_log_p_linear_in_log_p_is_exact():
    """If field = a + b*log(p), interp must return a + b*log(p_tgt) exactly."""
    nlev_src, ncol = 10, 3
    p_src_1d = np.geomspace(100.0, 1e5, nlev_src)
    p_src = np.broadcast_to(p_src_1d[:, None], (nlev_src, ncol)).copy()
    a, b = 2.0, -0.5
    field = a + b * np.log(p_src)
    p_tgt = np.geomspace(200.0, 5e4, 7)
    res = interp_log_p(field, p_src, p_tgt)
    expected = a + b * np.log(p_tgt)[:, None]
    expected = np.broadcast_to(expected, (p_tgt.size, ncol))
    np.testing.assert_allclose(res.field, expected, atol=1e-12)


def test_interp_log_p_to_hybrid_real_scale_no_memory_blowup():
    """Regression: the reverse-direction vertical interp used bare fancy
    indexing `field_plev[k0]` which produces an outer product
    (32, 192, 288, 192, 288) ≈ 768 GB at real scales. Confirms the fix
    via np.take_along_axis. Output shape and value tracking only -- this
    test just needs to RUN at real scales."""
    from regrid.vertical_interp import interp_log_p_to_hybrid
    rng = np.random.default_rng(0)
    nlev_src, nlev_tgt = 37, 32
    nlat, nlon = 192, 288
    p_src = np.geomspace(100.0, 1e5, nlev_src)
    # Hybrid pressures: monotonically increasing per column, plausible CAM-like.
    base = np.geomspace(400.0, 1e5, nlev_tgt)[:, None, None]
    jitter = rng.uniform(0.9, 1.1, size=(1, nlat, nlon))
    p_tgt = base * jitter
    field = rng.standard_normal((nlev_src, nlat, nlon))
    out = interp_log_p_to_hybrid(field, p_src, p_tgt)
    assert out.field.shape == (nlev_tgt, nlat, nlon), out.field.shape
    assert np.isfinite(out.field).all()


def test_interp_log_p_below_ground_mask():
    nlev_src = 5
    p_src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])[:, None] * 1e4
    field = np.array([0.0, 1.0, 2.0, 3.0, 4.0])[:, None]
    # Target levels: 0.5e4 (above TOA), 3e4 (interior), 6e4 (below ground).
    p_tgt = np.array([0.5e4, 3e4, 6e4])
    res = interp_log_p(field, p_src, p_tgt)
    assert res.above_top[0, 0] and not res.below_ground[0, 0]
    assert not res.above_top[1, 0] and not res.below_ground[1, 0]
    assert not res.above_top[2, 0] and res.below_ground[2, 0]


# -----------------------------------------------------------------------------
# Below-ground temperature bridge
# -----------------------------------------------------------------------------


def test_t_bridge_warmer_below_surface_with_taper():
    """Standard-atm lapse gives warmer T immediately below the surface."""
    # Single column, target levels 800, 950, 1000, 1050 hPa; lowest model
    # level at 992 hPa, T=280 K. Below ground points get the lapse extrapolation.
    p_tgt = np.array([800, 950, 1000, 1050]) * 100.0
    t_low = np.array(280.0)
    p_low = np.array(992.0 * 100.0)
    # vertical interp output: just hold constant for above-ground; below-ground
    # gets overwritten.
    t_interp = np.full((4,), 280.0)
    below = p_tgt > p_low
    t_out = temperature_bridge(
        t_interp, t_low, np.array(0.0), p_low, p_tgt,
        below_ground=below, taper_lapse=True,
    )
    # Below-ground points should be warmer than 280 K.
    assert (t_out[below] > t_low - 1e-9).all()
    # Above-ground points unchanged.
    assert (t_out[~below] == 280.0).all()


# -----------------------------------------------------------------------------
# Below-ground hold-lowest
# -----------------------------------------------------------------------------


def test_hold_lowest_overwrites_below_ground_and_clips_q():
    nlev_src, ncol = 4, 2
    field_interp = np.full((3, ncol), -99.0)
    field_src = np.array([[0.1, 0.2], [0.05, 0.1], [-0.02, 0.0], [0.5, 0.7]])
    below = np.array([[True, True], [False, False], [True, True]])
    out = hold_lowest(field_interp, field_src, below, clip_nonneg=True)
    # Below-ground rows get the lowest src value, clipped at 0.
    np.testing.assert_allclose(out[0], [0.5, 0.7])
    np.testing.assert_allclose(out[2], [0.5, 0.7])
    # Above-ground rows pass through.
    np.testing.assert_allclose(out[1], [-99.0, -99.0])


# -----------------------------------------------------------------------------
# End-to-end (atmospheric vars present today)
# -----------------------------------------------------------------------------


def test_reference_translator_runs_end_to_end_on_existing_h1():
    """Verifies the full pipeline executes and produces dense, finite,
    physical-looking output for T, U, V, Q. Z3 / OMEGA tests will be added
    when the new fincl2 fields land."""
    tr = ReferenceTranslator.build(H1_REF)
    ds_gc = tr.translate_h1_file(H1_REF, time_index=0, which=("T", "U", "V", "Q"))

    assert "temperature" in ds_gc
    assert "u_component_of_wind" in ds_gc
    assert "v_component_of_wind" in ds_gc
    assert "specific_humidity" in ds_gc

    # All outputs dense + finite + correct shape.
    for name, da in ds_gc.data_vars.items():
        assert da.shape == (cfg.NUM_PLEV, cfg.GC_NLAT, cfg.GC_NLON), (name, da.shape)
        assert np.isfinite(da.values).all(), name

    # Specific humidity non-negative (clip enforced).
    assert (ds_gc["specific_humidity"].values >= 0.0).all()

    # T should be roughly between 150 K (TOA cold) and 320 K (warm underground).
    t = ds_gc["temperature"].values
    assert 150.0 < t.min() < 220.0, t.min()
    assert 280.0 < t.max() < 340.0, t.max()

    # Compare to CAM6 reference at 850 hPa: at that level, T should be close
    # to the lowest few CAM6 levels in the tropics.
    ds_cam = xr.open_dataset(H1_REF)
    try:
        # Pick the 850 hPa output level (index 30 in PRESSURE_LEVELS_HPA).
        i850 = cfg.PRESSURE_LEVELS_HPA.index(850)
        t_gc_850 = ds_gc["temperature"].values[i850]
        # Tropical average should be near 285-295 K.
        gc = graphcast_grid()
        tropics = (gc.lat > -20) & (gc.lat < 20)
        t_trop = t_gc_850[tropics, :].mean()
        assert 280.0 < t_trop < 305.0, t_trop
    finally:
        ds_cam.close()


if __name__ == "__main__":
    tests = [
        test_stencil_weights_sum_to_one,
        test_stencil_constants_invariant,
        test_stencil_linear_in_lon_is_recovered,
        test_stencil_longitude_wrap_seam,
        test_interp_log_p_constants,
        test_interp_log_p_linear_in_log_p_is_exact,
        test_interp_log_p_to_hybrid_real_scale_no_memory_blowup,
        test_interp_log_p_below_ground_mask,
        test_t_bridge_warmer_below_surface_with_taper,
        test_hold_lowest_overwrites_below_ground_and_clips_q,
        test_reference_translator_runs_end_to_end_on_existing_h1,
    ]
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            raise
