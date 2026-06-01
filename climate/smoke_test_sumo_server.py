#!/usr/bin/env python
"""
smoke_test_sumo_server.py
--------------------------
Tests for camulator_sumo_server.py SUMO exchange protocol.

Tests (run in order):
  1. Unit: write_sumo_state_nc / read_sumo_combined_nc round-trip
  2. Unit: apply_sumo_nudging math (alpha = 0, 0.5, 1.0; level mismatch)
  3. Smoke: 3-step coupling loop with subprocess server + mock CESM + mock coordinator

Usage:
  conda activate /glade/work/wchapman/conda-envs/credit-coupling-ud
  cd /glade/work/wchapman/Roman_Coupling/camulator_sumo/climate
  python smoke_test_sumo_server.py
"""

import os
import sys
import time
import tempfile
import threading
import subprocess
from pathlib import Path

import numpy as np
import netCDF4 as nc
import torch

# ── paths ──────────────────────────────────────────────────────────────────
CLIMATE_DIR   = Path(__file__).parent
CONFIG        = CLIMATE_DIR / "camulator_config.yml"
MODEL_NAME    = "checkpoint.pt00044.pt"
SUMO_IC       = Path(
    "/glade/derecho/scratch/wchapman/CREDIT_runs"
    "/NEW_CLI_JOHN_CASPER_extended_v2/init_times"
    "/sumo_init_camulator_condition_tensor_1980-01-01T00Z.pth"
)

sys.path.insert(0, str(CLIMATE_DIR))

# Import only the module-level helpers — avoids running main()
from camulator_sumo_server import (
    write_sumo_state_nc,
    read_sumo_combined_nc,
    apply_sumo_nudging,
    write_cam_nc,
    write_flag,
    delete_flag,
    T62_NLAT, T62_NLON,
)

# ── colours ────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

_pass = lambda s: print(f"  {GREEN}PASS{RESET}  {s}")
_fail = lambda s: print(f"  {RED}FAIL{RESET}  {s}")
_info = lambda s: print(f"        {s}")


# =============================================================================
# Helpers
# =============================================================================

def make_fake_sst_nc(path, ymd=19800101, tod=0):
    """Write a minimal camulator_sst_in.nc (T62 grid, SST~285 K, ifrac=0)."""
    ngrid = T62_NLAT * T62_NLON
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("ngrid", ngrid)
        for name, val, units in [
            ("sst",   np.full(ngrid, 285.0), "K"),
            ("ifrac", np.zeros(ngrid),       "1"),
        ]:
            v = ds.createVariable(name, "f8", ("ngrid",))
            v[:] = val
            v.units = units
        for name, val in [("ymd", ymd), ("tod", tod)]:
            v = ds.createVariable(name, "i4")
            v[()] = val


def _echo_coordinator(rundir, n_steps, results, perturb=0.0):
    """
    Simulate sumo_coordinator.py:
      - Read sumo_cam_state.nc
      - Optionally add `perturb` to every variable
      - Write sumo_combined_state.nc  (same grid, same ymd/tod)
      - Signal sumo_combine_done.flag
    """
    rundir      = Path(rundir)
    cam_ready   = rundir / "sumo_cam_ready.flag"
    done_flag   = rundir / "sumo_combine_done.flag"
    cam_state   = rundir / "sumo_cam_state.nc"
    combined    = rundir / "sumo_combined_state.nc"

    results["coordinator_steps"] = []

    for step in range(n_steps):
        # Wait for CAMulator to write cam_ready
        deadline = time.time() + 300
        while not cam_ready.exists():
            time.sleep(0.1)
            if time.time() > deadline:
                results["coordinator_error"] = f"Timeout step {step}: cam_ready.flag"
                return

        # Read state
        with nc.Dataset(str(cam_state), "r") as ds:
            lats = ds["lat"][:].data.astype(np.float32)
            lons = ds["lon"][:].data.astype(np.float32)
            ymd  = int(ds["ymd"][()])
            tod  = int(ds["tod"][()])
            var_arrays = {}
            for vname in ds.variables:
                if vname not in ("lat", "lon", "ymd", "tod"):
                    var_arrays[vname] = ds[vname][:].data.astype(np.float32)

        nlev = next(iter(var_arrays.values())).shape[0]
        results["coordinator_steps"].append({
            "step": step, "ymd": ymd, "tod": tod,
            "vars": list(var_arrays.keys()),
            "U_mean": float(var_arrays.get("U", np.array([0])).mean()),
        })

        # Clean up cam_ready flag
        delete_flag(cam_ready)

        # Write combined state (echo + optional perturbation)
        combined_arrays = {k: v + perturb for k, v in var_arrays.items()}
        write_sumo_state_nc(combined, combined_arrays, lats, lons,
                            list(range(nlev)), ymd, tod)
        write_flag(done_flag)

    results["coordinator_ok"] = True


def _mock_cesm(rundir, n_steps, results):
    """
    Simulate CESM coupling loop:
      - Wait for camulator_server_ready.flag
      - For each step: write SST → write go.flag → wait for done.flag → verify cam_out.nc
    """
    rundir       = Path(rundir)
    ready_flag   = rundir / "camulator_server_ready.flag"
    go_flag      = rundir / "camulator_go.flag"
    done_flag    = rundir / "camulator_done.flag"
    sst_file     = rundir / "camulator_sst_in.nc"
    cam_out_file = rundir / "camulator_cam_out.nc"

    results["cesm_steps"] = []

    # Wait for server to be ready (model load + JIT trace can take ~60s)
    deadline = time.time() + 300
    while not ready_flag.exists():
        time.sleep(0.5)
        if time.time() > deadline:
            results["cesm_error"] = "Timeout waiting for camulator_server_ready.flag"
            return

    _info("  [mock CESM] server ready — starting coupling steps")

    # Dates: 1980-01-01 00Z, 06Z, 12Z
    dates = [(19800101, i * 21600) for i in range(n_steps)]

    for step, (ymd, tod) in enumerate(dates):
        make_fake_sst_nc(sst_file, ymd=ymd, tod=tod)
        write_flag(go_flag)
        _info(f"  [mock CESM] step {step}: wrote go.flag  ymd={ymd} tod={tod}")

        # Wait for done.flag
        deadline = time.time() + 300
        while not done_flag.exists():
            time.sleep(0.1)
            if time.time() > deadline:
                results["cesm_error"] = f"Timeout step {step}: done.flag"
                return
        delete_flag(done_flag)

        # Verify cam_out.nc has all expected variables with sensible values
        step_ok  = True
        step_msgs = []
        expected = ["u10", "v10", "tbot", "zbot", "tref", "qbot",
                    "pbot", "fsds", "fsus", "flds", "flus", "prect"]
        with nc.Dataset(str(cam_out_file), "r") as ds:
            for var in expected:
                if var not in ds.variables:
                    step_ok = False
                    step_msgs.append(f"missing {var}")
                    continue
                arr = ds[var][:].data
                if not np.isfinite(arr).all():
                    step_ok = False
                    step_msgs.append(f"{var} has non-finite values")

        results["cesm_steps"].append({
            "step": step, "ymd": ymd, "tod": tod,
            "ok": step_ok, "msgs": step_msgs,
        })

    results["cesm_ok"] = True


# =============================================================================
# Unit test 1 — SUMO NC I/O round-trip
# =============================================================================

def test_sumo_nc_roundtrip():
    print("\n[Unit 1] SUMO NC I/O round-trip")
    with tempfile.TemporaryDirectory() as tmpdir:
        path_out = Path(tmpdir) / "sumo_state.nc"
        path_comb = Path(tmpdir) / "sumo_combined.nc"

        # Build fake state: U (32 lev), V (32 lev), T (1 lev for PS proxy)
        nlev = 32
        nlat, nlon = 4, 8
        lats = np.linspace(-90, 90, nlat).astype(np.float32)
        lons = np.linspace(0, 360, nlon, endpoint=False).astype(np.float32)
        rng = np.random.default_rng(42)
        var_arrays = {
            "U": rng.standard_normal((nlev, nlat, nlon)).astype(np.float32) * 10,
            "V": rng.standard_normal((nlev, nlat, nlon)).astype(np.float32) * 8,
        }

        # Write
        write_sumo_state_nc(path_out, var_arrays, lats, lons,
                            list(range(nlev)), ymd=19800101, tod=0)
        # Echo as combined
        import shutil
        shutil.copy(path_out, path_comb)

        # Read back
        read_back, ymd_r, tod_r = read_sumo_combined_nc(path_comb)

        ok = True
        for k in ("U", "V"):
            if k not in read_back:
                _fail(f"Variable '{k}' missing after round-trip")
                ok = False
            elif not np.allclose(var_arrays[k], read_back[k], atol=1e-5):
                _fail(f"Variable '{k}' values differ after round-trip  "
                      f"max_err={np.abs(var_arrays[k]-read_back[k]).max():.2e}")
                ok = False
        if ymd_r != 19800101 or tod_r != 0:
            _fail(f"ymd/tod mismatch: got ({ymd_r},{tod_r})")
            ok = False
        if ok:
            _pass("write_sumo_state_nc → read_sumo_combined_nc: values preserved")
    return ok


# =============================================================================
# Unit test 2 — apply_sumo_nudging math
# =============================================================================

def test_apply_sumo_nudging():
    print("\n[Unit 2] apply_sumo_nudging math")
    all_ok = True
    nlev, nlat, nlon = 4, 6, 8
    n_u_ch = nlev
    # prediction tensor: (1, 2*nlev, 1, nlat, nlon), U in channels 0:nlev, V in nlev:2*nlev
    n_ch = 2 * nlev

    # Minimal mock accessor
    class MockAccessor:
        def get_state_var(self, tensor, var):
            sl = slice(0, nlev) if var == "U" else slice(nlev, 2 * nlev)
            return tensor[:, sl, :, :, :]
        def set_state_var(self, tensor, var, value):
            sl = slice(0, nlev) if var == "U" else slice(nlev, 2 * nlev)
            tensor[:, sl, :, :, :] = value

    # Mock state_transformer — identity normalization (mean=0, std=1)
    class MockTransformer:
        mean_tensors = {"U": np.float32(0.0), "V": np.float32(0.0)}
        std_tensors  = {"U": np.float32(1.0), "V": np.float32(1.0)}

    accessor = MockAccessor()
    transformer = MockTransformer()
    device = torch.device("cpu")

    pred_val = 3.0
    comb_val = 7.0
    combined_arrays = {
        "U": np.full((nlev, nlat, nlon), comb_val, dtype=np.float32),
        "V": np.full((nlev, nlat, nlon), comb_val, dtype=np.float32),
    }

    for alpha, expected_u in [(1.0, comb_val), (0.0, pred_val), (0.5, 0.5*(pred_val+comb_val))]:
        pred = torch.full((1, n_ch, 1, nlat, nlon), pred_val)
        result = apply_sumo_nudging(
            pred, combined_arrays, ["U", "V"],
            alpha, accessor, transformer, cam_flip=False, device=device
        )
        u_out = accessor.get_state_var(result, "U").mean().item()
        tol = 1e-5
        if abs(u_out - expected_u) < tol:
            _pass(f"α={alpha:.1f}: U = {u_out:.2f}  (expected {expected_u:.2f})")
        else:
            _fail(f"α={alpha:.1f}: U = {u_out:.4f}  (expected {expected_u:.2f})")
            all_ok = False

    # Level mismatch: combined has fewer levels than prediction
    pred = torch.full((1, n_ch, 1, nlat, nlon), pred_val)
    short_combined = {
        "U": np.full((nlev - 1, nlat, nlon), comb_val, dtype=np.float32),
    }
    try:
        result = apply_sumo_nudging(
            pred, short_combined, ["U"],
            1.0, accessor, transformer, cam_flip=False, device=device
        )
        _pass("Level-mismatch guard: no exception raised")
    except Exception as e:
        _fail(f"Level-mismatch guard raised: {e}")
        all_ok = False

    # cam_flip=True: combined is in ascending lat, prediction in descending — should still work
    pred = torch.zeros(1, n_ch, 1, nlat, nlon)
    # U in combined = ascending gradient 0..nlat-1
    ascending = np.arange(nlat, dtype=np.float32)[None, :, None] * np.ones((nlev, nlat, nlon))
    result = apply_sumo_nudging(
        pred, {"U": ascending.astype(np.float32)},
        ["U"], 1.0, accessor, transformer, cam_flip=True, device=device
    )
    u_out = accessor.get_state_var(result, "U")[0, 0, 0, :, 0].tolist()
    expected_flip = list(reversed(range(nlat)))
    if u_out == [float(x) for x in expected_flip]:
        _pass("cam_flip=True: lat axis correctly inverted before nudging")
    else:
        _fail(f"cam_flip=True: expected {expected_flip}, got {u_out}")
        all_ok = False

    return all_ok


# =============================================================================
# Smoke test — full 3-step coupling loop
# =============================================================================

def test_smoke_coupling(n_steps=3, device="cpu"):
    print(f"\n[Smoke] {n_steps}-step coupling loop (server + mock CESM + mock coordinator)")

    if not CONFIG.exists():
        _fail(f"Config not found: {CONFIG}")
        return False
    if not SUMO_IC.exists():
        _fail(f"SUMO IC not found: {SUMO_IC}")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        rundir = Path(tmpdir)
        _info(f"rundir: {rundir}")

        results = {}
        server_proc = None

        try:
            # Start server subprocess
            cmd = [
                sys.executable,
                str(CLIMATE_DIR / "camulator_sumo_server.py"),
                "--config",     str(CONFIG),
                "--model_name", MODEL_NAME,
                "--rundir",     str(rundir),
                "--init_cond",  str(SUMO_IC),
                "--sumo",
                "--sumo_vars",  "U", "V",
                "--sumo_tau",   "6.0",
                "--sumo_timeout", "120",
                "--device",     device,
            ]
            log_path = rundir / "server.log"
            with open(log_path, "w") as log_f:
                server_proc = subprocess.Popen(
                    cmd,
                    cwd=str(CLIMATE_DIR),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
            _info(f"Server PID {server_proc.pid} started  log: {log_path}")

            # Launch mock coordinator and mock CESM as threads
            t_coord = threading.Thread(
                target=_echo_coordinator,
                args=(rundir, n_steps, results),
                daemon=True,
            )
            t_cesm = threading.Thread(
                target=_mock_cesm,
                args=(rundir, n_steps, results),
                daemon=True,
            )
            t_coord.start()
            t_cesm.start()

            # Wait for both threads (generous timeout for model loading)
            t_cesm.join(timeout=600)
            t_coord.join(timeout=60)

        finally:
            if server_proc is not None and server_proc.poll() is None:
                server_proc.terminate()
                server_proc.wait(timeout=10)
            _info(f"Server exit code: {server_proc.returncode if server_proc else 'N/A'}")

        # ── report ───────────────────────────────────────────────────────────
        all_ok = True

        if "cesm_error" in results:
            _fail(f"CESM thread error: {results['cesm_error']}")
            all_ok = False
        if "coordinator_error" in results:
            _fail(f"Coordinator thread error: {results['coordinator_error']}")
            all_ok = False

        cesm_steps = results.get("cesm_steps", [])
        coord_steps = results.get("coordinator_steps", [])

        if len(cesm_steps) < n_steps:
            _fail(f"Only {len(cesm_steps)}/{n_steps} CESM steps completed")
            all_ok = False
        else:
            _pass(f"All {n_steps} CESM steps completed")

        if len(coord_steps) < n_steps:
            _fail(f"Only {len(coord_steps)}/{n_steps} coordinator exchanges completed")
            all_ok = False
        else:
            _pass(f"All {n_steps} SUMO coordinator exchanges completed")

        for s in cesm_steps:
            label = f"step {s['step']} (ymd={s['ymd']} tod={s['tod']})"
            if s["ok"]:
                _pass(f"cam_out.nc valid: {label}")
            else:
                _fail(f"cam_out.nc problems at {label}: {s['msgs']}")
                all_ok = False

        for s in coord_steps:
            _info(f"  Coordinator step {s['step']}: vars={s['vars']}  "
                  f"U_mean={s['U_mean']:.4f} m/s")

        if results.get("coordinator_ok") and results.get("cesm_ok"):
            _pass("Both mock threads completed cleanly")
        else:
            _fail(f"Thread completion flags: cesm_ok={results.get('cesm_ok')}  "
                  f"coordinator_ok={results.get('coordinator_ok')}")
            all_ok = False

        # Show last few lines of server log for context
        log_path = Path(tmpdir) / "server.log"
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            print(f"\n  --- server log (last {min(20, len(lines))} lines) ---")
            for line in lines[-20:]:
                print(f"  {line}")

        return all_ok


# =============================================================================
# Entry point
# =============================================================================

def main():
    print("=" * 65)
    print("SUMO server test suite")
    print("=" * 65)

    results = {}

    results["io_roundtrip"]   = test_sumo_nc_roundtrip()
    results["nudging_math"]   = test_apply_sumo_nudging()

    print("\n" + "-" * 65)
    print("Running smoke test (loads model — expect ~60s startup) ...")
    results["smoke_coupling"] = test_smoke_coupling(n_steps=3, device="cpu")

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    all_passed = True
    for name, ok in results.items():
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")
        if not ok:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
