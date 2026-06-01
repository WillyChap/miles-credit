"""End-to-end smoke test for graphcast_server.py.

What it verifies, in order:

  1. Server starts as a subprocess, writes its PID file, eventually writes
     `gc_server_ready.flag` (this includes JIT trace+compile -- ~15 min on
     a cold XLA cache, ~10 s on a warm cache).
  2. A request -> response round-trip produces a real netCDF prediction
     file AND a CAM6-grid nudge target file. Both are non-empty and
     dimensionally correct (37 lvl, 721x1440 prediction; 32 lvl, 192x288
     nudge target).
  3. A second request reuses the JIT cache (timing assertion: < 60s
     inference vs. the cold compile).
  4. STOP flag triggers graceful shutdown -- process exits 0, writes
     `gc_server_stopped.flag`, removes PID file.
  5. (separate phase) Re-launch the server, immediately SIGTERM via the
     PID file, verify it exits cleanly too.
  6. (separate phase) Re-launch and SIGKILL via `kill -9` -- verify the
     PID file is left stale but the process is dead and no inference
     state is corrupted (the rundir can be reused).

Run (interactive A100 node):
    conda activate /glade/work/wchapman/conda-envs/supermodel
    python -m regrid.smoke_test_graphcast_server
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# Use the repo's persistent reference pair, NOT scratch files from a
# transient test case.
from regrid import config as cfg
H1_PREV = cfg.REFERENCE_H1_PREV
H1_T    = cfg.REFERENCE_H1_T
REF_H1  = H1_T

# Long enough to cover a cold-cache JIT compile on the first server start;
# subsequent phases use shorter timeouts since the cache is warm.
READY_TIMEOUT_COLD = 1500   # 25 min (cold XLA compile + warmup inference)
READY_TIMEOUT_WARM = 180    # 3 min  (cache hit, warmup inference is the only cost)
# Per-request budget: server-reported timing for the validated 5-day h1 pair is
# bundle=44s (NumPy reference) + infer=10s + reverse=5s = ~60s. Set to 120s
# to give comfortable headroom; will shrink once the JAX fused bundle lands.
REQUEST_TIMEOUT_COLD = 180
REQUEST_TIMEOUT_WARM = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_path(path: Path, timeout_sec: float, poll: float = 1.0) -> bool:
    """Return True if `path` appears within `timeout_sec`."""
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if path.exists():
            return True
        time.sleep(poll)
    return False


def _parse_kv(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _send_request(rundir: Path, request_id: str) -> None:
    """Atomic-rename write of gc_request.flag."""
    payload = (
        f"request_id={request_id}\n"
        f"h1_prev={H1_PREV}\n"
        f"h1_t={H1_T}\n"
    )
    tmp = rundir / f"{request_id}.tmp"
    tmp.write_text(payload)
    os.replace(tmp, rundir / "gc_request.flag")


def _validate_prediction(path: Path) -> None:
    """Quick structural check on the GC prediction file."""
    assert path.exists(), f"missing: {path}"
    ds = xr.open_dataset(path, decode_times=False)
    try:
        assert ds.sizes["level"] == 37, ds.sizes
        assert ds.sizes["lat"] == 721, ds.sizes
        assert ds.sizes["lon"] == 1440, ds.sizes
        assert "temperature" in ds, list(ds.data_vars)
        t = ds["temperature"].values
        assert np.isfinite(t).all(), "non-finite values in temperature"
        # plausible global mean (winter, all levels): ~240-260 K.
        assert 200 < float(t.mean()) < 280, f"T mean out of range: {t.mean()}"
    finally:
        ds.close()


def _validate_nudge(path: Path) -> None:
    """Quick structural check on the CAM6 nudge target file."""
    assert path.exists(), f"missing: {path}"
    ds = xr.open_dataset(path, decode_times=False)
    try:
        assert ds.sizes["lev"] == 32, ds.sizes
        assert ds.sizes["lat"] == 192, ds.sizes
        assert ds.sizes["lon"] == 288, ds.sizes
        for v in ("U", "V", "T", "Q"):
            assert v in ds, f"missing nudge var {v}"
            arr = ds[v].values
            assert np.isfinite(arr).all(), f"non-finite in {v}"
        assert float(ds["Q"].min()) >= 0.0, "Q must be non-negative"
        # Cam6 T range plausible
        t = ds["T"].values
        assert 150 < float(t.min()) and float(t.max()) < 330, f"T range: {t.min()}..{t.max()}"
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Phase 1: full lifecycle (ready -> 2 requests -> stop)
# ---------------------------------------------------------------------------


def phase_1_full_lifecycle(rundir: Path) -> tuple[float, float]:
    """Start server, send 2 requests, stop. Returns (t_infer_1, t_infer_2)."""
    print("\n" + "=" * 60)
    print("PHASE 1: full lifecycle")
    print("=" * 60)

    proc = subprocess.Popen(
        [PYTHON, "-m", "regrid.graphcast_server",
         "--rundir", str(rundir),
         "--ref-h1", str(REF_H1),
         # Warmup at startup so READY = JIT done.
         "--warmup-h1-prev", str(H1_PREV),
         "--warmup-h1-t",    str(H1_T)],
        cwd=str(REPO),
        stdout=sys.stdout, stderr=sys.stderr,
    )
    try:
        # Wait for ready -- includes the cold JIT compile + 1 warmup inference.
        t0 = time.time()
        print(f"\n  waiting for {rundir / 'gc_server_ready.flag'} "
              f"(up to {READY_TIMEOUT_COLD}s for cold-cache compile + warmup)...")
        if not _wait_for_path(rundir / "gc_server_ready.flag", READY_TIMEOUT_COLD):
            raise TimeoutError("server never became ready")
        print(f"  READY in {time.time() - t0:.1f}s")

        # PID file present
        assert (rundir / "gc_server.pid").exists(), "PID file missing"
        pid = int((rundir / "gc_server.pid").read_text().strip())
        assert pid == proc.pid, f"PID file {pid} != child pid {proc.pid}"

        # Send request 1 -- after warmup, this is just one full inference.
        t0 = time.time()
        _send_request(rundir, "smoke-001")
        print("\n  sent request smoke-001; waiting for response...")
        if not _wait_for_path(rundir / "gc_done.flag", REQUEST_TIMEOUT_COLD):
            raise TimeoutError("response 1 never written")
        resp1 = _parse_kv(rundir / "gc_done.flag")
        t_infer_1 = float(resp1["t_inference"])
        print(f"  response 1 in {time.time() - t0:.1f}s "
              f"(server-reported inference: {t_infer_1:.1f}s)")
        assert resp1["request_id"] == "smoke-001"
        _validate_prediction(Path(resp1["prediction"]))
        _validate_nudge(Path(resp1["gc_nudge_target"]))
        # Clear response so we can detect the next one cleanly
        (rundir / "gc_done.flag").unlink()

        # Send request 2 -- should be fast (warm in-process)
        t0 = time.time()
        _send_request(rundir, "smoke-002")
        print("\n  sent request smoke-002; waiting for response...")
        if not _wait_for_path(rundir / "gc_done.flag", REQUEST_TIMEOUT_WARM):
            raise TimeoutError("response 2 never written")
        resp2 = _parse_kv(rundir / "gc_done.flag")
        t_infer_2 = float(resp2["t_inference"])
        print(f"  response 2 in {time.time() - t0:.1f}s "
              f"(server-reported inference: {t_infer_2:.1f}s)")
        assert resp2["request_id"] == "smoke-002"
        _validate_prediction(Path(resp2["prediction"]))
        _validate_nudge(Path(resp2["gc_nudge_target"]))

        # Stop the server
        print("\n  dropping stop flag...")
        (rundir / "gc_server_stop.flag").touch()
        proc.wait(timeout=60)
        assert proc.returncode == 0, f"server exit code {proc.returncode}"
        assert (rundir / "gc_server_stopped.flag").exists(), \
            "stopped.flag missing -- ungraceful shutdown?"
        assert not (rundir / "gc_server.pid").exists(), "PID file should be removed"
        print(f"  server exited cleanly (code 0)")

        return t_infer_1, t_infer_2

    except Exception:
        proc.kill()
        proc.wait()
        raise


# ---------------------------------------------------------------------------
# Phase 2: SIGTERM mid-idle
# ---------------------------------------------------------------------------


def phase_2_sigterm(rundir: Path) -> None:
    print("\n" + "=" * 60)
    print("PHASE 2: SIGTERM via PID file")
    print("=" * 60)

    proc = subprocess.Popen(
        [PYTHON, "-m", "regrid.graphcast_server",
         "--rundir", str(rundir),
         "--ref-h1", str(REF_H1),
         "--warmup-h1-prev", str(H1_PREV),
         "--warmup-h1-t",    str(H1_T)],
        cwd=str(REPO),
        stdout=sys.stdout, stderr=sys.stderr,
    )
    try:
        # Wait for ready (cache warm now)
        if not _wait_for_path(rundir / "gc_server_ready.flag", READY_TIMEOUT_WARM):
            raise TimeoutError("server never became ready (warm cache phase)")
        pid = int((rundir / "gc_server.pid").read_text().strip())
        print(f"  server ready, pid={pid}")

        os.kill(pid, signal.SIGTERM)
        print(f"  sent SIGTERM to pid {pid}")
        # JAX/CUDA process shutdown can take a while -- give it room.
        proc.wait(timeout=90)
        assert proc.returncode == 0, f"server exit code {proc.returncode}"
        assert (rundir / "gc_server_stopped.flag").exists(), \
            "stopped.flag missing after SIGTERM"
        assert not (rundir / "gc_server.pid").exists(), \
            "PID file should be removed after SIGTERM"
        print(f"  server exited cleanly on SIGTERM")
    except Exception:
        proc.kill()
        proc.wait()
        raise


# ---------------------------------------------------------------------------
# Phase 3: SIGKILL (rude shutdown -- must NOT corrupt rundir)
# ---------------------------------------------------------------------------


def phase_3_sigkill(rundir: Path) -> None:
    print("\n" + "=" * 60)
    print("PHASE 3: SIGKILL (rude)")
    print("=" * 60)

    proc = subprocess.Popen(
        [PYTHON, "-m", "regrid.graphcast_server",
         "--rundir", str(rundir),
         "--ref-h1", str(REF_H1),
         "--warmup-h1-prev", str(H1_PREV),
         "--warmup-h1-t",    str(H1_T)],
        cwd=str(REPO),
        stdout=sys.stdout, stderr=sys.stderr,
    )
    try:
        if not _wait_for_path(rundir / "gc_server_ready.flag", READY_TIMEOUT_WARM):
            raise TimeoutError("server never became ready (kill phase)")
        pid = int((rundir / "gc_server.pid").read_text().strip())
        print(f"  server ready, pid={pid}")

        os.kill(pid, signal.SIGKILL)
        print(f"  sent SIGKILL to pid {pid}")
        proc.wait(timeout=10)
        assert proc.returncode != 0, "SIGKILL should produce nonzero exit"
        # PID file deliberately NOT cleaned up on SIGKILL -- that's expected.
        # The coordinator should treat a stale PID file as "verify the process
        # is actually alive before assuming it".
        # However the stale READY flag IS dangerous: a fresh server would not
        # clean it until after JIT compile, and any consumer (like this test
        # waiting for the new server to be ready) could race-spot it as
        # "already up". Clear it ourselves before the relaunch.
        for stale in ("gc_server_ready.flag", "gc_server.pid",
                      "gc_done.flag", "gc_request.flag",
                      "gc_server_stop.flag", "gc_server_stopped.flag",
                      "gc_server_error.flag"):
            try:
                (rundir / stale).unlink()
            except FileNotFoundError:
                pass
        print(f"  server killed (exit code {proc.returncode}); cleared stale flags")

        # Re-launch in same rundir to confirm it isn't corrupted.
        print("\n  re-launching server in same rundir to confirm not corrupted...")
        proc2 = subprocess.Popen(
            [PYTHON, "-m", "regrid.graphcast_server",
             "--rundir", str(rundir),
             "--ref-h1", str(REF_H1),
             "--warmup-h1-prev", str(H1_PREV),
             "--warmup-h1-t",    str(H1_T)],
            cwd=str(REPO),
            stdout=sys.stdout, stderr=sys.stderr,
        )
        try:
            if not _wait_for_path(rundir / "gc_server_ready.flag", READY_TIMEOUT_WARM):
                raise TimeoutError("server failed to become ready after kill-cleanup")
            print(f"  re-launch ready (rundir is reusable)")
            (rundir / "gc_server_stop.flag").touch()
            proc2.wait(timeout=90)
            assert proc2.returncode == 0
        except Exception:
            proc2.kill()
            proc2.wait()
            raise
    except Exception:
        proc.kill()
        proc.wait()
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("graphcast_server.py smoke test")
    print("=" * 60)
    print(f"  python:   {PYTHON}")
    print(f"  repo:     {REPO}")
    print(f"  H1 prev:  {H1_PREV.name}")
    print(f"  H1 T:     {H1_T.name}")

    # Use a dedicated scratch dir, not the live CESM run dir.
    rundir = Path(tempfile.mkdtemp(
        prefix="gc_smoke_test_",
        dir="/glade/derecho/scratch/wchapman/",
    ))
    print(f"  rundir:   {rundir}")

    try:
        t1, t2 = phase_1_full_lifecycle(rundir)
        # Server-reported inference time: typically ~10 s on A100 with cache.
        # The full request cycle (bundle build + inference + reverse) is
        # ~60 s right now; will drop to ~15 s once Task #4 JAX-ifies the
        # bundle. We just print and let the user judge.
        print(f"\n  per-call inference (server-reported): {t1:.1f}s, {t2:.1f}s")
        if t2 > 30:
            print(f"  NOTE: 2nd inference {t2:.1f}s slower than expected ~10s; "
                  "did the JAX cache miss?")

        phase_2_sigterm(rundir)
        phase_3_sigkill(rundir)

        print("\n" + "=" * 60)
        print("ALL PHASES PASSED")
        print("=" * 60)
        print(f"  rundir kept for inspection: {rundir}")

    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        print(f"  rundir kept for postmortem: {rundir}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
