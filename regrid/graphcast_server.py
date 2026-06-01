"""Long-lived GraphCast inference server for SUMO/supermodel coupling.

The server is OPTIONAL. For low-frequency testing, the SUMO coordinator can
just shell out to `python -m regrid.run_graphcast_inference` per coupling
step; the JAX persistent cache (~10 s per cached compile) keeps that
practical. Use this server when paying the JIT trace+compile (~10-15 min)
on every coupling step would dominate wall-clock cost.

Lifecycle
=========

Startup:
  - Build forward + reverse translators (precompute horizontal stencils).
  - Load checkpoint + per-level normalization stats.
  - JIT-compile the predictor on a representative input shape. The XLA
    persistent cache is honored, so the second startup with the same
    shapes is fast.
  - Write `gc_server.pid` (PID file -- enables emergency kill).
  - Write `gc_server_ready.flag` -- the coordinator polls this.

Serve loop:
  - Wait for `gc_request.flag` (or `gc_server_stop.flag`).
  - Request format: text file, key=value per line:
        request_id = <opaque echo back in response>
        h1_prev    = <abs path to CAM6 h1 at T-6h>
        h1_t       = <abs path to CAM6 h1 at T>
    `request_id` is echoed in the response so the coordinator can match.
  - On success: write
        gc_prediction.<request_id>.nc        (full GraphCast +6h state)
        gc_nudge_target.<request_id>.nc      (CAM6 hybrid, U/V/T/Q)
    then atomic-write `gc_done.flag` with metadata, then delete the request.
  - On failure: write `gc_server_error.flag` with the exception, delete the
    request (so the coordinator doesn't loop on a bad input), continue.

Shutdown:
  - `gc_server_stop.flag` exists -> exit code 0
  - SIGTERM -> graceful exit (write `gc_server_stopped.flag`, remove pid file)
  - SIGKILL (or `kill -9 $(cat gc_server.pid)`) -> ungraceful but allowed

The server is intentionally single-threaded and synchronous: one request,
one inference, one response. Coordinators serialize requests.
"""

from __future__ import annotations

import argparse
import functools
import os
import signal
import sys
import time
from pathlib import Path

# Persistent XLA cache: set BEFORE importing jax. Same path as the one-shot
# runner so they share the cache.
DEFAULT_JAX_CACHE = "/glade/derecho/scratch/wchapman/jax_compilation_cache"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", DEFAULT_JAX_CACHE)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import numpy as np
import xarray as xr


# ----------------------------------------------------------------------------
# Flag-file names
# ----------------------------------------------------------------------------

REQUEST_FLAG   = "gc_request.flag"
RESPONSE_FLAG  = "gc_done.flag"
ERROR_FLAG     = "gc_server_error.flag"
READY_FLAG     = "gc_server_ready.flag"
STOP_FLAG      = "gc_server_stop.flag"
STOPPED_FLAG   = "gc_server_stopped.flag"
PID_FILE       = "gc_server.pid"


# ----------------------------------------------------------------------------
# Tiny IO helpers
# ----------------------------------------------------------------------------


def _write_kv_atomic(path: Path, fields: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(f"{k}={v}" for k, v in fields.items()) + "\n")
    os.replace(tmp, path)


def _parse_kv(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ----------------------------------------------------------------------------
# SIGTERM / SIGINT handling
# ----------------------------------------------------------------------------


class StopRequested(Exception):
    """Internal sentinel to break the serve loop on signal."""


_stop_requested = False


def _signal_handler(signum, _frame):
    global _stop_requested
    _stop_requested = True
    print(f"[gc-server] signal {signum} received, will exit after current request",
          flush=True)


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def serve(
    rundir: Path, ref_h1: Path, checkpoint: Path, poll_sec: float = 1.0,
    warmup_h1_prev: Path | None = None, warmup_h1_t: Path | None = None,
) -> int:
    """Run the GC inference server. Returns the exit code."""
    rundir = rundir.resolve()
    assert rundir.is_dir(), f"rundir does not exist: {rundir}"

    pid_path     = rundir / PID_FILE
    ready_path   = rundir / READY_FLAG
    request_path = rundir / REQUEST_FLAG
    response_path = rundir / RESPONSE_FLAG
    error_path   = rundir / ERROR_FLAG
    stop_path    = rundir / STOP_FLAG
    stopped_path = rundir / STOPPED_FLAG

    # Clean stale lifecycle flags from any previous run in the same rundir.
    for p in (ready_path, request_path, response_path, error_path,
              stop_path, stopped_path):
        _safe_unlink(p)

    # PID file -- the coordinator can `kill -9 $(cat gc_server.pid)`.
    pid_path.write_text(f"{os.getpid()}\n")

    _install_signal_handlers()

    try:
        # ---- Heavy imports + startup -------------------------------------
        print(f"[gc-server] pid={os.getpid()} rundir={rundir}", flush=True)
        print(f"[gc-server] XLA cache: {os.environ['JAX_COMPILATION_CACHE_DIR']}",
              flush=True)

        # Local imports so import-time failures show in our error path.
        from .graphcast_bundle import build_graphcast_input_bundle
        from .reference_translate import ReferenceTranslator
        from .reverse_translate import ReverseTranslator
        from .run_graphcast_inference import (
            build_predictor_fn,
            load_checkpoint,
            load_stats,
        )

        t0 = time.time()
        fwd_translator = ReferenceTranslator.build(ref_h1)
        rev_translator = ReverseTranslator.build(ref_h1)
        print(f"[gc-server] translators built in {time.time() - t0:.1f}s",
              flush=True)

        t0 = time.time()
        ckpt = load_checkpoint(checkpoint)
        diffs_stddev, mean_by_level, stddev_by_level = load_stats()
        print(f"[gc-server] checkpoint + stats loaded in {time.time() - t0:.1f}s",
              flush=True)

        # JIT-compile is triggered on first inference; XLA cache may make
        # this near-instant. We do not warmup at startup -- that would
        # require building a bundle we don't have yet.
        import jax
        run_forward = build_predictor_fn(
            ckpt.model_config, ckpt.task_config,
            diffs_stddev, mean_by_level, stddev_by_level,
        )

        def with_configs(fn):
            return functools.partial(
                fn, model_config=ckpt.model_config, task_config=ckpt.task_config,
            )

        def with_params(fn):
            return functools.partial(fn, params=ckpt.params, state={})

        def drop_state(fn):
            return lambda **kw: fn(**kw)[0]

        run_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

        # ---- Warmup: pay the JIT compile cost BEFORE signaling READY -----
        # Without this, the first request after READY hits the ~18 min cold
        # XLA compile and looks like a hang. With warmup, READY means
        # "every subsequent request takes ~10 s".
        warmup_seconds = 0.0
        if warmup_h1_prev is not None and warmup_h1_t is not None:
            print(f"[gc-server] warming up with {warmup_h1_prev.name} -> "
                  f"{warmup_h1_t.name}...", flush=True)
            wt0 = time.time()
            warmup_bundle = build_graphcast_input_bundle(
                (warmup_h1_prev, warmup_h1_t), translator=fwd_translator,
            )
            from graphcast import data_utils, rollout
            wi, wt, wf = data_utils.extract_inputs_targets_forcings(
                warmup_bundle,
                input_variables=ckpt.task_config.input_variables,
                target_variables=ckpt.task_config.target_variables,
                forcing_variables=ckpt.task_config.forcing_variables,
                pressure_levels=ckpt.task_config.pressure_levels,
                input_duration=ckpt.task_config.input_duration,
                target_lead_times="6h",
            )
            # Trigger compile + 1 real inference. Result is discarded.
            _ = rollout.chunked_prediction(
                run_jitted,
                rng=jax.random.PRNGKey(0),
                inputs=wi,
                targets_template=wt * np.nan,
                forcings=wf,
            )
            warmup_seconds = time.time() - wt0
            print(f"[gc-server] warmup complete in {warmup_seconds:.1f}s "
                  f"(XLA cache now populated)", flush=True)

        # ---- Ready ---------------------------------------------------------
        _write_kv_atomic(ready_path, {
            "pid": os.getpid(),
            "startup_seconds": f"{time.time() - t0:.2f}",
            "warmup_seconds": f"{warmup_seconds:.2f}",
            "checkpoint": str(checkpoint),
        })
        print(f"[gc-server] READY -- {ready_path}", flush=True)

        # ---- Serve loop ---------------------------------------------------
        served = 0
        while True:
            if _stop_requested or stop_path.exists():
                print(f"[gc-server] stop requested after {served} requests",
                      flush=True)
                _safe_unlink(stop_path)
                break

            if not request_path.exists():
                time.sleep(poll_sec)
                continue

            try:
                req = _parse_kv(request_path)
                request_id = req.get("request_id", f"req{served:05d}")
                h1_prev = Path(req["h1_prev"])
                h1_t = Path(req["h1_t"])
                print(f"[gc-server] request id={request_id}: "
                      f"{h1_prev.name} -> {h1_t.name}", flush=True)

                t0 = time.time()
                bundle = build_graphcast_input_bundle(
                    (h1_prev, h1_t), translator=fwd_translator,
                )
                t_bundle = time.time() - t0

                t0 = time.time()
                from graphcast import data_utils, rollout
                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    bundle,
                    input_variables=ckpt.task_config.input_variables,
                    target_variables=ckpt.task_config.target_variables,
                    forcing_variables=ckpt.task_config.forcing_variables,
                    pressure_levels=ckpt.task_config.pressure_levels,
                    input_duration=ckpt.task_config.input_duration,
                    target_lead_times="6h",
                )
                predictions = rollout.chunked_prediction(
                    run_jitted,
                    rng=jax.random.PRNGKey(0),
                    inputs=inputs,
                    targets_template=targets * np.nan,
                    forcings=forcings,
                )
                t_infer = time.time() - t0

                # Write outputs.
                t0 = time.time()
                pred_path = rundir / f"gc_prediction.{request_id}.nc"
                predictions.to_netcdf(pred_path)

                # PS at coupling time T for hybrid pressure.
                h1_ds = xr.open_dataset(h1_t)
                try:
                    ps_cam = np.asarray(h1_ds["PS"].isel(time=0).values,
                                        dtype=np.float64)
                finally:
                    h1_ds.close()
                gc_targets = rev_translator.translate_prediction(predictions, ps_cam)

                from .reverse_translate import write_nudge_file
                nudge_path = rundir / f"gc_nudge_target.{request_id}.nc"
                write_nudge_file(gc_targets, h1_t, nudge_path)
                t_reverse = time.time() - t0

                # Response flag last, so the coordinator only sees it once
                # both files are on disk.
                _write_kv_atomic(response_path, {
                    "request_id": request_id,
                    "prediction": str(pred_path),
                    "gc_nudge_target": str(nudge_path),
                    "t_bundle":  f"{t_bundle:.2f}",
                    "t_inference": f"{t_infer:.2f}",
                    "t_reverse": f"{t_reverse:.2f}",
                })
                _safe_unlink(request_path)
                served += 1
                print(f"[gc-server] served #{served} ({request_id}): "
                      f"bundle={t_bundle:.1f}s infer={t_infer:.1f}s "
                      f"reverse={t_reverse:.1f}s",
                      flush=True)

            except Exception as exc:
                _write_kv_atomic(error_path, {
                    "request_id": req.get("request_id", "?") if "req" in locals() else "?",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                _safe_unlink(request_path)
                print(f"[gc-server] ERROR: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)

        # ---- Clean shutdown ----------------------------------------------
        _write_kv_atomic(stopped_path, {
            "served": served,
            "pid": os.getpid(),
        })
        _safe_unlink(ready_path)
        _safe_unlink(pid_path)
        print(f"[gc-server] exited cleanly after {served} requests",
              flush=True)
        return 0

    except Exception as exc:
        _write_kv_atomic(error_path, {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fatal": "true",
        })
        _safe_unlink(ready_path)
        _safe_unlink(pid_path)
        print(f"[gc-server] FATAL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


def main():
    from . import config as cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", type=Path, required=True,
                    help="Coupling exchange directory (CAM6 run dir).")
    ap.add_argument("--ref-h1", type=Path, default=cfg.REFERENCE_H1_T,
                    help="Any CAM6 h1 file (for stencil + hybrid coord setup). "
                         f"Default: {cfg.REFERENCE_H1_T}")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to the GraphCast .npz checkpoint. Defaults to "
                         "the repo's params/ checkpoint.")
    ap.add_argument("--poll-sec", type=float, default=1.0)
    ap.add_argument("--warmup-h1-prev", type=Path, default=cfg.REFERENCE_H1_PREV,
                    help="h1 at T-6h for a warmup inference at startup. The "
                         "server compiles + runs one inference BEFORE writing "
                         "gc_server_ready.flag so READY = sub-10s per-request "
                         "latency. Pass --warmup-h1-prev '' to disable warmup. "
                         f"Default: {cfg.REFERENCE_H1_PREV}")
    ap.add_argument("--warmup-h1-t", type=Path, default=cfg.REFERENCE_H1_T,
                    help="h1 at T for warmup (paired with --warmup-h1-prev). "
                         f"Default: {cfg.REFERENCE_H1_T}")
    args = ap.parse_args()
    if args.checkpoint is None:
        from .run_graphcast_inference import CHECKPOINT_PATH as _CKPT
        args.checkpoint = _CKPT
    sys.exit(serve(
        args.rundir, args.ref_h1, args.checkpoint, poll_sec=args.poll_sec,
        warmup_h1_prev=args.warmup_h1_prev, warmup_h1_t=args.warmup_h1_t,
    ))


if __name__ == "__main__":
    main()
