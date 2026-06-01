"""GraphCast SUMO coordinator.

GC-side analog of sumo_coordinator.py. Runs concurrently with:
  - graphcast_server.py  (GPU node, warmup at startup, JAX-cached inference)
  - A CAM6 BHIST B-compset CESM case (active CAM6 + CLM5 + POP2 + CICE)

This coordinator does NOT involve CAMulator. For a supermodel run with both
CAMulator AND GraphCast, see supermodel_coordinator.py (eventual; the two
are unified once both single-source modes are stable).

Per-step protocol
=================

CAM6 barrier (unchanged from SUMO):
    CAM6 pauses after wshist at each 6-hour boundary, writes
    cesm_h1_ready.flag, blocks until the coordinator writes
    coordinator_done.flag.

GC server handshake:
    gc_request.flag    (this coord -> GC server)  -- "process these two h1 paths"
    gc_done.flag       (GC server -> this coord)  -- "prediction + nudge file ready"
    gc_server_ready.flag, gc_server.pid           -- server lifecycle
    gc_server_stop.flag                            -- coordinator asks server to exit

Step flow:

  step 0:
     - Wait for cesm_h1_ready.flag at start_ymd / start_tod.
     - Read its h1 path, cache as prev_h1.
     - No GC inference (we need 2 h1 frames for the 12-hour input duration).
     - Write coordinator_done.flag -> CAM6 advances from T_0 to T_1 without
       any GC nudging. (1 step of free run before nudging kicks in.)

  step >= 1:
     - Wait for cesm_h1_ready.flag at the next ymd/tod.
     - Get this h1's path = curr_h1.
     - request_id = stringified (ymd, tod)
     - Drop gc_request.flag with h1_prev=prev_h1, h1_t=curr_h1.
     - Wait for gc_done.flag; read gc_nudge_target path from it.
     - Copy that file to sumo_cam6_nudge.<T+6h>.nc (the time the contents
       are valid at) -- atomic rename so CAM6 never reads a half-written file.
     - Write coordinator_done.flag -> CAM6 advances, nudges toward GC.
     - prev_h1 = curr_h1.

Usage
=====

    python gc_coordinator.py \\
        --rundir /scratch/.../run/ \\
        --start_ymd 19800101 --start_tod 21600 \\
        [--max_steps 20]   [--gc_request_timeout 600]

The coordinator runs indefinitely (or until --max_steps), polling at
POLL_SLEEP seconds. Sends SIGTERM to the GC server on graceful exit IF a
PID file is present.
"""

import argparse
import logging
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Re-use the existing SUMO constants + helpers wherever possible so the
# CAM6 side stays bit-identical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sumo_coordinator import (
    CESM_H1_READY_FLAG, COORDINATOR_DONE_FLAG, SUMO_ACTIVE_FLAG,
    SUMO_CAM6_NUDGE_TEMPLATE, NUDGE_CLIMO_YEAR,
    POLL_SLEEP, POLL_LOG,
    delete_flag, write_flag,
    find_and_wait_for_cam6_h1, wait_for_cesm_h1_ready,
)

# GC server flag names match regrid/graphcast_server.py.
GC_REQUEST_FLAG  = "gc_request.flag"
GC_RESPONSE_FLAG = "gc_done.flag"
GC_ERROR_FLAG    = "gc_server_error.flag"
GC_READY_FLAG    = "gc_server_ready.flag"
GC_STOP_FLAG     = "gc_server_stop.flag"
GC_PID_FILE      = "gc_server.pid"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gc-coord")


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def step_six_hours(ymd: int, tod: int) -> tuple[int, int]:
    """Return (ymd, tod) advanced by 6 hours. CESM noleap calendar."""
    y, rest = divmod(ymd, 10000)
    m, d = divmod(rest, 100)
    new_tod = tod + 21600
    days_added = new_tod // 86400
    new_tod %= 86400
    if days_added == 0:
        return ymd, new_tod
    # Single day rollover (we always add 0 or 1 days here).
    days_in_month_noleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d += days_added
    while d > days_in_month_noleap[m - 1]:
        d -= days_in_month_noleap[m - 1]
        m += 1
        if m > 12:
            m = 1
            y += 1
    return y * 10000 + m * 100 + d, new_tod


def nudge_file_name(ymd: int, tod: int) -> str:
    """Build the sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc name for a (ymd, tod).

    Uses NUDGE_CLIMO_YEAR (= 1000) as the year stamp, matching the convention
    in the CAM6 nudging.F90 patch SUMO already uses. The month/day/seconds
    come from the actual simulation time.
    """
    _, rest = divmod(ymd, 10000)
    m, d = divmod(rest, 100)
    return SUMO_CAM6_NUDGE_TEMPLATE.format(y=NUDGE_CLIMO_YEAR, m=m, d=d, s=tod)


# -----------------------------------------------------------------------------
# GC server interaction
# -----------------------------------------------------------------------------

def send_gc_request(rundir: Path, request_id: str, h1_prev: Path, h1_t: Path) -> None:
    """Atomic-rename write of gc_request.flag for the GC server."""
    payload = (
        f"request_id={request_id}\n"
        f"h1_prev={h1_prev}\n"
        f"h1_t={h1_t}\n"
    )
    tmp = rundir / f"gc_request.{request_id}.tmp"
    tmp.write_text(payload)
    os.replace(tmp, rundir / GC_REQUEST_FLAG)


def parse_gc_response(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def wait_for_gc_done(
    rundir: Path, expected_request_id: str, timeout: float
) -> dict:
    """Block until gc_done.flag appears with the right request_id."""
    done = rundir / GC_RESPONSE_FLAG
    err  = rundir / GC_ERROR_FLAG
    t0 = time.time()
    while time.time() - t0 < timeout:
        if err.exists():
            info = parse_gc_response(err)
            raise RuntimeError(
                f"GC server reported error for "
                f"{info.get('request_id', '?')}: "
                f"{info.get('error_type', '?')}: {info.get('error', '?')}"
            )
        if done.exists():
            info = parse_gc_response(done)
            if info.get("request_id") != expected_request_id:
                logger.warning(
                    f"gc_done.flag found but request_id={info.get('request_id')} "
                    f"!= expected {expected_request_id} -- ignoring"
                )
                # Stale response; delete and keep waiting.
                delete_flag(done)
            else:
                return info
        time.sleep(POLL_SLEEP)
    raise TimeoutError(
        f"GC server did not produce gc_done.flag for {expected_request_id} "
        f"within {timeout}s"
    )


def wait_for_gc_server_ready(rundir: Path, timeout: float) -> None:
    """Block until the GC server has written its ready flag.

    Cold compile + warmup takes 15-25 min on a cold XLA cache; subsequent
    server starts (cache populated) are ~10-30 s. The caller passes a
    timeout matching expected scenarios.
    """
    ready = rundir / GC_READY_FLAG
    t0 = time.time()
    last_log = 0.0
    while not ready.exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"GC server ready flag never appeared at {ready} within {timeout}s"
            )
        if time.time() - last_log > 30:
            logger.info(f"  waiting for GC server ready flag... "
                        f"({time.time() - t0:.0f}s elapsed)")
            last_log = time.time()
        time.sleep(POLL_SLEEP)
    logger.info(f"  GC server READY after {time.time() - t0:.1f}s")


def stop_gc_server(rundir: Path, grace_sec: float = 30.0) -> None:
    """Ask the GC server to stop gracefully; fall back to SIGTERM via PID file.

    The graphcast_server.py polls for gc_server_stop.flag and exits when it
    appears. If a PID file is present and the server hasn't exited after
    `grace_sec`, send SIGTERM. As a last resort the operator can SIGKILL
    via `kill -9 $(cat gc_server.pid)` manually -- we don't do that here.
    """
    stop_path = rundir / GC_STOP_FLAG
    pid_path = rundir / GC_PID_FILE
    if not pid_path.exists():
        logger.info("  no GC server PID file -- nothing to stop")
        return
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, FileNotFoundError):
        logger.info("  GC server PID file unreadable -- skipping stop")
        return

    write_flag(stop_path)
    logger.info(f"  wrote {GC_STOP_FLAG}; waiting up to {grace_sec}s "
                f"for GC server (pid={pid}) to exit...")
    t0 = time.time()
    while time.time() - t0 < grace_sec:
        try:
            os.kill(pid, 0)  # alive?
        except OSError:
            logger.info(f"  GC server exited cleanly")
            return
        time.sleep(POLL_SLEEP)

    logger.warning(f"  GC server still alive after {grace_sec}s, sending SIGTERM")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "GraphCast SUMO coordinator: shuttles state between a CAM6 BHIST "
            "B-compset case and a long-lived graphcast_server.py process."
        )
    )
    p.add_argument("--rundir", required=True,
                   help="CAM6 CESM run directory. The GC server uses the same "
                        "directory for its I/O. h1 files appear here; dated "
                        "nudge files are written here.")
    p.add_argument("--cam6_case", default=None,
                   help="CAM6 case name prefix (e.g. g.e21.SUMO_GC_v01). If "
                        "omitted, the coordinator globs for *.cam.h1.*.nc.")
    p.add_argument("--start_ymd", type=int, default=19800101,
                   help="CESM start date YYYYMMDD (default 19800101).")
    p.add_argument("--start_tod", type=int, default=21600,
                   help="CESM start seconds-of-day (default 21600 = 6h). "
                        "CESM writes the first h1 6h into the run.")
    p.add_argument("--gc_request_timeout", type=float, default=600.0,
                   help="Max seconds to wait for one gc_done.flag response "
                        "(default 600 = 10 min; ~10 s typical with warm cache).")
    p.add_argument("--gc_ready_timeout", type=float, default=1800.0,
                   help="Max seconds to wait for gc_server_ready.flag at "
                        "startup (default 1800 = 30 min for cold compile).")
    p.add_argument("--cam6_h1_timeout", type=float, default=3600.0,
                   help="Max seconds to wait for the next CAM6 h1 (default 1h).")
    p.add_argument("--max_steps", type=int, default=0,
                   help="Stop after this many steps (0 = run indefinitely).")
    p.add_argument("--no_stop_gc_server", action="store_true",
                   help="On exit, do NOT signal the GC server to stop.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    rundir = Path(args.rundir)
    if not rundir.is_dir():
        logger.error(f"--rundir does not exist: {rundir}")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("GraphCast coordinator starting")
    logger.info(f"  rundir            : {rundir}")
    logger.info(f"  CAM6 case prefix  : {args.cam6_case or '(auto-glob)'}")
    logger.info(f"  start ymd/tod     : {args.start_ymd} / {args.start_tod}s")
    logger.info(f"  gc_request_timeout: {args.gc_request_timeout}s")
    logger.info(f"  gc_ready_timeout  : {args.gc_ready_timeout}s")
    logger.info(f"  max_steps         : "
                f"{args.max_steps if args.max_steps > 0 else '(unbounded)'}")
    logger.info("=" * 65)

    # 1. Ensure F90 barrier sentinel is in place.
    sumo_active_path = rundir / SUMO_ACTIVE_FLAG
    if not sumo_active_path.exists():
        write_flag(sumo_active_path)
        logger.info(f"  created {SUMO_ACTIVE_FLAG} (enables CAM6 F90 barrier)")

    # 2. Clean stale lifecycle flags from a previous run.
    for f in (GC_REQUEST_FLAG, GC_RESPONSE_FLAG, GC_ERROR_FLAG,
              CESM_H1_READY_FLAG, COORDINATOR_DONE_FLAG):
        delete_flag(rundir / f)

    # 3. Wait for the GC server to come up (cold compile path takes longest).
    logger.info("  waiting for GC server ready flag...")
    try:
        wait_for_gc_server_ready(rundir, args.gc_ready_timeout)
    except TimeoutError as exc:
        logger.error(f"  {exc}")
        sys.exit(1)

    poll_max = int(args.cam6_h1_timeout / POLL_SLEEP)
    next_ymd, next_tod = args.start_ymd, args.start_tod
    prev_h1: Path | None = None
    step = 0
    exit_code = 0

    try:
        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                logger.info(f"  reached --max_steps={args.max_steps}, exiting")
                break

            logger.info(f"--- step {step}  (expecting CAM6 h1 ymd={next_ymd} "
                        f"tod={next_tod}s) ---")

            # 1. Wait for CAM6 to pause at the next 6h boundary.
            curr_h1 = wait_for_cesm_h1_ready(
                rundir, args.cam6_case, next_ymd, next_tod, poll_max,
            )
            logger.info(f"  CAM6 paused, h1 ready: {curr_h1.name}")
            time.sleep(0.5)   # NFS write-through

            if prev_h1 is None:
                # Step 0: no prediction possible (need T-6h and T).
                logger.info("  step 0 has no T-6h frame; skipping GC inference "
                            "(CAM6 will free-run this 6h)")
                prev_h1 = curr_h1
            else:
                # Step >=1: full request/response cycle with the server.
                request_id = f"{next_ymd}-{next_tod:05d}"
                logger.info(f"  sending GC request {request_id}: "
                            f"{prev_h1.name} -> {curr_h1.name}")
                send_gc_request(rundir, request_id, prev_h1, curr_h1)

                t0 = time.time()
                try:
                    resp = wait_for_gc_done(
                        rundir, request_id, args.gc_request_timeout,
                    )
                except (RuntimeError, TimeoutError) as exc:
                    logger.error(f"  GC server failure: {exc}")
                    # Release CAM6 anyway so it doesn't deadlock; this 6h
                    # window will be free-running. The operator should fix
                    # the server and (optionally) resubmit.
                    delete_flag(rundir / GC_RESPONSE_FLAG)
                    delete_flag(rundir / GC_ERROR_FLAG)
                    nudge_failed = True
                else:
                    nudge_failed = False
                    logger.info(f"  GC done in {time.time() - t0:.1f}s "
                                f"(server: bundle={resp.get('t_bundle')}s "
                                f"infer={resp.get('t_inference')}s "
                                f"reverse={resp.get('t_reverse')}s)")
                    # Copy the server's gc_nudge_target file to the dated
                    # name CAM6's nudging template expects. Atomic rename.
                    src = Path(resp["gc_nudge_target"])
                    target_ymd, target_tod = step_six_hours(next_ymd, next_tod)
                    dst = rundir / nudge_file_name(target_ymd, target_tod)
                    tmp = dst.with_suffix(dst.suffix + ".tmp")
                    shutil.copyfile(src, tmp)
                    os.replace(tmp, dst)
                    logger.info(f"  wrote dated nudge file: {dst.name} "
                                f"(target time {target_ymd}/{target_tod}s)")
                    # Clear gc_done.flag so the next step starts clean.
                    delete_flag(rundir / GC_RESPONSE_FLAG)

                prev_h1 = curr_h1

            # 2. Release CAM6: it advances from T to T+6h. Delete the
            # ready flag so the next iteration's poll only sees a fresh
            # one CAM6 writes after its next wshist (mirrors sumo_coordinator).
            write_flag(rundir / COORDINATOR_DONE_FLAG)
            delete_flag(rundir / CESM_H1_READY_FLAG)
            logger.info(f"  wrote {COORDINATOR_DONE_FLAG} -- CAM6 unblocked")

            step += 1
            next_ymd, next_tod = step_six_hours(next_ymd, next_tod)

    except KeyboardInterrupt:
        logger.info("  KeyboardInterrupt -- coordinator exiting")
    except Exception as exc:
        logger.error(f"  coordinator unhandled error: {type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        if not args.no_stop_gc_server:
            try:
                stop_gc_server(rundir)
            except Exception as exc:
                logger.warning(f"  stop_gc_server failed: {exc}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
