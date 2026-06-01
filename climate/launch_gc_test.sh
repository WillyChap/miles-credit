#!/bin/bash
# =============================================================================
# launch_gc_test.sh   --   *** INTERACTIVE / SINGLE-HOST ONLY ***
#
# This script orchestrates a GraphCast-only test run from ONE interactive
# shell that can both: (a) launch a GPU server, (b) call CESM's case.submit.
# In production this is the wrong split: Casper hosts the A100 servers,
# Derecho hosts the CESM compute nodes. case.submit on Casper will go to
# the wrong PBS queue, and the script's wait/trap loop cannot survive a
# Casper walltime.
#
# Use this script ONLY when:
#   - You are on a single Casper interactive A100 node AND
#   - --no-case-submit is passed (you'll run case.submit separately on
#     Derecho, or you're just exercising the server+coord).
#
# For production / batch:
#   - Submit Casper-side servers + coordinator via climate/submit_supermodel.pbs
#   - Submit Derecho-side CESM via the usual `cd $CASEROOT && ./case.submit`
#   - They share only the run dir on /glade/derecho/scratch.
#
# What this script does:
#   1. Start the GraphCast server in the background (with warmup at startup).
#   2. Wait for gc_server_ready.flag.
#   3. Start the GC coordinator in the background.
#   4. (Optional, --no-case-submit to skip): call ./case.submit.
#   5. On exit (clean or trapped): stop the coordinator, stop the GC server.
#
# Assumes the CESM case is already set up (mirrors the existing SUMO
# CAMulator case structure): nudging.F90 patched with Nudge_Do=0, F90
# barrier (sumo_barrier_after_h1) compiled into cam_comp.F90, fincl2 has
# the GraphCast input fields, sumo_active.flag is present in the run dir.
#
# Usage:
#   bash launch_gc_test.sh \\
#       --caseroot /glade/work/wchapman/cesm/CREDIT/g.e21.SUMO_GC_v01 \\
#       --rundir   /glade/derecho/scratch/wchapman/g.e21.SUMO_GC_v01/run \\
#       --ref-h1   /glade/derecho/scratch/wchapman/g.e21.SUMO_GC_v01/run/some_h1.nc \\
#       --start-ymd 19800101 --start-tod 21600 \\
#       [--max-steps 20]
#
# Lifecycle flags written to rundir:
#   gc_server.pid              -- server PID (for emergency kill)
#   gc_server_ready.flag       -- server is JIT-compiled + warmed up
#   gc_coordinator.pid         -- coordinator PID
#   gc_server_stop.flag        -- ask server to exit
#   gc_server_stopped.flag     -- server confirms clean exit
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUPERMODEL_PY="/glade/work/wchapman/conda-envs/supermodel/bin/python"

# Persistent reference h1 pair (lives on /glade/work, NOT scratch). These are
# used by the GC server for grid metadata + JIT warmup, NOT for any case-
# specific physics -- so a single canonical pair works for every test case.
REF_H1_PREV="$REPO_DIR/regrid/reference_data/reference_h1_prev.nc"
REF_H1_T="$REPO_DIR/regrid/reference_data/reference_h1_t.nc"

CASEROOT=""
RUNDIR=""
REF_H1=""
WARMUP_H1_PREV=""
WARMUP_H1_T=""
START_YMD=19800101
START_TOD=21600
MAX_STEPS=0
NO_CASE_SUBMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --caseroot)        CASEROOT="$2"; shift 2 ;;
        --rundir)          RUNDIR="$2";   shift 2 ;;
        --ref-h1)          REF_H1="$2";   shift 2 ;;
        --warmup-h1-prev)  WARMUP_H1_PREV="$2"; shift 2 ;;
        --warmup-h1-t)     WARMUP_H1_T="$2";    shift 2 ;;
        --start-ymd)       START_YMD="$2"; shift 2 ;;
        --start-tod)       START_TOD="$2"; shift 2 ;;
        --max-steps)       MAX_STEPS="$2"; shift 2 ;;
        --no-case-submit)  NO_CASE_SUBMIT=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Defaults: use the persistent reference pair shipped with the repo.
[[ -z "$REF_H1" ]]         && REF_H1="$REF_H1_T"
[[ -z "$WARMUP_H1_PREV" ]] && WARMUP_H1_PREV="$REF_H1_PREV"
[[ -z "$WARMUP_H1_T" ]]    && WARMUP_H1_T="$REF_H1_T"

[[ -z "$RUNDIR" ]] && { echo "--rundir is required" >&2; exit 2; }
[[ ! -d "$RUNDIR" ]] && { echo "rundir does not exist: $RUNDIR" >&2; exit 2; }
[[ ! -f "$REF_H1" ]] && { echo "ref-h1 does not exist: $REF_H1" >&2; exit 2; }
[[ ! -f "$WARMUP_H1_PREV" ]] && { echo "warmup-h1-prev does not exist: $WARMUP_H1_PREV" >&2; exit 2; }
[[ ! -f "$WARMUP_H1_T" ]]    && { echo "warmup-h1-t does not exist: $WARMUP_H1_T" >&2; exit 2; }
if [[ $NO_CASE_SUBMIT -eq 0 ]]; then
    [[ -z "$CASEROOT" ]] && { echo "--caseroot is required unless --no-case-submit" >&2; exit 2; }
    [[ ! -d "$CASEROOT" ]] && { echo "caseroot does not exist: $CASEROOT" >&2; exit 2; }
fi

GC_SERVER_LOG="$RUNDIR/gc_server.log"
GC_COORD_LOG="$RUNDIR/gc_coordinator.log"
GC_SERVER_PID_FILE="$RUNDIR/gc_server.pid"
GC_COORD_PID_FILE="$RUNDIR/gc_coordinator.pid"

GC_SERVER_READY_FLAG="$RUNDIR/gc_server_ready.flag"
GC_SERVER_STOP_FLAG="$RUNDIR/gc_server_stop.flag"

cleanup() {
    local rc=$?
    echo "[launcher] cleanup (exit $rc)..."

    # Stop coordinator first (it owns the GC server stop signal).
    if [[ -f "$GC_COORD_PID_FILE" ]]; then
        local cpid
        cpid=$(cat "$GC_COORD_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$cpid" ]] && kill -0 "$cpid" 2>/dev/null; then
            echo "[launcher] stopping coordinator pid=$cpid"
            kill -TERM "$cpid" || true
            # Give it ~30s to exit cleanly (it then signals server).
            for i in $(seq 1 30); do
                kill -0 "$cpid" 2>/dev/null || break
                sleep 1
            done
            kill -0 "$cpid" 2>/dev/null && kill -KILL "$cpid" || true
        fi
        rm -f "$GC_COORD_PID_FILE"
    fi

    # Belt-and-suspenders: if the server is still alive, signal it via flag
    # and then SIGTERM via PID. The coordinator should have done this, but
    # we cover the case where the coordinator died abruptly.
    if [[ -f "$GC_SERVER_PID_FILE" ]]; then
        echo "[launcher] dropping server stop flag (belt-and-suspenders)"
        touch "$GC_SERVER_STOP_FLAG"
        local spid
        spid=$(cat "$GC_SERVER_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$spid" ]]; then
            for i in $(seq 1 30); do
                kill -0 "$spid" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$spid" 2>/dev/null; then
                echo "[launcher] server still alive, SIGTERM pid=$spid"
                kill -TERM "$spid" || true
                sleep 5
                kill -0 "$spid" 2>/dev/null && kill -KILL "$spid" || true
            fi
        fi
    fi
    echo "[launcher] cleanup done"
}
trap cleanup EXIT INT TERM

# Clean stale flags from a prior run.
rm -f "$GC_SERVER_READY_FLAG" "$GC_SERVER_STOP_FLAG" \
      "$RUNDIR/gc_done.flag" "$RUNDIR/gc_request.flag" \
      "$RUNDIR/gc_server_error.flag" "$RUNDIR/gc_server_stopped.flag" \
      "$GC_SERVER_PID_FILE" "$GC_COORD_PID_FILE"

# Ensure the SUMO F90 barrier sentinel is in place.
touch "$RUNDIR/sumo_active.flag"

# ---- 1. Start GC server in the background ---------------------------------
echo "[launcher] starting GC server (logs -> $GC_SERVER_LOG)"
nohup "$SUPERMODEL_PY" -m regrid.graphcast_server \
    --rundir "$RUNDIR" \
    --ref-h1 "$REF_H1" \
    --warmup-h1-prev "$WARMUP_H1_PREV" \
    --warmup-h1-t    "$WARMUP_H1_T" \
  > "$GC_SERVER_LOG" 2>&1 &
SERVER_BASH_PID=$!
echo "[launcher] GC server bash pid=$SERVER_BASH_PID"

# Wait for ready (cold compile path may take ~20 min; subsequent are ~30 s).
echo "[launcher] waiting for $GC_SERVER_READY_FLAG (up to 1800 s)..."
WAITED=0
while [[ ! -f "$GC_SERVER_READY_FLAG" ]]; do
    if ! kill -0 "$SERVER_BASH_PID" 2>/dev/null; then
        echo "[launcher] GC server process died before becoming ready -- see log" >&2
        tail -50 "$GC_SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ $WAITED -ge 1800 ]]; then
        echo "[launcher] GC server never became ready in 1800 s" >&2
        exit 1
    fi
    if [[ $((WAITED % 60)) -eq 0 ]]; then
        echo "[launcher]   still waiting ($WAITED s)..."
    fi
done
echo "[launcher] GC server READY (after $WAITED s)"

# ---- 2. Start GC coordinator in the background ----------------------------
echo "[launcher] starting GC coordinator (logs -> $GC_COORD_LOG)"
COORD_ARGS=(
    --rundir "$RUNDIR"
    --start_ymd "$START_YMD"
    --start_tod "$START_TOD"
)
[[ "$MAX_STEPS" -gt 0 ]] && COORD_ARGS+=(--max_steps "$MAX_STEPS")

nohup "$SUPERMODEL_PY" "$REPO_DIR/climate/gc_coordinator.py" \
    "${COORD_ARGS[@]}" > "$GC_COORD_LOG" 2>&1 &
COORD_PID=$!
echo "$COORD_PID" > "$GC_COORD_PID_FILE"
echo "[launcher] coordinator pid=$COORD_PID"

# ---- 3. Submit CESM (or skip) --------------------------------------------
if [[ $NO_CASE_SUBMIT -eq 1 ]]; then
    echo "[launcher] --no-case-submit set; not running case.submit."
    echo "[launcher] server + coordinator are running. CTRL-C or kill -TERM "
    echo "[launcher] this script's pid ($$) to stop everything."
    # Block on the coordinator so the trap fires when it exits.
    wait "$COORD_PID" || true
    exit $?
fi

echo "[launcher] submitting CESM via $CASEROOT/case.submit"
pushd "$CASEROOT" > /dev/null
./case.submit
SUBMIT_RC=$?
popd > /dev/null

if [[ $SUBMIT_RC -ne 0 ]]; then
    echo "[launcher] case.submit returned $SUBMIT_RC" >&2
    exit "$SUBMIT_RC"
fi

echo "[launcher] CESM submitted. Blocking on coordinator (CTRL-C to stop)."
wait "$COORD_PID" || true
echo "[launcher] coordinator exited."
