#!/bin/bash
# =============================================================================
# run_coordinators.sh
#
# Single-command launcher for SUMO server + coordinator processes.
# Replaces juggling several SSH/terminal sessions.
#
# Modes:
#   camulator   - start CAMulator server + sumo_coordinator
#   graphcast   - start GraphCast server + gc_coordinator
#   both        - start all four (warn: supermodel_coordinator TBD; runs the
#                 two single-source coords with a conflict warning so you can
#                 manually arbitrate while we build the blend coord)
#
# Actions:
#   start (default)  - launch what's requested, write PID files, optionally tail
#   status           - report which servers/coordinators are alive in --rundir
#   stop             - send SIGTERM to all tracked PIDs in --rundir, then SIGKILL
#                      if still alive after the grace period
#
# Lifecycle files live in --rundir:
#   gc_server.pid               - GC server
#   gc_coordinator.pid          - GC coordinator (this script writes it)
#   camulator_server.pid        - CAMulator server (this script writes it)
#   camulator_coordinator.pid   - CAMulator coordinator
#   gc_server.log               - server stdout/stderr
#   gc_coordinator.log
#   camulator_server.log
#   camulator_coordinator.log
#
# Examples:
#   bash run_coordinators.sh --mode graphcast --rundir <rundir>
#   bash run_coordinators.sh --mode camulator --rundir <rundir> --tail
#   bash run_coordinators.sh --mode graphcast --rundir <rundir> --action status
#   bash run_coordinators.sh --mode graphcast --rundir <rundir> --action stop
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUPERMODEL_PY="/glade/work/wchapman/conda-envs/supermodel/bin/python"
CREDIT_COUPLING_PY="/glade/work/wchapman/conda-envs/credit-coupling/bin/python"

# Defaults to persistent reference pair (NOT scratch).
REF_H1_PREV="$REPO_DIR/regrid/reference_data/reference_h1_prev.nc"
REF_H1_T="$REPO_DIR/regrid/reference_data/reference_h1_t.nc"

MODE=""
RUNDIR=""
ACTION="start"
START_YMD=19800101
START_TOD=21600
MAX_STEPS=0
TAIL=0
NO_SERVER=0
NO_COORDINATOR=0
CAMULATOR_CONFIG=""
CAMULATOR_CHECKPOINT=""
CAMULATOR_IC=""
CAM6_CASE=""
GRACE_SEC=30

usage() {
    sed -n '2,40p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)          MODE="$2"; shift 2 ;;
        --rundir)        RUNDIR="$2"; shift 2 ;;
        --action)        ACTION="$2"; shift 2 ;;
        --start-ymd)     START_YMD="$2"; shift 2 ;;
        --start-tod)     START_TOD="$2"; shift 2 ;;
        --max-steps)     MAX_STEPS="$2"; shift 2 ;;
        --tail)          TAIL=1; shift ;;
        --no-server)     NO_SERVER=1; shift ;;
        --no-coordinator) NO_COORDINATOR=1; shift ;;
        --ref-h1)        REF_H1_T="$2"; shift 2 ;;
        --warmup-h1-prev) REF_H1_PREV="$2"; shift 2 ;;
        --warmup-h1-t)    REF_H1_T="$2"; shift 2 ;;
        --camulator-config)     CAMULATOR_CONFIG="$2"; shift 2 ;;
        --camulator-checkpoint) CAMULATOR_CHECKPOINT="$2"; shift 2 ;;
        --camulator-ic)         CAMULATOR_IC="$2"; shift 2 ;;
        --cam6-case)            CAM6_CASE="$2"; shift 2 ;;
        --grace-sec)            GRACE_SEC="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -z "$MODE" ]]   && { echo "--mode is required" >&2; exit 2; }
[[ -z "$RUNDIR" ]] && { echo "--rundir is required" >&2; exit 2; }
[[ ! -d "$RUNDIR" ]] && { echo "rundir does not exist: $RUNDIR" >&2; exit 2; }

case "$MODE" in
    camulator|graphcast|both) ;;
    *) echo "--mode must be one of: camulator graphcast both" >&2; exit 2 ;;
esac
case "$ACTION" in
    start|status|stop) ;;
    *) echo "--action must be one of: start status stop" >&2; exit 2 ;;
esac

# Standardized log + PID file names
GC_SRV_LOG="$RUNDIR/gc_server.log"
GC_SRV_PID="$RUNDIR/gc_server.pid"            # written by graphcast_server itself
GC_COORD_LOG="$RUNDIR/gc_coordinator.log"
GC_COORD_PID="$RUNDIR/gc_coordinator.pid"     # written by this script
CAM_SRV_LOG="$RUNDIR/camulator_server.log"
CAM_SRV_PID="$RUNDIR/camulator_server.pid"    # written by this script
CAM_COORD_LOG="$RUNDIR/camulator_coordinator.log"
CAM_COORD_PID="$RUNDIR/camulator_coordinator.pid"


# -----------------------------------------------------------------------------
# Status & stop helpers
# -----------------------------------------------------------------------------

_pid_alive() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 1
    local pid
    pid=$(cat "$pid_file" 2>/dev/null || echo "")
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

_print_status_line() {
    local label="$1" pid_file="$2" log="$3"
    if _pid_alive "$pid_file"; then
        local pid
        pid=$(cat "$pid_file")
        local etime
        etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
        printf "  %-26s  ALIVE  pid=%-7s  etime=%-12s  log=%s\n" \
            "$label" "$pid" "${etime:-?}" "$log"
    elif [[ -f "$pid_file" ]]; then
        printf "  %-26s  STALE  (pid file present but process dead)\n" "$label"
    else
        printf "  %-26s  DOWN\n" "$label"
    fi
}

do_status() {
    echo "===== SUMO process status in $RUNDIR ====="
    _print_status_line "GraphCast server"        "$GC_SRV_PID"    "$GC_SRV_LOG"
    _print_status_line "GraphCast coordinator"   "$GC_COORD_PID"  "$GC_COORD_LOG"
    _print_status_line "CAMulator server"        "$CAM_SRV_PID"   "$CAM_SRV_LOG"
    _print_status_line "CAMulator coordinator"   "$CAM_COORD_PID" "$CAM_COORD_LOG"
    [[ -f "$RUNDIR/gc_server_ready.flag" ]] && echo "  gc_server_ready.flag PRESENT" || true
    [[ -f "$RUNDIR/sumo_active.flag" ]]     && echo "  sumo_active.flag PRESENT (F90 barrier enabled)" || true
}

_stop_one() {
    local label="$1" pid_file="$2"
    if ! _pid_alive "$pid_file"; then
        [[ -f "$pid_file" ]] && rm -f "$pid_file" && echo "  $label: stale PID file removed"
        return
    fi
    local pid
    pid=$(cat "$pid_file")
    echo "  $label: SIGTERM pid=$pid (grace ${GRACE_SEC}s)..."
    kill -TERM "$pid" || true
    for i in $(seq 1 "$GRACE_SEC"); do
        kill -0 "$pid" 2>/dev/null || { echo "    $label exited cleanly"; rm -f "$pid_file"; return; }
        sleep 1
    done
    echo "  $label: SIGKILL pid=$pid"
    kill -KILL "$pid" 2>/dev/null || true
    rm -f "$pid_file"
}

do_stop() {
    echo "===== Stopping SUMO processes in $RUNDIR ====="
    # Stop coordinators first (so the server stops politely via the
    # stop-flag protocol if the coord wants), then the servers.
    _stop_one "GC coordinator"      "$GC_COORD_PID"
    _stop_one "CAMulator coordinator" "$CAM_COORD_PID"
    # Belt-and-suspenders: drop GC server's stop flag in case the
    # coordinator didn't (e.g., killed before it could send).
    touch "$RUNDIR/gc_server_stop.flag" 2>/dev/null || true
    _stop_one "GC server"           "$GC_SRV_PID"
    _stop_one "CAMulator server"    "$CAM_SRV_PID"
    echo "  done."
}


# -----------------------------------------------------------------------------
# Action: status / stop short-circuits
# -----------------------------------------------------------------------------

if [[ "$ACTION" == "status" ]]; then
    do_status
    exit 0
fi

if [[ "$ACTION" == "stop" ]]; then
    do_stop
    exit 0
fi


# -----------------------------------------------------------------------------
# Action: start
# -----------------------------------------------------------------------------

# Ensure the F90 barrier sentinel is present.
touch "$RUNDIR/sumo_active.flag"

start_gc_server() {
    if _pid_alive "$GC_SRV_PID"; then
        echo "  GC server already alive (pid=$(cat "$GC_SRV_PID")); skipping launch"
        return
    fi
    # Clear stale lifecycle flags from a SIGKILLed previous server. The new
    # server's own startup cleanup would do this AFTER imports + before warmup,
    # which is a long-enough window that downstream consumers (this script's
    # callers, the coordinator's wait_for_gc_server_ready) might race-spot the
    # stale READY flag and think the new server is already up.
    rm -f "$RUNDIR/gc_server_ready.flag" "$RUNDIR/gc_server_stop.flag" \
          "$RUNDIR/gc_server_stopped.flag" "$RUNDIR/gc_server_error.flag" \
          "$RUNDIR/gc_done.flag" "$RUNDIR/gc_request.flag"
    [[ -f "$GC_SRV_PID" ]] && rm -f "$GC_SRV_PID"
    echo "  starting GC server (logs -> $GC_SRV_LOG)"
    nohup "$SUPERMODEL_PY" -m regrid.graphcast_server \
        --rundir "$RUNDIR" \
        --ref-h1 "$REF_H1_T" \
        --warmup-h1-prev "$REF_H1_PREV" \
        --warmup-h1-t    "$REF_H1_T" \
      > "$GC_SRV_LOG" 2>&1 &
    # graphcast_server writes its own gc_server.pid; just give it a moment.
    sleep 1
    local pid=$!
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "    GC server failed to start; tail of log:" >&2
        tail -30 "$GC_SRV_LOG" >&2 || true
        return 1
    fi
    echo "    GC server bash pid=$pid; check $GC_SRV_PID once compile/warmup completes"
}

start_gc_coordinator() {
    if _pid_alive "$GC_COORD_PID"; then
        echo "  GC coordinator already alive (pid=$(cat "$GC_COORD_PID")); skipping"
        return
    fi
    # Coord waits for gc_server_ready.flag internally; we don't need to.
    echo "  starting GC coordinator (logs -> $GC_COORD_LOG)"
    local args=(--rundir "$RUNDIR" --start_ymd "$START_YMD" --start_tod "$START_TOD")
    [[ -n "$CAM6_CASE" ]] && args+=(--cam6_case "$CAM6_CASE")
    [[ "$MAX_STEPS" -gt 0 ]] && args+=(--max_steps "$MAX_STEPS")
    nohup "$SUPERMODEL_PY" "$REPO_DIR/climate/gc_coordinator.py" \
        "${args[@]}" > "$GC_COORD_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$GC_COORD_PID"
    echo "    GC coordinator pid=$pid"
}

start_camulator_server() {
    if _pid_alive "$CAM_SRV_PID"; then
        echo "  CAMulator server already alive (pid=$(cat "$CAM_SRV_PID")); skipping"
        return
    fi
    if [[ -z "$CAMULATOR_CONFIG" || -z "$CAMULATOR_CHECKPOINT" || -z "$CAMULATOR_IC" ]]; then
        echo "  --camulator-config, --camulator-checkpoint, and --camulator-ic " \
             "must all be set to start the CAMulator server" >&2
        return 1
    fi
    echo "  starting CAMulator server (logs -> $CAM_SRV_LOG)"
    nohup "$CREDIT_COUPLING_PY" "$REPO_DIR/climate/camulator_sumo_server.py" \
        --config "$CAMULATOR_CONFIG" \
        --model_name "$CAMULATOR_CHECKPOINT" \
        --rundir "$RUNDIR" \
        --init_cond "$CAMULATOR_IC" \
        --sumo --sumo_vars U V --sumo_tau 6.0 \
        --save_atm_nc camulator_out --daily_mean \
      > "$CAM_SRV_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$CAM_SRV_PID"
    echo "    CAMulator server pid=$pid"
}

start_camulator_coordinator() {
    if _pid_alive "$CAM_COORD_PID"; then
        echo "  CAMulator coordinator already alive (pid=$(cat "$CAM_COORD_PID")); skipping"
        return
    fi
    echo "  starting CAMulator coordinator (logs -> $CAM_COORD_LOG)"
    local args=(
        --rundir "$RUNDIR"
        --cam6dir "$RUNDIR"
        --vars U V --alpha_cam 0.5
        --start_ymd "$START_YMD"
        --start_tod "$START_TOD"
    )
    [[ -n "$CAM6_CASE" ]] && args+=(--cam6_case "$CAM6_CASE")
    [[ "$MAX_STEPS" -gt 0 ]] && args+=(--max_steps "$MAX_STEPS")
    nohup "$CREDIT_COUPLING_PY" "$REPO_DIR/climate/sumo_coordinator.py" \
        "${args[@]}" > "$CAM_COORD_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$CAM_COORD_PID"
    echo "    CAMulator coordinator pid=$pid"
}

echo "===== Launching ($MODE) in $RUNDIR ====="

case "$MODE" in
    graphcast)
        [[ $NO_SERVER -eq 0 ]]      && start_gc_server
        [[ $NO_COORDINATOR -eq 0 ]] && start_gc_coordinator
        ;;
    camulator)
        [[ $NO_SERVER -eq 0 ]]      && start_camulator_server
        [[ $NO_COORDINATOR -eq 0 ]] && start_camulator_coordinator
        ;;
    both)
        echo "  [warn] mode=both runs both single-source coordinators concurrently."
        echo "         They each write sumo_cam6_nudge.*.nc, which will conflict."
        echo "         Use this only for development of supermodel_coordinator.py."
        if [[ $NO_SERVER -eq 0 ]]; then
            start_gc_server || true
            start_camulator_server || true
        fi
        if [[ $NO_COORDINATOR -eq 0 ]]; then
            start_gc_coordinator
            start_camulator_coordinator
        fi
        ;;
esac

echo
do_status

if [[ $TAIL -eq 1 ]]; then
    echo
    echo "===== Tailing logs (Ctrl-C to stop tailing, processes keep running) ====="
    LOGS=()
    [[ -f "$GC_SRV_LOG" ]]        && LOGS+=("$GC_SRV_LOG")
    [[ -f "$GC_COORD_LOG" ]]      && LOGS+=("$GC_COORD_LOG")
    [[ -f "$CAM_SRV_LOG" ]]       && LOGS+=("$CAM_SRV_LOG")
    [[ -f "$CAM_COORD_LOG" ]]     && LOGS+=("$CAM_COORD_LOG")
    if [[ ${#LOGS[@]} -gt 0 ]]; then
        tail -F "${LOGS[@]}"
    fi
fi
