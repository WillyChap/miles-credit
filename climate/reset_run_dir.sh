#!/bin/bash
# =============================================================================
# reset_run_dir.sh
#
# Resets the SUMO_CAM6_v03 run directory to a clean 1980-01-01 start:
#   1. Removes all case output files (restarts, history, logs)
#   2. Re-symlinks LE2 1980-01-01 restart data files
#   3. Copies rpointer files as writable local files (not symlinks)
#
# Usage: bash reset_run_dir.sh
# =============================================================================

set -e

CASENAME=g.e21.SUMO_CAM6_v03
RUNDIR=/glade/derecho/scratch/wchapman/${CASENAME}/run

REFCASE=b.e21.BHISTcmip6.f09_g17.LE2-1231.002
REFDATE=1980-01-01
REFTOD=00000
REFDIR=/glade/campaign/cgd/cesm/CESM2-LE/restarts/${REFCASE}/rest/${REFDATE}-${REFTOD}

# Canonical nudge template shipped with the coordinator (camulator_sumo/climate/).
# Copied into RUNDIR so sumo_coordinator.py uses a guaranteed-CAM6-compatible
# netCDF layout for every sumo_cam6_nudge.*.nc it writes.
CLIMATE_DIR=$(cd "$(dirname "$0")" && pwd)
NUDGE_TEMPLATE_SRC=${CLIMATE_DIR}/sumo_nudge_template.nc

cd ${RUNDIR}

# --- Remove all case output from previous runs ---
echo "==> Removing case output files..."
rm -fv ${CASENAME}.*.nc
rm -fv ${CASENAME}.*.r ${CASENAME}.*.ro
rm -fv ${CASENAME}.pop.ro.*

# --- Remove SUMO coordinator exchange files from previous runs ---
echo "==> Removing stale SUMO coordinator files..."
rm -fv sumo_cam6_nudge.*.nc sumo_cam6_nudge.*.nc.tmp
rm -fv sumo_nudge_template.nc
rm -fv camulator_atm_restart.pth
rm -fv camulator_sst_in.nc sumo_cam_state.nc sumo_combined_state.nc
rm -fv camulator_go.flag sumo_cam_ready.flag sumo_combine_done.flag camulator_done.flag
# SUMO barrier handshake flags (paired with cam_comp.F90:sumo_barrier_after_h1)
rm -fv cesm_h1_ready.flag coordinator_done.flag
# sumo_active.flag (sentinel that enables the F90 barrier) — recreated below
rm -fv sumo_active.flag
# GraphCast supermodel: server lifecycle flags, handshake, outputs, logs
rm -fv gc_request.flag gc_done.flag gc_server_error.flag \
       gc_server_ready.flag gc_server_stop.flag gc_server_stopped.flag \
       gc_server.pid gc_coordinator.pid \
       gc_server.log gc_coordinator.log
rm -fv gc_prediction.*.nc gc_nudge_target.*.nc

# --- Remove any stale rpointer.ocn.tavg created by previous runs ---
# (LE2 refdir has no rpointer.ocn.tavg; POP creates it after its first tavg write)
rm -fv rpointer.ocn.tavg

# --- Re-symlink restart data files ---
echo "==> Symlinking LE2 restart data files..."
for f in \
    ${REFCASE}.cam.i.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cam.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cam.rs.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.clm2.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cice.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cism.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cpl.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.mosart.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.pop.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.pop.ro.${REFDATE}-${REFTOD}
do
    src="${REFDIR}/${f}"
    if [[ -e "${src}" ]]; then
        ln -sf "${src}" "${f}"
        echo "    Linked: ${f}"
    else
        echo "    WARNING: not found (may be optional) — ${src}"
    fi
done

# --- Copy rpointer files as writable local files ---
# Must be copies, not symlinks: CESM writes updated rpointers during init
# (restFileMod.F90) and campaign storage is read-only.
echo "==> Copying rpointer files..."
for f in ${REFDIR}/rpointer.*; do
    fname=$(basename "$f")
    cp "$f" "${fname}"
    chmod u+w "${fname}"
    echo "    Copied: ${fname}"
done

# --- Copy canonical nudge template into the run directory ---
echo "==> Copying canonical nudge template..."
if [[ -f "${NUDGE_TEMPLATE_SRC}" ]]; then
    cp "${NUDGE_TEMPLATE_SRC}" sumo_nudge_template.nc
    chmod u+w sumo_nudge_template.nc
    echo "    Copied: sumo_nudge_template.nc (from ${NUDGE_TEMPLATE_SRC})"
else
    echo "    WARNING: ${NUDGE_TEMPLATE_SRC} not found —"
    echo "             sumo_coordinator.py will fall back to cloning the first h1."
fi

# --- Recreate sumo_active.flag so cam_comp.F90 enables its SUMO barrier ---
# Without this file, the F90 barrier short-circuits and CESM behaves like a
# normal (non-SUMO) case — that's how the same cesm.exe stays reusable.
touch sumo_active.flag
echo "==> Created sumo_active.flag (enables CAM6 SUMO barrier)"

echo ""
echo "==> Reset complete. Run directory: ${RUNDIR}"
echo "    Restart date : ${REFDATE}"
echo "    rpointer.atm : $(cat rpointer.atm)"
