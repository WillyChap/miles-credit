#!/bin/bash
# =============================================================================
# setup_SUMO_B_case.sh
#
# Creates and configures the CAM6 side of the CAMulator+CAM6 supermodel (SUMO).
#
# Compset: BHIST-SWAV  (CAM6 + CLM5 + CICE + POP2 + MOSART + CISM2 + SWAV)
# SWAV (stub wave) replaces WW3: LE2 restarts have no ww3.r, so active WW3
# cannot branch from them. SWAV allows branch restart and is compatible with
# all LE2-based restarts (SUMO here, GIAF later).
# Resolution: f09_g17  (0.9°×1.25° atmosphere, gx1v7 1-degree ocean)
#
# Architecture:
#   - This case runs CAM6 as the ACTIVE atmosphere coupled to POP2+CICE.
#   - CAMulator runs as an EXTERNAL server on a GPU node (camulator_sumo_server.py)
#     in AMIP mode with prescribed ERA5/IAF SSTs (no DATM in this case).
#   - At each 6h coupling interval:
#       a) CAMulator finishes its step, writes U,V to sumo_cam_state.nc,
#          touches sumo_cam_ready.flag, and WAITS.
#       b) CAM6 writes its 6-hourly h1 history (U:I, V:I instantaneous).
#       c) The coordinator (sumo_coordinator.py) reads both, computes
#          X_combined = 0.5*X_cam + 0.5*X_cam6, and:
#            - writes sumo_combined_state.nc  -> signals CAMulator via
#              sumo_combine_done.flag (CAMulator applies nudging and advances)
#            - writes sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc in RUNDIR
#   - The CESM nudging toolbox (Nudge_Model=.true.) reads the dated nudge
#     file and relaxes CAM6 toward the combined state at the next step.
#   - The ocean (POP2+CICE) is driven by CAM6's atmospheric state.
#
# No SuperModel_CAM / Python-Fortran bridge modifications needed.
# No Fortran recompilation beyond the standard BHIST build.
#
# SUMO coordinator (run separately, on any node accessible to both rundirs):
#   python sumo_coordinator.py \
#       --rundir  <THIS_RUNDIR>  \   # CAMulator flag/state files + nudge output
#       --cam6dir <THIS_RUNDIR>  \   # CAM6 h1 files (same dir for BHIST)
#       --cam6_case <CASENAME>   \   # optional but speeds up h1 detection
#       --vars U V               \
#       --alpha_cam 0.5
#   NOTE: --rundir == --cam6dir because both models share the run directory.
#
# CAMulator server (run on Casper GPU node):
#   python camulator_sumo_server.py \
#       --config  .../camulator_config.yml \
#       --model_name checkpoint.pt00091.pt \
#       --rundir  <THIS_RUNDIR>    \
#       --init_cond <SUMO_IC>.pth  \   # created by make_sumo_ic_from_cam6_restart.py
#       --sumo --sumo_vars U V     \
#       --sumo_tau 6.0             \
#       --save_atm_nc camulator_out --daily_mean
#
# Usage:
#   bash setup_SUMO_B_case.sh            # create + setup + build
#   bash setup_SUMO_B_case.sh nocreate   # skip create_newcase
#   bash setup_SUMO_B_case.sh nobuild    # skip case.build
#
# Prerequisites:
#   1. camulator_sumo_server.py running on a GPU node BEFORE ./case.submit
#   2. sumo_coordinator.py running on a CPU node BEFORE ./case.submit
#   3. SUMO IC created:
#      python make_sumo_ic_from_cam6_restart.py \
#          --config     ./camulator_config.yml \
#          --model_name checkpoint.pt00091.pt \
#          --era5_ic    /path/to/init_cond_tensor_1980-01-01T00z.pth \
#          --cam6_restart <REFDIR>/<REFCASE>.cam.r.<REFDATE>-<REFTOD>.nc \
#          --output     <SUMO_IC>.pth
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION — update these as needed
# =============================================================================

CESM_ROOT=/glade/work/wchapman/JE_help_cnn/camulator_sandbox_fluxupdown/

CASE_DIR=/glade/work/wchapman/cesm/CREDIT/g.e21.SUMO_CAM6_v03
CASENAME=g.e21.SUMO_CAM6_v03

# Restart from a BHIST ocean/ice state
# (POP + CICE restart from CESM2-LE member, g17 grid)
REFCASE=b.e21.BHISTcmip6.f09_g17.LE2-1231.002
REFDATE=1980-01-01
REFTOD=00000
REFDIR=/glade/campaign/cgd/cesm/CESM2-LE/restarts/${REFCASE}/rest/${REFDATE}-${REFTOD}

PROJECT=P03010039
MACH=derecho
COMPILER=intel

# B compset: CAM6 + CLM5 + CICE + POP2 + MOSART + CISM2 + SWAV (stub wave)
# SWAV replaces WW3 so we can branch from LE2 refdats, which have no ww3.r
# (LE2 ran without active wave model — no rpointer.wav at any date)
COMPSET="HIST_CAM60_CLM50%BGC-CROP_CICE_POP2%ECO%ABIO-DIC_MOSART_CISM2%NOEVOLVE_SWAV_BGC%BDRD"

# f09_g17: 0.9°×1.25° atmosphere (CAM6 native), gx1v7 1-degree ocean
RES=f09_g17

# =============================================================================
# STEP 1 — create_newcase
# =============================================================================
if [[ "$1" != "nocreate" && "$1" != "nobuild" ]]; then
    echo "==> Creating case: $CASE_DIR"
    ${CESM_ROOT}/cime/scripts/create_newcase \
        --case     ${CASE_DIR} \
        --mach     ${MACH} \
        --compiler ${COMPILER} \
        --compset  ${COMPSET} \
        --res      ${RES} \
        --project  ${PROJECT} \
        --run-unsupported
else
    echo "==> Skipping create_newcase (case assumed to exist at $CASE_DIR)"
fi

cd ${CASE_DIR}

# =============================================================================
# STEP 2 — xmlchanges
# =============================================================================
echo "==> Applying xmlchanges..."

# NCPL: use compset defaults (ATM=48→1800s, OCN=1/day for POP2, ROF follows ATM).
# Do NOT override to 4: SUMO's 6-hourly exchange is driven by nhtfrq=-6 history
# output, not the coupling interval. ATM_NCPL=4 sets dtime=21600s and crashes
# radiation (iradsw=nint(3600/21600)=0 → mod-by-zero in radiation_do).
./xmlchange NCPL_BASE_PERIOD=day

# --- Hybrid run: CLM reads LE2 clm2.r via finidat (use_init_interp handles
# the 25-landunit mismatch: LE2=62125, our build=62100). ATM reads cam.i
# (flat eta levels), but fv_nspltvrm=4 keeps remapping intervals short
# enough (450s) to prevent Lagrangian level inversion with 90 m/s jets.
./xmlchange RUN_TYPE=hybrid
./xmlchange RUN_REFCASE=${REFCASE}
./xmlchange RUN_REFDATE=${REFDATE}
./xmlchange RUN_REFTOD=${REFTOD}
./xmlchange RUN_STARTDATE=${REFDATE}
./xmlchange GET_REFCASE=FALSE

# --- Run length ---
./xmlchange STOP_OPTION=ndays
./xmlchange STOP_N=5
./xmlchange RESUBMIT=0

# --- Queue and walltime ---
./xmlchange JOB_QUEUE=main
./xmlchange JOB_WALLCLOCK_TIME=00:30:00
./xmlchange JOB_PRIORITY=premium

# --- Turn off archiving during development ---
./xmlchange DOUT_S=FALSE

# --- PE layout: match BHIST LE layout for f09_g17 ---
./xmlchange NTASKS_ATM=288,NTASKS_LND=288,NTASKS_CPL=288
./xmlchange NTASKS_ICE=288,NTASKS_WAV=1,NTASKS_GLC=288
./xmlchange NTASKS_ROF=288
./xmlchange NTASKS_OCN=288
./xmlchange ROOTPE_OCN=288

# =============================================================================
# STEP 3 — case.setup
# =============================================================================
echo "==> Running case.setup..."
./case.setup

# =============================================================================
# STEP 4 — MPI GPU-compatibility fixes
# =============================================================================
echo "==> Patching env_mach_specific.xml with MPI fixes..."

for nameval in \
    "MPICH_GPU_SUPPORT_ENABLED:0" \
    "FI_CXI_DISABLE_HOST_REGISTER:1" \
    "MPICH_SMP_SINGLE_COPY_MODE:NONE"
do
    varname="${nameval%%:*}"
    varval="${nameval##*:}"
    if ! grep -q "name=\"${varname}\"" env_mach_specific.xml; then
        sed -i "s|</environment_variables>|    <env name=\"${varname}\">${varval}</env>\n  </environment_variables>|" \
            env_mach_specific.xml
        echo "    Added ${varname}=${varval}"
    else
        echo "    Already present: ${varname}"
    fi
done

# =============================================================================
# STEP 5 — symlink BHIST restarts into RUNDIR
#
# For the B compset hybrid restart we need:
#   <REFCASE>.cam.r.<REFDATE>-<REFTOD>.nc    — CAM6 atmosphere restart
#   <REFCASE>.cam.rs.<REFDATE>-<REFTOD>.nc   — CAM6 radiation restart
#   <REFCASE>.clm2.r.<REFDATE>-<REFTOD>.nc   — CLM land restart
#   <REFCASE>.pop.r.<REFDATE>-<REFTOD>.nc    — POP2 ocean state
#   <REFCASE>.pop.ro.<REFDATE>-<REFTOD>      — POP2 overflow (binary)
#   <REFCASE>.cice.r.<REFDATE>-<REFTOD>.nc   — CICE sea ice state
#   <REFCASE>.cpl.r.<REFDATE>-<REFTOD>.nc    — coupler fractions
#   <REFCASE>.mosart.r.<REFDATE>-<REFTOD>.nc — river routing (if present)
#
# The CAM restart is also the source for the CAMulator SUMO IC:
#   make_sumo_ic_from_cam6_restart.py reads it to create a matched
#   CAMulator initial-condition tensor that starts from the same
#   atmospheric state as CAM6 (critical for fast supermodel synchronization).
# =============================================================================
echo "==> Symlinking BHIST restart files into RUNDIR..."
RUNDIR=$(./xmlquery RUNDIR --value)
mkdir -p ${RUNDIR}

# Core restarts (ocean, ice, coupler)
for f in \
    ${REFCASE}.pop.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.pop.ro.${REFDATE}-${REFTOD} \
    ${REFCASE}.cice.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cpl.r.${REFDATE}-${REFTOD}.nc
do
    src="${REFDIR}/${f}"
    dst="${RUNDIR}/${f}"
    if [[ -e "${src}" ]]; then
        ln -sf "${src}" "${dst}"
        echo "    Linked: ${f}"
    else
        echo "    WARNING: not found — ${src}"
    fi
done

# B-compset-specific restarts (atmosphere, land, runoff)
# cam.i is NOT needed for branch RUN_TYPE (cam.r is read instead)
for f in \
    ${REFCASE}.cam.i.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cam.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cam.rs.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.clm2.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.mosart.r.${REFDATE}-${REFTOD}.nc \
    ${REFCASE}.cism.r.${REFDATE}-${REFTOD}.nc
do
    src="${REFDIR}/${f}"
    dst="${RUNDIR}/${f}"
    if [[ -e "${src}" ]]; then
        ln -sf "${src}" "${dst}"
        echo "    Linked: ${f}"
    else
        echo "    WARNING: not found (may be optional) — ${src}"
    fi
done

# --- rpointer files: copy (not symlink) from REFDIR ---
# Must be writable local files: CESM writes updated rpointer.* files during
# CLM/CAM initialization (restFileMod.F90). If they are symlinks to a
# read-only archive the write fails with "forrtl severe (47): READONLY file".
echo "==> Copying rpointer files from REFDIR (writable local copies)..."
for f in ${REFDIR}/rpointer.*; do
    fname=$(basename "$f")
    cp "$f" "${RUNDIR}/${fname}"
    chmod u+w "${RUNDIR}/${fname}"
    echo "    Copied: ${fname}"
done

echo ""
echo "    CAM restart (also needed for CAMulator IC creation):"
echo "      ${REFDIR}/${REFCASE}.cam.r.${REFDATE}-${REFTOD}.nc"
echo ""
echo "    Create CAMulator SUMO IC with:"
echo "      python make_sumo_ic_from_cam6_restart.py \\"
echo "          --config     ./camulator_config.yml \\"
echo "          --model_name checkpoint.pt00091.pt \\"
echo "          --era5_ic    /path/to/init_cond_tensor_1980-01-01T00z.pth \\"
echo "          --cam6_restart ${REFDIR}/${REFCASE}.cam.r.${REFDATE}-${REFTOD}.nc \\"
echo "          --output     ${RUNDIR}/sumo_init_1980-01-01T00z.pth"

# =============================================================================
# STEP 6 — user namelists
# =============================================================================
echo "==> Writing user namelists..."

# --- user_nl_cam: CESM nudging toolbox for SUMO ---
#
# CAM6 is nudged toward the combined state via the standard CESM nudging
# toolbox (Davis et al. 2022).  No Python-Fortran bridge or source
# modifications are required.
#
# The coordinator (sumo_coordinator.py) writes dated nudge-target files:
#   ${RUNDIR}/sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc
# The CESM nudging toolbox finds them via Nudge_File_Template.
#
# Nudge_Uprof / Nudge_Vprof = 3: nudge the full atmospheric column.
# Nudge_Utau / Nudge_Vtau = 21600.0: 6-hour relaxation timescale,
#   matching the SUMO coupling interval.
# Nudge_Tprof / Nudge_Qprof / Nudge_PSprof = 0: nudge U, V only.
#
cat >> user_nl_cam << EOF
! SUMO synchronization via CESM nudging toolbox
! The SUMO coordinator writes dated nudge-target files:
!   ${RUNDIR}/sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc
Nudge_Model    = .true.
Nudge_Path     = '${RUNDIR}/'
Nudge_File_Template = 'sumo_cam6_nudge.%y-%m-%d-%s.nc'
! Force_Opt=1: Newtonian relaxation. dU/dt = Ucoef * window * (target - U) / Utau
! Spreads the Δ over Utau (6h) rather than fractional replacement-per-nudge-step,
! avoiding the CFL shock that crashed Force_Opt=0 with any nonzero Ucoef.
Nudge_Force_Opt      = 1
Nudge_Times_Per_Day  = 4
! Nudge_Do=0: standard CAM6 nudging  Nudge_Ustep = (Target - Model) * Tscale * Utau
! (Nudge_Do=1 is the ML-tendency mode where Target_X must be pre-computed
!  tendencies, NOT field values. SUMO writes field values → must use 0.)
Nudge_Do             = 0
Nudge_Uprof    = 1
Nudge_Vprof    = 1
Nudge_Tprof    = 0
Nudge_Qprof    = 0
Nudge_PSprof   = 0
! Gentle relaxation gain — Force_Opt=1 at Ucoef=1.0 crashed at the first full
! T-T nudge step; 0.3 gives per-step ΔU ~0.9 m/s at the worst cell (well below
! the FV CFL margin). Tune up if simulations show insufficient pull.
Nudge_Ucoef    = 0.3
Nudge_Vcoef    = 0.3
! Vertical window: zero in top ~5 model levels (above ~36 hPa where polar
! stratosphere is FV-CFL fragile) with a smooth 3-level ramp, full strength
! from level ~8 (≈70 hPa) down to the surface. CAM6 has pver=32, so
! Hindex=33 disables any cutoff at the surface.
Nudge_Vwin_Lindex  = 5.0
Nudge_Vwin_Ldelta  = 3.0
Nudge_Vwin_Hindex  = 33.0
Nudge_Vwin_Hdelta  = 0.001
Nudge_Vwin_Invert  = .false.
! Horizontal window: full nudging in tropics/midlats, smooth taper near the
! poles where the FV pole-row singularity is CFL-fragile. Window = 1 for
! |lat| < ~67°, ramps to 0 by |lat| > ~82°. Longitude unconstrained.
Nudge_Hwin_lat0     = 0.0
Nudge_Hwin_latWidth = 150.0
Nudge_Hwin_latDelta = 15.0
Nudge_Hwin_lon0     = 180.0
Nudge_Hwin_lonWidth = 9999.0
Nudge_Hwin_lonDelta = 1.0
Nudge_Hwin_Invert   = .false.
Nudge_Beg_Year  = 1980
Nudge_Beg_Month = 1
Nudge_Beg_Day   = 2
Nudge_End_Year  = 2025
Nudge_End_Month = 12
Nudge_End_Day   = 31
EOF

# --- user_nl_cice: subcycle CICE dynamics ---
cat >> user_nl_cice << 'EOF'
ndtd = 2
EOF

# --- user_nl_clm: allow interpolation when LE2 landunit count differs from our build ---
# LE2 clm2.r has 62125 landunits; CLM5%BGC-CROP builds 62100 — use_init_interp handles the mismatch
cat >> user_nl_clm << 'EOF'
use_init_interp = .true.
EOF

# --- 6-hourly instantaneous U,V history for SUMO coordinator ---
#
# The coordinator reads *.cam.h1.YYYY-MM-DD-SSSSS.nc at each coupling
# interval to get CAM6's current U,V state.
#
# ':I' = instantaneous snapshot (not a time average) — required for
# correct state exchange in the supermodel.
#
# mfilt = 1,1 → one record per file, so file timestamps map 1-to-1
# to coupling steps, making coordinator file detection unambiguous.
#
# Extra instantaneous fields (T,Q,PS) are written for diagnostics;
# only U and V are used by the coordinator.
#
cat >> user_nl_cam << 'EOF'

! More frequent vertical remapping: prevents Lagrangian level inversion during
! hybrid restart with strong jets (90 m/s). dt_vrm = 1800/4 = 450s < inversion threshold.
! Divisibility: auto-nsplit=8, nspltrac=4, nspltvrm=4 → 8%4=0, 4%4=0
fv_nspltrac = 4
fv_nspltvrm = 4

! 6-hourly instantaneous state for SUMO coordinator
! TS:I   = surface temperature (= POP SST over ocean) -> relayed to CAMulator as SST
! ICEFRAC:I = sea ice fraction -> relayed to CAMulator to overwrite climatological ICEFRAC
!
! GraphCast input requirements (added 2026-05-31 for GC supermodel coupling):
!   OMEGA:I  -> vertical_velocity            (Pa/s, 3D)
!   Z3:I     -> geopotential                 (m; translator multiplies by g for ERA5 geopotential)
!   TREFHT:I -> 2m_temperature               (K)
!   PSL:I    -> mean_sea_level_pressure      (Pa)
!   U10:I + UBOT:I + VBOT:I -> 10m_{u,v}_component_of_wind
!            CAM6 has no native 10m U/V components, only the magnitude U10.
!            Translator reconstructs components: U10_{u,v} = U10 * {UBOT,VBOT}/sqrt(UBOT^2+VBOT^2)
!   PRECT    -> total_precipitation_6hr      (m/s, averaged over 6h; translator * 21600 -> m)
!   PHIS     -> geopotential_at_surface      (m^2/s^2, static)
!   LANDFRAC -> land_sea_mask                (fraction, ~static)
fincl2 = 'U:I','V:I','T:I','Q:I','PS:I','TS:I','ICEFRAC:I','SST:I',
         'OMEGA:I','Z3:I','TREFHT:I','PSL:I',
         'U10:I','UBOT:I','VBOT:I',
         'PRECT','PHIS','LANDFRAC'
nhtfrq = 0,-6
mfilt  = 1,1
EOF

# --- Mark this run directory as SUMO-mode so cam_comp.F90 enables its barrier ---
# The same cesm.exe is reusable for non-SUMO cases: without this sentinel,
# sumo_barrier_after_h1 is a single integer compare per cam_run4 (no-op).
touch "${RUNDIR}/sumo_active.flag"
echo "==> Created ${RUNDIR}/sumo_active.flag (enables CAM6 SUMO barrier)"

# =============================================================================
# STEP 7 — case.build (optional)
# =============================================================================
if [[ "$1" != "nobuild" ]]; then
    echo "==> Running case.build (this takes ~25-50 minutes for B compset)..."
    ./case.build
    echo "==> Build complete."
    echo ""
    echo "    Run directory: ${RUNDIR}"
    echo ""
    echo "    Before submitting:"
    echo "      1. Create CAMulator SUMO IC (see Step 5 output above)"
    echo ""
    echo "      2. Launch sumo_coordinator.py (any CPU node or login node):"
    echo "         python sumo_coordinator.py \\"
    echo "             --rundir  ${RUNDIR} \\"
    echo "             --cam6dir ${RUNDIR} \\"
    echo "             --cam6_case ${CASENAME} \\"
    echo "             --vars U V \\"
    echo "             --alpha_cam 0.5"
    echo ""
    echo "      3. Launch camulator_sumo_server.py on Casper GPU node:"
    echo "         python camulator_sumo_server.py \\"
    echo "             --config  .../camulator_config.yml \\"
    echo "             --model_name checkpoint.pt00091.pt \\"
    echo "             --rundir  ${RUNDIR} \\"
    echo "             --init_cond ${RUNDIR}/sumo_init_1980-01-01T00z.pth \\"
    echo "             --sumo --sumo_vars U V --sumo_tau 6.0 \\"
    echo "             --save_atm_nc camulator_out --daily_mean"
    echo ""
    echo "      4. cd ${CASE_DIR} && ./case.submit"
else
    echo "==> Skipping case.build (run './case.build' manually from $CASE_DIR)"
fi

echo ""
echo "==> Done. Case is at: ${CASE_DIR}"
echo ""
echo "SUMO architecture summary:"
echo "  CAM6 (this case)    : BHIST f09_g17 — active atmosphere driving POP2+CICE"
echo "  CAMulator server    : external GPU server, prescribed ERA5 SSTs (AMIP-style)"
echo "  Exchange            : U,V every 6h via sumo_coordinator.py"
echo "  CAM6 nudging        : CESM nudging toolbox reads sumo_cam6_nudge.*.nc"
echo "  Ocean forcing       : CAM6's nudged atmospheric state -> POP2+CICE"
