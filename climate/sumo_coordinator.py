"""
sumo_coordinator.py
-------------------
SUMO coordinator process.  Runs concurrently with:
  - camulator_sumo_server.py   (CAMulator SUMO server, GPU node, prescribed-SST AMIP mode)
  - A CAM6 BHIST B-compset CESM case (active CAM6 + CLM5 + POP2 + CICE)

python sumo_coordinator.py --rundir  /glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run --cam6dir /glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run --cam6_case g.e21.SUMO_CAM6_v03 --vars U V --alpha_cam 0.5 --start_ymd 19800101

No Python-Fortran bridge or Fortran modifications required.

At each 6-hour coupling step the coordinator:
  1. Waits for CAMulator to write sumo_cam_state.nc + sumo_cam_ready.flag
  2. Reads the timestamp (ymd/tod) from the CAMulator state file
  3. Waits for CAM6's corresponding 6-hourly h1 history file to appear
  4. Reads CAMulator state  (CAMulator grid, ascending lat, physical units)
  5. Reads CAM6 state       (CAM6 grid, ascending lat, physical units)
  6. Remaps CAM6 state to CAMulator grid (if grids differ)
  7. Computes: X_combined = alpha_cam * X_cam + (1-alpha_cam) * X_cam6
  8. Writes sumo_combined_state.nc  (CAMulator grid) -> signals sumo_combine_done.flag
  9. Writes sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc  (CAM6 grid, CESM nudging-toolbox format)
     matched to Nudge_File_Template = 'sumo_cam6_nudge.%y-%m-%d-%s.nc'
     Label = h1_time + 6h, which is the EXACT time the contents are valid at
     (the CAMulator forecast from input at h1_time T is a prediction for T+6h).
     Files accumulate — no deletion — so the full history is always available.
 10. Writes coordinator_done.flag → CESM unpauses from sumo_barrier_after_h1
     (in cam_comp.F90). The barrier guarantees the nudge file is on disk
     BEFORE CAM6 advances past the matching model step, eliminating both the
     legacy T+12h race-avoidance hack and the file-read race condition.

CAM6 is nudged toward the CAMulator forecast via the CESM nudging toolbox
(user_nl_cam: Nudge_Model=.true., Nudge_File_Template=..., Force_Opt=1).
With the F90 barrier in place, nudge file labels match contents exactly:
the file labeled T contains CAMulator state valid at T.

CAMulator is nudged toward the combined state in real-time (camulator_sumo_server.py
waits for sumo_combine_done.flag before advancing its state).

Usage:
  python sumo_coordinator.py \\
      --rundir    /path/to/camulator_rundir/    \\  # where CAMulator flag/state files live
      --cam6dir   /path/to/cam6_rundir/         \\  # where CAM6 h1 files appear
      --vars      U V                           \\
      --alpha_cam 0.5

  For BHIST B-case, both models typically share the same run directory:
  python sumo_coordinator.py \\
      --rundir  ${RUNDIR}  --cam6dir ${RUNDIR}  \\
      --vars U V --alpha_cam 0.5

Optional: --cam6_case <CASENAME> speeds up file lookup; if omitted the
coordinator globs for matching *.cam.h1.YYYY-MM-DD-SSSSS.nc files.

The coordinator runs indefinitely (or until --max_steps).
It can be started before or after the model runs begin.
"""

import os
import sys
import glob
import time
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import netCDF4 as nc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Flag / file names
# =============================================================================

SUMO_CAM_READY_FLAG    = "sumo_cam_ready.flag"
SUMO_CAM_STATE_FILE    = "sumo_cam_state.nc"
SUMO_COMBINE_DONE_FLAG = "sumo_combine_done.flag"
SUMO_COMBINED_FILE     = "sumo_combined_state.nc"
# SST relay file: CAMulator server reads this each step to get CESM ocean SST + ice
# Written on the CAMulator/f09 grid (192×288) — coordinator reads TS/ICEFRAC
# directly from CAM6 h1 which is already on f09; no T62 intermediary needed.
SST_FILE               = "camulator_sst_in.nc"
# go.flag: coordinator writes this to trigger each CAMulator step (B-compset has
# no DATM to write it; coordinator owns the CAMulator clock in SUMO mode).
GO_FLAG                = "camulator_go.flag"
# Dated nudge file: sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc
# Matches Nudge_File_Template = 'sumo_cam6_nudge.%y-%m-%d-%s.nc'
# NOTE: CAM6 nudging.F90 hardcodes Nudge_Climo_Year=1000 (not in namelist).
# All nudge files use year 1000 regardless of simulation year so CESM can find them.
SUMO_CAM6_NUDGE_TEMPLATE = "sumo_cam6_nudge.{y:04d}-{m:02d}-{d:02d}-{s:05d}.nc"
NUDGE_CLIMO_YEAR         = 1000
# Template (clone of first-seen h1) used as the structural source for every nudge file.
# CAM6's FV nudging reader is sticky about netCDF layout (PS rank, time dim, fill values).
# Using a real h1 as template guarantees structure CAM6 already accepts.
NUDGE_TEMPLATE_NAME      = "sumo_nudge_template.nc"
# SUMO barrier handshake (paired with sumo_barrier_after_h1 in cam_comp.F90).
# CAM6 writes CESM_H1_READY_FLAG (with timestamp content) after wshist at every
# 6-hour boundary, then blocks until the coordinator writes COORDINATOR_DONE_FLAG.
# The barrier eliminates the file-availability race AND lets us label nudge files
# at exactly the time their contents are valid (no more T+12h workaround).
CESM_H1_READY_FLAG       = "cesm_h1_ready.flag"
COORDINATOR_DONE_FLAG    = "coordinator_done.flag"
# Sentinel that enables the F90 barrier; coordinator writes it at startup so the
# rundir is marked as SUMO-mode even on bare resubmits.
SUMO_ACTIVE_FLAG         = "sumo_active.flag"

POLL_SLEEP = 0.5          # seconds between file polls
POLL_LOG   = int(60 / POLL_SLEEP)


# =============================================================================
# Helpers
# =============================================================================

def write_flag(path):
    Path(path).touch()


def delete_flag(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def cesm_ymd_tod_to_datetime(ymd, tod):
    """CESM ymd (YYYYMMDD) + tod (seconds) -> Python datetime (no-leap ignored)."""
    y, m, d = ymd // 10000, (ymd % 10000) // 100, ymd % 100
    return datetime(y, m, d) + timedelta(seconds=int(tod))


def dated_nudge_name(ymd, tod):
    """Return the dated nudge filename for a given CESM date.

    Year is always NUDGE_CLIMO_YEAR (1000) because CAM6 nudging.F90 hardcodes
    Nudge_Climo_Year=1000 and is not overridable via namelist.
    """
    m = (ymd % 10000) // 100
    d = ymd % 100
    return SUMO_CAM6_NUDGE_TEMPLATE.format(y=NUDGE_CLIMO_YEAR, m=m, d=d, s=tod)


def wait_for_cesm_h1_ready(cam6dir, cam6_case, expected_ymd, expected_tod, poll_max):
    """
    Poll for the CESM_H1_READY_FLAG written by CAM6's sumo_barrier_after_h1
    after WSHIST completes at a 6-hour boundary. The flag content is the
    timestamp YYYY-MM-DD-SSSSS; validate it matches the step we expect
    (paranoia — should never disagree).

    Then locate the matching h1 file (which is guaranteed on disk because the
    barrier fires AFTER wshist completes) and return its Path.
    """
    ready_path = Path(cam6dir) / CESM_H1_READY_FLAG
    _poll = 0
    while not ready_path.exists():
        time.sleep(POLL_SLEEP)
        _poll += 1
        if _poll > poll_max:
            logger.error(f"Timeout waiting for {CESM_H1_READY_FLAG}")
            sys.exit(1)
        if _poll % POLL_LOG == 0:
            logger.info(f"  Waiting for {CESM_H1_READY_FLAG} "
                        f"(poll {_poll}/{poll_max}) ...")

    # Validate timestamp content
    try:
        ts = ready_path.read_text().strip()
        y_str, m_str, d_str, s_str = ts.split("-")
        flag_ymd = int(y_str) * 10000 + int(m_str) * 100 + int(d_str)
        flag_tod = int(s_str)
    except Exception as e:
        logger.warning(f"  Could not parse timestamp from {ready_path}: {e}")
        flag_ymd, flag_tod = expected_ymd, expected_tod

    if flag_ymd != expected_ymd or flag_tod != expected_tod:
        logger.warning(f"  Flag timestamp mismatch: flag={flag_ymd}/{flag_tod}, "
                       f"expected={expected_ymd}/{expected_tod} — using flag value")
        expected_ymd, expected_tod = flag_ymd, flag_tod

    # Find the matching h1 (already on disk — barrier guarantees this)
    y = expected_ymd // 10000
    m = (expected_ymd % 10000) // 100
    d = expected_ymd % 100
    date_suffix = f"{y:04d}-{m:02d}-{d:02d}-{expected_tod:05d}.nc"
    if cam6_case:
        candidate = Path(cam6dir) / f"{cam6_case}.cam.h1.{date_suffix}"
        if candidate.exists():
            return candidate
    hits = sorted(glob.glob(str(Path(cam6dir) / f"*.cam.h1.{date_suffix}")))
    if hits:
        return Path(hits[0])
    logger.error(f"Ready flag present but no h1 file found for {date_suffix}")
    sys.exit(1)


def find_and_wait_for_cam6_h1(cam6dir, cam6_case, ymd, tod, poll_max):
    """
    Legacy poller — kept for non-barrier mode (e.g. debugging without the F90
    barrier). Polls for a matching CAM6 h1 file by name and returns its Path.

    Filename pattern: <CASENAME>.cam.h1.YYYY-MM-DD-SSSSS.nc
    If cam6_case is None, globs for any *.cam.h1.YYYY-MM-DD-SSSSS.nc.
    """
    y = ymd // 10000
    m = (ymd % 10000) // 100
    d = ymd % 100
    date_suffix = f"{y:04d}-{m:02d}-{d:02d}-{tod:05d}.nc"

    _poll = 0
    while True:
        if cam6_case:
            candidate = Path(cam6dir) / f"{cam6_case}.cam.h1.{date_suffix}"
            if candidate.exists():
                return candidate
        else:
            hits = sorted(glob.glob(str(Path(cam6dir) / f"*.cam.h1.{date_suffix}")))
            if hits:
                return Path(hits[0])

        time.sleep(POLL_SLEEP)
        _poll += 1
        if _poll > poll_max:
            label = (f"{cam6_case}.cam.h1.{date_suffix}"
                     if cam6_case else f"*.cam.h1.{date_suffix}")
            logger.error(f"Timeout waiting for CAM6 h1 file: {label}")
            sys.exit(1)
        if _poll % POLL_LOG == 0:
            label = (f"{cam6_case}.cam.h1.{date_suffix}"
                     if cam6_case else f"*.cam.h1.{date_suffix}")
            logger.info(f"  Waiting for CAM6 h1: {label}  (poll {_poll}/{poll_max}) ...")


def read_sumo_state_nc(path):
    """Read sumo_cam_state.nc written by camulator_sumo_server.py."""
    var_arrays = {}
    with nc.Dataset(str(path), "r") as ds:
        lats = ds.variables["lat"][:].data.astype(np.float64)
        lons = ds.variables["lon"][:].data.astype(np.float64)
        ymd  = int(ds.variables["ymd"][()])
        tod  = int(ds.variables["tod"][()])
        skip = {"lat", "lon", "lev", "ymd", "tod"}
        for vname in ds.variables:
            if vname not in skip:
                var_arrays[vname] = ds.variables[vname][:].data.astype(np.float32)
    return var_arrays, lats, lons, ymd, tod


def read_cam6_h1(path, vars_needed):
    """
    Read selected variables from a CAM6 6-hourly h1 history file.

    Also reads hybrid level coordinates (lev, hyam, hybm, P0) needed to
    write CESM-compatible nudging files.

    Returns
    -------
    var_arrays    : dict[str, np.ndarray]  shape (nlev, nlat, nlon)
    lats          : 1-D ascending float64
    lons          : 1-D float64 in [0, 360)
    hybrid_coords : dict with keys 'lev', 'hyam', 'hybm', 'P0'
                    (or empty if not present in file)
    """
    var_arrays = {}
    hybrid_coords = {}

    with nc.Dataset(str(path), "r") as ds:
        lats_raw = ds.variables["lat"][:].data.astype(np.float64)
        lons_raw = ds.variables["lon"][:].data.astype(np.float64) % 360.0

        flip = lats_raw[0] > lats_raw[-1]
        lats = lats_raw[::-1].copy() if flip else lats_raw.copy()

        # Hybrid level coordinates (kept in model order — descending pressure)
        for coord in ("lev", "hyam", "hybm"):
            if coord in ds.variables:
                hybrid_coords[coord] = ds.variables[coord][:].data.astype(np.float64)
        hybrid_coords["P0"] = (
            float(ds.variables["P0"][()]) if "P0" in ds.variables else 100000.0
        )

        for var in vars_needed:
            if var not in ds.variables:
                logger.warning(f"  '{var}' not found in CAM6 h1 file")
                continue
            data = ds.variables[var][:].data.astype(np.float32)
            # Squeeze time dim
            if data.ndim == 4:
                data = data[0]                         # (lev, lat, lon)
            elif data.ndim == 3:
                data = data[0][np.newaxis, ...]        # (1, lat, lon)
            if flip:
                data = data[:, ::-1, :].copy()
            var_arrays[var] = data

    return var_arrays, lats, lons_raw, hybrid_coords


def write_sst_nc(path, sst_2d, ifrac_2d, lats, lons, ymd, tod):
    """
    Write camulator_sst_in.nc for the CAMulator SUMO server.

    SST and ICEFRAC are on the CAM6/CAMulator f09 grid (192×288), read directly
    from CAM6 h1 history.  The server injects them without any remapping.

    Variables written (matching what camulator_sumo_server.py reads via read_sst_nc):
      sst   : float64 (nlat, nlon)  — surface temp over ocean (K)
      ifrac : float64 (nlat, nlon)  — sea ice fraction [0, 1]
      ymd   : scalar int            — CESM date YYYYMMDD
      tod   : scalar int            — CESM seconds-of-day
    """
    with nc.Dataset(str(path), "w") as ds:
        ds.source = "sumo_coordinator.py — SST/ICEFRAC from CAM6 h1 (f09 grid)"
        ds.createDimension("nlat", sst_2d.shape[0])
        ds.createDimension("nlon", sst_2d.shape[1])

        v = ds.createVariable("lat", "f8", ("nlat",))
        v[:] = lats;  v.units = "degrees_north"
        v = ds.createVariable("lon", "f8", ("nlon",))
        v[:] = lons;  v.units = "degrees_east"

        v = ds.createVariable("sst", "f8", ("nlat", "nlon"))
        v[:] = sst_2d;  v.units = "K";  v.long_name = "sea surface temperature (SST/TS from CAM6 h1)"
        v = ds.createVariable("ifrac", "f8", ("nlat", "nlon"))
        v[:] = ifrac_2d;  v.units = "1";  v.long_name = "sea ice fraction (ICEFRAC from CAM6 h1)"

        v = ds.createVariable("ymd", "i4");  v[()] = ymd
        v = ds.createVariable("tod", "i4");  v[()] = tod


def write_combined_nc(path, var_arrays, lats, lons, ymd, tod, label="coordinator"):
    """Write sumo_combined_state.nc for CAMulator to read (simple format)."""
    nlev = next(iter(var_arrays.values())).shape[0]
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("lev", nlev)
        ds.createDimension("lat", len(lats))
        ds.createDimension("lon", len(lons))
        ds.source = label

        v = ds.createVariable("lat", "f4", ("lat",))
        v[:] = lats.astype(np.float32);  v.units = "degrees_north"

        v = ds.createVariable("lon", "f4", ("lon",))
        v[:] = lons.astype(np.float32);  v.units = "degrees_east"

        v = ds.createVariable("ymd", "i4");  v[()] = ymd
        v = ds.createVariable("tod", "i4");  v[()] = tod

        for varname, data in var_arrays.items():
            v = ds.createVariable(varname, "f4", ("lev", "lat", "lon"))
            v[:] = data.astype(np.float32)


def build_nudge_template_from_h1(h1_path, template_path):
    """
    Create a one-time nudge-file template by cloning a CAM6 h1 file.

    CAM6's FV nudging reader (`nudging_update_analyses_fv`) is sticky about
    netCDF layout — wrong PS rank, fixed-size `time`, or stray `_FillValue`
    silently corrupt the destination buffer and crash the dycore in tp_core.
    Using a real h1 file as the structural template eliminates those mismatches
    by construction: CAM6 wrote the h1 itself, so it accepts the same layout.

    The template is just a verbatim copy — U/V/T/Q/PS data are left in place
    and overwritten cell-by-cell at every nudge step by `write_cam6_nudge_dated_nc`.
    """
    shutil.copy(str(h1_path), str(template_path))


def write_cam6_nudge_dated_nc(nudge_dir, var_arrays, lats, lons,
                               hybrid_coords, ymd, tod,
                               template_path):
    """
    Write a CESM nudging-toolbox-compatible nudge file for CAM6 by cloning
    a known-good template and overwriting only U/V/T/Q/PS.

    Atomic strategy:
        1. shutil.copy(template, <final>.nc.tmp)
        2. open .tmp with mode='r+', overwrite U/V/T/Q/PS data slabs and
           the time/date/datesec coordinates
        3. os.replace(.tmp, <final>.nc)  ← atomic; CAM6 never sees a partial file

    File name: sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc
    Matches user_nl_cam setting:
        Nudge_File_Template = 'sumo_cam6_nudge.%y-%m-%d-%s.nc'

    Parameters
    ----------
    nudge_dir      : directory to write the file (typically CESM RUNDIR)
    var_arrays     : dict[str -> np.ndarray]
                     U/V/T/Q expected as (nlev, nlat, nlon);
                     PS expected as (nlat, nlon) or (1, nlat, nlon)
    lats, lons     : unused (template already has correct coordinates)
    hybrid_coords  : unused (template already has lev/hyam/hybm/P0)
    ymd            : CESM date YYYYMMDD (used in filename + `date` var)
    tod            : CESM seconds-of-day (used in filename + `datesec` var)
    template_path  : Path to nudge template built by build_nudge_template_from_h1.

    Returns the written filename (without directory).
    """
    fname      = dated_nudge_name(ymd, tod)
    final_path = Path(nudge_dir) / fname
    tmp_path   = final_path.with_suffix(final_path.suffix + ".tmp")

    # 1. Clone template
    shutil.copy(str(template_path), str(tmp_path))

    # 2. Overwrite payload + coordinate metadata in place
    with nc.Dataset(str(tmp_path), "r+") as ds:
        for varname, data in var_arrays.items():
            if varname not in ds.variables:
                logger.warning(f"  '{varname}' not in nudge template — skipping")
                continue
            v = ds.variables[varname]
            arr = np.asarray(data, dtype=np.float32)
            if v.ndim == 4 and arr.ndim == 3:
                # (time, lev, lat, lon)  <-  (lev, lat, lon)
                v[0, :, :, :] = arr
            elif v.ndim == 3 and arr.ndim == 2:
                # (time, lat, lon)  <-  (lat, lon)
                v[0, :, :] = arr
            elif v.ndim == 3 and arr.ndim == 3 and arr.shape[0] == 1:
                # (time, lat, lon)  <-  (1, lat, lon)
                v[0, :, :] = arr[0]
            elif v.ndim == arr.ndim:
                v[...] = arr
            else:
                logger.warning(
                    f"  shape mismatch writing '{varname}': "
                    f"template {v.shape} vs data {arr.shape}"
                )

        # Stamp date/datesec for this nudge label so CAM6 timestamp metadata
        # is consistent with the filename (FV reader uses filename for bracket
        # matching, but other diagnostics inspect date/datesec).
        if "date" in ds.variables:
            ds.variables["date"][0] = ymd
        if "datesec" in ds.variables:
            ds.variables["datesec"][0] = tod

    # 3. Atomic rename — file appears whole or not at all
    os.replace(str(tmp_path), str(final_path))
    return fname


# =============================================================================
# Bilinear remap
# =============================================================================

class BilinearRemap:
    def __init__(self, src_lats, src_lons, dst_lats, dst_lons):
        nlat_src = len(src_lats);  nlon_src = len(src_lons)
        lat_g, lon_g = np.meshgrid(dst_lats, dst_lons, indexing="ij")
        flat_lats = lat_g.ravel();  flat_lons = lon_g.ravel() % 360.0
        i0 = np.clip(np.searchsorted(src_lats, flat_lats, side="right") - 1, 0, nlat_src - 2)
        i1 = i0 + 1
        j0 = np.clip(np.searchsorted(src_lons, flat_lons, side="right") - 1, 0, nlon_src - 1)
        j1 = (j0 + 1) % nlon_src
        lon_r = np.where(j0 < nlon_src - 1, src_lons[j1], src_lons[0] + 360.0)
        dlat = src_lats[i1] - src_lats[i0];  dlon = lon_r - src_lons[j0]
        a = np.clip((flat_lats - src_lats[i0]) / np.where(dlat == 0, 1.0, dlat), 0.0, 1.0)
        b = np.clip((flat_lons  - src_lons[j0]) / np.where(dlon == 0, 1.0, dlon), 0.0, 1.0)
        self.i0=i0; self.i1=i1; self.j0=j0; self.j1=j1
        self.w00=((1-a)*(1-b)).astype(np.float32); self.w01=((1-a)*b).astype(np.float32)
        self.w10=(a*(1-b)).astype(np.float32);     self.w11=(a*b).astype(np.float32)
        self.shape_out = (len(dst_lats), len(dst_lons))

    def __call__(self, f):
        return (self.w00*f[self.i0,self.j0] + self.w01*f[self.i0,self.j1]
              + self.w10*f[self.i1,self.j0] + self.w11*f[self.i1,self.j1]).reshape(self.shape_out)

    def remap_3d(self, f3):
        return np.stack([self(f3[k]) for k in range(f3.shape[0])], axis=0)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "SUMO coordinator: file-based state exchange between CAMulator "
            "and CAM6 BHIST. No Fortran bridge required."
        )
    )
    p.add_argument("--rundir", required=True,
                   help="Directory where CAMulator writes sumo_cam_state.nc / "
                        "sumo_cam_ready.flag (also receives sumo_combine_done.flag "
                        "and sumo_combined_state.nc)")
    p.add_argument("--cam6dir", required=True,
                   help="CAM6 CESM run directory where h1 history files appear "
                        "and where dated nudge files (sumo_cam6_nudge.*.nc) are "
                        "written for the CESM nudging toolbox. "
                        "May be the same path as --rundir for a BHIST B-case.")
    p.add_argument("--cam6_case", default=None,
                   help="CAM6 case name prefix (e.g. g.e21.SUMO_CAM6_v01). "
                        "If omitted, the coordinator globs for any "
                        "*.cam.h1.YYYY-MM-DD-SSSSS.nc matching the date.")
    p.add_argument("--vars",        nargs="+", default=["U", "V"],
                   help="Variables to combine (default: U V)")
    p.add_argument("--alpha_cam",   type=float, default=0.5,
                   help="Weight for CAMulator in combined state (default 0.5)")
    p.add_argument("--timeout",     type=float, default=7200.0,
                   help="Max seconds to wait for each model's data (default 7200)")
    p.add_argument("--max_steps",   type=int, default=0,
                   help="Stop after this many steps (0 = run indefinitely)")
    p.add_argument("--model_year_offset", type=int, default=0,
                   help="Year offset: CAM6 year = CAMulator year + offset. "
                        "Use when CAM6 and CAMulator track different calendar years. "
                        "Default 0 (both start from the same year).")
    p.add_argument("--start_ymd", type=int, default=19800101,
                   help="CESM start date YYYYMMDD (default 19800101). "
                        "Coordinator polls for CAM6 h1 starting at this date.")
    p.add_argument("--start_tod", type=int, default=21600,
                   help="CESM start seconds-of-day (default 21600 = 6h). "
                        "CESM writes the first h1 at 6h into the run, not at t=0. "
                        "First h1 expected at start_ymd / start_tod.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    rundir   = Path(args.rundir)
    cam6dir  = Path(args.cam6dir)

    if not rundir.is_dir():
        logger.error(f"--rundir does not exist: {rundir}")
        sys.exit(1)
    if not cam6dir.is_dir():
        logger.error(f"--cam6dir does not exist: {cam6dir}")
        sys.exit(1)

    cam_ready_flag    = rundir / SUMO_CAM_READY_FLAG
    cam_state_file    = rundir / SUMO_CAM_STATE_FILE
    combine_done_flag = rundir / SUMO_COMBINE_DONE_FLAG
    combined_file     = rundir / SUMO_COMBINED_FILE

    _alpha_cam  = args.alpha_cam
    _alpha_cam6 = 1.0 - _alpha_cam
    _poll_max   = int(args.timeout / POLL_SLEEP)
    _vars       = args.vars
    _yr_offset  = args.model_year_offset

    logger.info("=" * 65)
    logger.info("SUMO coordinator starting (file-based, no Fortran bridge)")
    logger.info(f"  CAMulator rundir      : {rundir}")
    logger.info(f"  CAM6 BHIST rundir     : {cam6dir}")
    logger.info(f"  CAM6 case name        : {args.cam6_case or '(auto-detect via glob)'}")
    logger.info(f"  Variables             : {_vars}")
    logger.info(f"  Weights               : CAMulator={_alpha_cam:.2f}  CAM6={_alpha_cam6:.2f}")
    logger.info(f"  Timeout per model     : {args.timeout:.0f}s")
    logger.info(f"  Model year offset     : CAM6 year = CAMulator year + {_yr_offset}")
    logger.info(f"  go.flag               : coordinator writes {GO_FLAG} (B-compset, no DATM)")
    logger.info(f"  Nudge file template   : {SUMO_CAM6_NUDGE_TEMPLATE}")
    logger.info(f"  Nudge climo year      : {NUDGE_CLIMO_YEAR}  (hardcoded in nudging.F90)")
    logger.info(f"  Start date            : ymd={args.start_ymd}  tod={args.start_tod}s")
    logger.info("=" * 65)

    # Cached remap objects (built on first step when both grids are known)
    _cam6_to_cam_remap = None
    _cam_to_cam6_remap = None
    _cam_lats = _cam_lons = None
    _cam6_lats = _cam6_lons = None

    # Cached hybrid level coordinates from first CAM6 h1 read
    _hybrid_coords = {}

    # One-time template used as structural source for every nudge file.
    # Lookup order:
    #   1. Already in cam6dir from a previous run / manual placement → use as-is.
    #   2. Canonical copy shipped next to this script
    #      (camulator_sumo/climate/sumo_nudge_template.nc) → copy into cam6dir.
    #   3. Build from the first h1 the coordinator sees (legacy fallback).
    _nudge_template_path = cam6dir / NUDGE_TEMPLATE_NAME
    _canonical_template  = Path(__file__).resolve().parent / NUDGE_TEMPLATE_NAME
    if not _nudge_template_path.exists() and _canonical_template.exists():
        shutil.copy(str(_canonical_template), str(_nudge_template_path))
        logger.info(f"  Copied canonical nudge template → {_nudge_template_path}")

    # Rolling nudge file cleanup — keep only the two most recent files on disk
    # Nudge files accumulate on disk (no deletion) so CAM6 can always find both bracket files.

    # Current expected h1 timestamp (coordinator leads: h1 → SST → go.flag → CAMulator)
    _next_ymd = args.start_ymd
    _next_tod = args.start_tod

    # Clean stale flags from previous runs (including the barrier handshake pair)
    delete_flag(combine_done_flag)
    delete_flag(rundir / GO_FLAG)
    delete_flag(cam6dir / CESM_H1_READY_FLAG)
    delete_flag(cam6dir / COORDINATOR_DONE_FLAG)

    # Ensure the F90 barrier sentinel exists so cam_run4 enters the SUMO path.
    sumo_active_path = cam6dir / SUMO_ACTIVE_FLAG
    if not sumo_active_path.exists():
        write_flag(sumo_active_path)
        logger.info(f"  Created {SUMO_ACTIVE_FLAG} to enable CAM6 F90 barrier")

    step = 0
    while True:
        logger.info(f"--- Coordinator step {step}  (expecting h1 ymd={_next_ymd} tod={_next_tod}s) ---")

        # ------------------------------------------------------------------
        # 1. Compute CAM6 date for this step (apply year offset if needed)
        # ------------------------------------------------------------------
        cam_year  = _next_ymd // 10000
        cam6_year = cam_year + _yr_offset
        cam6_ymd  = (cam6_year * 10000) + (_next_ymd % 10000)

        # ------------------------------------------------------------------
        # 2. Wait for CAM6 to pause after WSHIST (barrier handshake)
        # ------------------------------------------------------------------
        cam6_h1_path = wait_for_cesm_h1_ready(
            cam6dir, args.cam6_case, cam6_ymd, _next_tod, _poll_max
        )
        logger.info(f"  CAM6 h1 ready (paused): {cam6_h1_path.name}")

        # Brief pause for NFS write-through
        time.sleep(0.5)

        _sst_aux  = ["SST", "TS", "ICEFRAC"]
        _nudge_aux = ["T", "Q", "PS"]   # required by nudging.F90 even when coeff=0
        cam6_arrays, cam6_lats, cam6_lons, hybrid_coords = read_cam6_h1(
            cam6_h1_path, _vars + _sst_aux + _nudge_aux
        )
        _found = list(cam6_arrays.keys())
        logger.info(f"  CAM6 state: vars={_found}  "
                    f"shape={next(iter(cam6_arrays.values())).shape}")

        # Build the nudge template once, from the first h1 we see.
        # The template is a verbatim copy — guarantees layout CAM6 will accept.
        if not _nudge_template_path.exists():
            build_nudge_template_from_h1(cam6_h1_path, _nudge_template_path)
            logger.info(f"  Built nudge template: {_nudge_template_path}")

        # Assign cam_ymd / cam_tod from the expected step (used downstream)
        cam_ymd = _next_ymd
        cam_tod = _next_tod

        # ------------------------------------------------------------------
        # 3. Write camulator_sst_in.nc (SST + ICEFRAC on f09 for CAMulator server)
        #    then trigger CAMulator with go.flag.
        #
        # In B-compset, nothing else writes go.flag (no DATM).  Coordinator owns
        # the CAMulator clock: h1 available → SST ready → go.flag → CAMulator runs.
        # Prefer SST (raw POP value, 0 over land) over TS (full surface temp).
        # ------------------------------------------------------------------
        _sst_field   = cam6_arrays.get("SST") if "SST" in cam6_arrays else cam6_arrays.get("TS")
        _ifrac_field = cam6_arrays.get("ICEFRAC")
        if _sst_field is not None and _ifrac_field is not None:
            sst_2d   = _sst_field[0].astype(np.float64)
            ifrac_2d = _ifrac_field[0].astype(np.float64)
            sst_path = rundir / SST_FILE
            write_sst_nc(sst_path, sst_2d, ifrac_2d, cam6_lats, cam6_lons,
                         cam6_ymd, cam_tod)
            _ocn = sst_2d[sst_2d > 200.0]
            logger.info(f"  Wrote {SST_FILE}: "
                        f"sst=[{sst_2d.min():.1f},{sst_2d.max():.1f}] K  "
                        f"ocn_mean={_ocn.mean():.1f} K  "
                        f"ifrac=[{ifrac_2d.min():.3f},{ifrac_2d.max():.3f}]")
        else:
            logger.warning("  SST/TS or ICEFRAC missing from h1 — skipping camulator_sst_in.nc write")

        # Trigger CAMulator to run this step
        write_flag(rundir / GO_FLAG)
        logger.info(f"  Wrote {GO_FLAG} — waiting for CAMulator to complete step ...")

        # ------------------------------------------------------------------
        # 4. Wait for CAMulator state (server runs, writes sumo_cam_ready.flag)
        # ------------------------------------------------------------------
        _poll = 0
        while not cam_ready_flag.exists():
            time.sleep(POLL_SLEEP)
            _poll += 1
            if _poll > _poll_max:
                logger.error(f"Timeout waiting for {SUMO_CAM_READY_FLAG}. Exiting.")
                sys.exit(1)
            if _poll % POLL_LOG == 0:
                logger.info(f"  Waiting for CAMulator ready flag "
                            f"(poll {_poll}/{_poll_max}) ...")

        delete_flag(cam_ready_flag)
        cam_arrays, cam_lats, cam_lons, _sv_ymd, _sv_tod = read_sumo_state_nc(cam_state_file)
        logger.info(f"  CAMulator state: vars={list(cam_arrays.keys())}  "
                    f"date={_sv_ymd}  tod={_sv_tod}s  "
                    f"shape={next(iter(cam_arrays.values())).shape}")
        if _sv_ymd != cam_ymd or _sv_tod != cam_tod:
            logger.warning(f"  Timestamp mismatch: state={_sv_ymd}/{_sv_tod}  "
                           f"expected={cam_ymd}/{cam_tod} — proceeding with expected")

        # Cache hybrid coords on first successful read
        if not _hybrid_coords and hybrid_coords:
            _hybrid_coords = hybrid_coords
            logger.info(f"  Cached hybrid level coords: "
                        f"nlev={len(hybrid_coords.get('lev', []))}  "
                        f"P0={hybrid_coords.get('P0', 1e5):.0f} Pa")

        # ------------------------------------------------------------------
        # 4. Build remap objects on first step
        # ------------------------------------------------------------------
        if _cam6_to_cam_remap is None and cam6_arrays:
            _cam_lats  = cam_lats
            _cam_lons  = cam_lons
            _cam6_lats = cam6_lats
            _cam6_lons = cam6_lons

            grids_identical = (
                len(cam_lats) == len(cam6_lats)
                and np.allclose(cam_lats, cam6_lats, atol=0.02)
                and len(cam_lons) == len(cam6_lons)
                and np.allclose(cam_lons % 360, cam6_lons % 360, atol=0.02)
            )
            if grids_identical:
                logger.info("  Grids IDENTICAL — no remap needed")
                _cam6_to_cam_remap = "identity"
                _cam_to_cam6_remap = "identity"
            else:
                logger.info(f"  Building remap: "
                            f"CAM6 {len(cam6_lats)}×{len(cam6_lons)} <-> "
                            f"CAMulator {len(cam_lats)}×{len(cam_lons)}")
                _cam6_to_cam_remap = BilinearRemap(cam6_lats, cam6_lons, cam_lats, cam_lons)
                _cam_to_cam6_remap = BilinearRemap(cam_lats, cam_lons, cam6_lats, cam6_lons)
                logger.info("  Remap weights ready")

        # ------------------------------------------------------------------
        # 5. Remap CAM6 state to CAMulator grid
        # ------------------------------------------------------------------
        cam6_on_cam = {}
        for var in _vars:
            if var not in cam6_arrays:
                logger.warning(f"  '{var}' missing from CAM6 — using CAMulator only")
                cam6_on_cam[var] = cam_arrays.get(var)
                continue
            if _cam6_to_cam_remap == "identity":
                cam6_on_cam[var] = cam6_arrays[var].copy()
            else:
                cam6_on_cam[var] = _cam6_to_cam_remap.remap_3d(cam6_arrays[var])

        # ------------------------------------------------------------------
        # 6. Compute combined state
        # ------------------------------------------------------------------
        combined_cam  = {}   # on CAMulator grid (input to camulator_sumo_server)
        combined_cam6 = {}   # combined on CAM6 grid (written to sumo_combined_state for logging)
        cam_nudge     = {}   # CAMulator-only output on CAM6 grid (nudge target for CESM)
        #
        # SUMO design:
        #   combined_cam  → sumo_combined_state.nc → CAMulator runs from this blended input
        #   cam_nudge     → sumo_cam6_nudge*.nc    → CAM6 nudged toward CAMulator forecast
        #
        # Nudging CAM6 toward the pure CAMulator output (not the blended state) means
        # CAM6 is steered toward the prediction that started from the combined input,
        # which is the correct SUMO target.

        for var in _vars:
            arr_cam  = cam_arrays.get(var)
            arr_cam6 = cam6_on_cam.get(var)
            if arr_cam is None or arr_cam6 is None:
                logger.warning(f"  Skipping '{var}' — missing from one model")
                continue

            nlev = min(arr_cam.shape[0], arr_cam6.shape[0])
            comb = _alpha_cam * arr_cam[:nlev] + _alpha_cam6 * arr_cam6[:nlev]
            combined_cam[var] = comb

            if _cam_to_cam6_remap == "identity":
                combined_cam6[var] = comb.copy()
                cam_nudge[var]     = arr_cam[:nlev].copy()
            else:
                combined_cam6[var] = _cam_to_cam6_remap.remap_3d(comb)
                cam_nudge[var]     = _cam_to_cam6_remap.remap_3d(arr_cam[:nlev])

            _delta = np.abs(arr_cam[:nlev] - arr_cam6[:nlev]).mean()
            logger.info(f"  {var}: |ΔCAM-CAM6|={_delta:.4f}  "
                        f"cam={arr_cam.mean():.3f}  cam6={arr_cam6.mean():.3f}  "
                        f"combined={comb.mean():.3f}")

        # Add T, Q, PS to cam_nudge from CAM6 h1 (their nudging coefficients
        # are 0 so these values have no effect, but nudging.F90 unconditionally
        # reads all five variables and calls endrun if any is missing).
        for aux_var in ("T", "Q"):
            if aux_var in cam6_arrays:
                cam_nudge[aux_var] = cam6_arrays[aux_var]
        if "PS" in cam6_arrays:
            ps = cam6_arrays["PS"]
            # PS comes from h1 as (1, lat, lon) via read_cam6_h1; squeeze to (lat, lon)
            cam_nudge["PS"] = ps[0] if ps.ndim == 3 else ps

        # ------------------------------------------------------------------
        # 7. Write combined state for CAMulator
        # ------------------------------------------------------------------
        write_combined_nc(
            combined_file, combined_cam,
            cam_lats, cam_lons, cam_ymd, cam_tod,
            label="sumo_coordinator"
        )
        logger.info(f"  Wrote {SUMO_COMBINED_FILE}")

        # ------------------------------------------------------------------
        # 8. Write dated nudge file for CESM nudging toolbox (CAM6 grid)
        #
        # Label = h1_time + 6h — so the file label EXACTLY MATCHES the time at
        # which its contents are valid. The CAMulator server's prediction came
        # from input at h1_time T and represents the forecast at T+6h, which
        # is what we now label it as.
        #
        # The F90 barrier (sumo_barrier_after_h1 in cam_comp.F90) guarantees
        # this file is on disk BEFORE CAM6 advances past the matching step,
        # so the legacy T+12h race-avoidance hack is no longer needed.
        # ------------------------------------------------------------------
        if cam_nudge:
            c6_lats = _cam6_lats if _cam6_lats is not None else cam6_lats
            c6_lons = _cam6_lons if _cam6_lons is not None else cam6_lons
            _dt_nudge = cesm_ymd_tod_to_datetime(cam_ymd, cam_tod) + timedelta(hours=6)
            _nudge_ymd = _dt_nudge.year * 10000 + _dt_nudge.month * 100 + _dt_nudge.day
            _nudge_tod = _dt_nudge.hour * 3600 + _dt_nudge.minute * 60 + _dt_nudge.second
            nudge_fname = write_cam6_nudge_dated_nc(
                cam6dir, cam_nudge, c6_lats, c6_lons,
                _hybrid_coords, _nudge_ymd, _nudge_tod,
                _nudge_template_path,
            )
            logger.info(f"  Wrote nudge file: {nudge_fname}  (CAMulator forecast, valid at label time)")

        # ------------------------------------------------------------------
        # 9. Signal CAMulator to continue (combine_done) AND CAM6 to unpause
        # ------------------------------------------------------------------
        write_flag(combine_done_flag)
        # Release CAM6 from the F90 barrier and consume its ready flag so the
        # next step's handshake starts from a clean slate.
        write_flag(cam6dir / COORDINATOR_DONE_FLAG)
        delete_flag(cam6dir / CESM_H1_READY_FLAG)
        logger.info(f"  Wrote combine_done.flag + coordinator_done.flag  "
                    f"(step={step}  ymd={cam_ymd}  tod={cam_tod})")

        # ------------------------------------------------------------------
        # 10. Advance expected timestamp by one 6-hour coupling step
        # ------------------------------------------------------------------
        _dt_next = cesm_ymd_tod_to_datetime(_next_ymd, _next_tod) + timedelta(hours=6)
        _next_ymd = _dt_next.year * 10000 + _dt_next.month * 100 + _dt_next.day
        _next_tod = _dt_next.hour * 3600 + _dt_next.minute * 60 + _dt_next.second

        step += 1
        if args.max_steps > 0 and step >= args.max_steps:
            logger.info(f"Reached max_steps={args.max_steps}. Coordinator exiting.")
            break

    logger.info("SUMO coordinator done.")


if __name__ == "__main__":
    main()
