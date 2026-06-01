"""
camulator_sumo_server.py
------------------------
CAMulator coupling server extended with the SUMO (supermodel) protocol.

python camulator_sumo_server.py --config   camulator_config.yml --model_name checkpoint.pt00044.pt --rundir   /glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run --init_cond /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/init_times/sumo_init_camulator_condition_tensor_1980-01-01T00Z.pth --sumo --sumo_vars U V --sumo_tau 6.0


At each 6-hour coupling step, after CAMulator inference:
  1. Writes its predicted U,V (physical) to   sumo_cam_state.nc + sumo_cam_ready.flag
  2. Waits for                                 sumo_combine_done.flag
  3. Reads the combined state from             sumo_combined_state.nc
  4. Blends (nudges) the prediction toward the combined state before advancing

The SUMO coordinator (sumo_coordinator.py) runs concurrently and:
  - Waits for  sumo_cam_ready.flag  (this server)
  - Waits for  sumo_cam6_ready.flag (CAM6 via SuperModel_CAM or history)
  - Computes combined state = alpha_cam * U_cam + alpha_cam6 * U_cam6
  - Writes sumo_combined_state.nc + sumo_combine_done.flag

Supermodel nudging (snapshot nudging, matching Chapman et al. 2025):
  dX/dt = F(X) + (X_combined - X) / tau
  With tau = Δt = 6h (snapshot nudging): X_nudged = X_combined
  General:                                X_nudged = (1 - α)*X_pred + α*X_combined
  where α = min(1, Δt / tau)

SUMO is activated by passing --sumo.  Without --sumo, the server behaves
identically to camulator_server.py (drop-in replacement).

Usage:
  python camulator_sumo_server.py \\
      --config     ./camulator_config.yml \\
      --model_name checkpoint.pt00091.pt \\
      --rundir     /path/to/cesm_run/ \\
      --sumo \\
      --sumo_tau   6.0 \\
      --sumo_vars  U V \\
      --sumo_timeout 7200

See climate/README_Coupling.md for full documentation.
"""

import os
import sys
import time
import shutil
import argparse
import logging
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import netCDF4 as nc
import yaml
from scipy.special import roots_legendre

from credit.output import make_xarray
from Model_State import initialize_camulator, StateVariableAccessor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants — base coupling protocol (same as camulator_server.py)
# =============================================================================

GO_FLAG      = "camulator_go.flag"
DONE_FLAG    = "camulator_done.flag"
SST_FILE     = "camulator_sst_in.nc"
CAM_FILE     = "camulator_cam_out.nc"
ATM_RESTART  = "camulator_atm_restart.pth"
READY_FLAG   = "camulator_server_ready.flag"

POLL_SLEEP = 0.1
POLL_MAX   = int(3 * 3600 / POLL_SLEEP)
POLL_LOG   = int(60   / POLL_SLEEP)

T62_NLAT  = 94
T62_NLON  = 192
T62_NGRID = T62_NLAT * T62_NLON

CAM_NLAT = 192
CAM_NLON = 288

DT_SEC = 21600.0

OCEAN_MIN_K   = 270.0
LAND_SST_FILL = 283.0

# =============================================================================
# Constants — SUMO exchange protocol
# =============================================================================

SUMO_CAM_STATE_FILE    = "sumo_cam_state.nc"      # written by this server
SUMO_CAM_READY_FLAG    = "sumo_cam_ready.flag"    # written by this server
SUMO_COMBINE_DONE_FLAG = "sumo_combine_done.flag"  # written by coordinator
SUMO_COMBINED_FILE     = "sumo_combined_state.nc"  # written by coordinator


# =============================================================================
# Grid helpers (identical to camulator_server.py)
# =============================================================================

def t62_latlons():
    mu, _ = roots_legendre(T62_NLAT)
    lats = np.degrees(np.arcsin(mu))
    lons = np.linspace(0.0, 360.0, T62_NLON, endpoint=False)
    return lats, lons


class BilinearRemap:
    def __init__(self, src_lats, src_lons, dst_lats, dst_lons):
        nlat_src = len(src_lats)
        nlon_src = len(src_lons)

        lat_g, lon_g = np.meshgrid(dst_lats, dst_lons, indexing="ij")
        flat_lats = lat_g.ravel()
        flat_lons = lon_g.ravel() % 360.0

        i0 = np.clip(
            np.searchsorted(src_lats, flat_lats, side="right") - 1,
            0, nlat_src - 2
        )
        i1 = i0 + 1

        j0 = np.clip(
            np.searchsorted(src_lons, flat_lons, side="right") - 1,
            0, nlon_src - 1
        )
        j1 = (j0 + 1) % nlon_src

        lon_right = np.where(j0 < nlon_src - 1, src_lons[j1], src_lons[0] + 360.0)

        dlat = src_lats[i1] - src_lats[i0]
        dlon = lon_right - src_lons[j0]
        a = np.clip((flat_lats - src_lats[i0]) / np.where(dlat == 0, 1.0, dlat), 0.0, 1.0)
        b = np.clip((flat_lons  - src_lons[j0]) / np.where(dlon == 0, 1.0, dlon), 0.0, 1.0)

        self.i0 = i0;  self.i1 = i1
        self.j0 = j0;  self.j1 = j1
        self.w00 = ((1 - a) * (1 - b)).astype(np.float32)
        self.w01 = ((1 - a) * b      ).astype(np.float32)
        self.w10 = (a       * (1 - b)).astype(np.float32)
        self.w11 = (a       * b      ).astype(np.float32)
        self.shape_out = (len(dst_lats), len(dst_lons))
        self.ndst      = len(flat_lats)

    def __call__(self, field):
        return (
            self.w00 * field[self.i0, self.j0]
          + self.w01 * field[self.i0, self.j1]
          + self.w10 * field[self.i1, self.j0]
          + self.w11 * field[self.i1, self.j1]
        ).reshape(self.shape_out)

    def batch(self, fields):
        return (
            self.w00 * fields[:, self.i0, self.j0]
          + self.w01 * fields[:, self.i0, self.j1]
          + self.w10 * fields[:, self.i1, self.j0]
          + self.w11 * fields[:, self.i1, self.j1]
        )


# =============================================================================
# NetCDF I/O
# =============================================================================

def read_sst_nc(path):
    with nc.Dataset(str(path), "r") as ds:
        sst   = ds["sst"][:].data.astype(np.float64)
        ifrac = ds["ifrac"][:].data.astype(np.float64)
        ymd   = int(ds["ymd"][:])
        tod   = int(ds["tod"][:])
    return sst, ifrac, ymd, tod


def write_cam_nc(path, u10, v10, tbot, zbot, tref, qbot, pbot, fsds, fsus, flds, flus, prect):
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("ngrid", T62_NGRID)

        def mkvar(name, data, units, long_name):
            v = ds.createVariable(name, "f8", ("ngrid",))
            v[:] = data
            v.units = units
            v.long_name = long_name

        mkvar("u10",   u10,   "m s-1",  "Zonal wind at CAMulator bottom model level (~60 m)")
        mkvar("v10",   v10,   "m s-1",  "Meridional wind at CAMulator bottom model level (~60 m)")
        mkvar("tbot",  tbot,  "K",      "Temperature at CAMulator bottom model level (Sa_tbot)")
        mkvar("zbot",  zbot,  "m",      "Height of bottom model level midpoint (Sa_z, dynamic ~50-67 m)")
        mkvar("tref",  tref,  "K",      "TREFHT 2 m reference temperature (diagnostic only)")
        mkvar("qbot",  qbot,  "kg kg-1","Specific humidity at CAMulator bottom model level (Sa_shum)")
        mkvar("pbot",  pbot,  "Pa",     "Surface pressure (PS -> Sa_pbot)")
        mkvar("fsds",  fsds,  "W m-2",  "Downwelling SW at surface (FSDS_J direct from model)")
        mkvar("fsus",  fsus,  "W m-2",  "Upwelling SW at surface (FSUS direct from model)")
        mkvar("flds",  flds,  "W m-2",  "Downwelling LW at surface (FLDS_J direct from model)")
        mkvar("flus",  flus,  "W m-2",  "Upwelling LW at surface (FLUS direct from model)")
        mkvar("prect", prect, "m s-1",  "Total precip, liquid-water equivalent")


def cesm_ymd_tod_to_dt(ymd, tod):
    y, m, d = ymd // 10000, (ymd % 10000) // 100, ymd % 100
    return datetime(y, m, d) + timedelta(seconds=int(tod))


def write_flag(path):
    Path(path).touch()


def delete_flag(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# =============================================================================
# SUMO-specific I/O
# =============================================================================

def write_sumo_state_nc(path, var_arrays, lats, lons, lev_indices, ymd, tod):
    """
    Write CAMulator's predicted state variables to sumo_cam_state.nc.

    Parameters
    ----------
    path       : str or Path  — output file
    var_arrays : dict[str, np.ndarray]  — physical-unit arrays, shape (nlev, nlat, nlon)
                 e.g. {"U": ..., "V": ..., "T": ..., "Q": ...}
    lats       : 1-D array, ascending lat (S->N)
    lons       : 1-D array, 0->360 lon
    lev_indices: list of int — level indices (for reference; not pressure levels)
    ymd        : int  YYYYMMDD  (CESM model date)
    tod        : int  seconds since midnight
    """
    nlev = next(iter(var_arrays.values())).shape[0]
    nlat = len(lats)
    nlon = len(lons)

    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("lev", nlev)
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)

        v = ds.createVariable("lat", "f4", ("lat",))
        v[:] = lats.astype(np.float32)
        v.units = "degrees_north"

        v = ds.createVariable("lon", "f4", ("lon",))
        v[:] = lons.astype(np.float32)
        v.units = "degrees_east"

        v = ds.createVariable("ymd", "i4")
        v[()] = ymd
        v.long_name = "CESM model date YYYYMMDD"

        v = ds.createVariable("tod", "i4")
        v[()] = tod
        v.long_name = "CESM model time-of-day seconds"

        for varname, data in var_arrays.items():
            v = ds.createVariable(varname, "f4", ("lev", "lat", "lon"))
            v[:] = data.astype(np.float32)
            v.long_name = f"{varname} physical units"

        ds.source = "camulator_sumo_server"
        ds.grid = f"{nlat}lat x {nlon}lon, ascending lat"


def read_sumo_combined_nc(path):
    """
    Read combined state from sumo_combined_state.nc written by sumo_coordinator.py.

    Returns
    -------
    var_arrays : dict[str, np.ndarray]  — physical-unit arrays, shape (nlev, nlat, nlon)
                 on the CAMulator grid (ascending lat)
    ymd        : int  YYYYMMDD
    tod        : int  seconds since midnight
    """
    var_arrays = {}
    with nc.Dataset(str(path), "r") as ds:
        ymd = int(ds["ymd"][()])
        tod = int(ds["tod"][()])
        skip = {"lat", "lon", "lev", "ymd", "tod"}
        for vname in ds.variables:
            if vname not in skip:
                var_arrays[vname] = ds[vname][:].data.astype(np.float32)
    return var_arrays, ymd, tod


# =============================================================================
# Atmospheric NC save helpers (top-level for mp.Pool)
# =============================================================================

def save_camulator_step_nc(upper_air, single_level, filepath):
    import os
    import xarray as xr
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ds = xr.merge([upper_air.to_dataset(dim="vars"),
                   single_level.to_dataset(dim="vars")])
    ds.to_netcdf(filepath, mode="w")


def save_camulator_daily_nc(buffer, filepath):
    import os
    import xarray as xr
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    datasets = [xr.merge([ua.to_dataset(dim="vars"), sl.to_dataset(dim="vars")])
                for ua, sl in buffer]
    ds_daily = xr.concat(datasets, dim="time").mean("time", keep_attrs=True)
    ds_daily.to_netcdf(filepath, mode="w")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CAMulator SUMO inference server — supermodel extension of camulator_server.py"
    )
    p.add_argument("--config",     required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--rundir",     required=True,
                   help="CESM run directory (go/done flags + SST files)")
    p.add_argument("--init_cond",  default=None)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--save_atm_nc", default=None, metavar="SUBDIR")
    p.add_argument("--daily_mean", action="store_true")
    p.add_argument("--save_vars",  nargs="+", default=None, metavar="VAR")
    p.add_argument("--co2_trend",  type=float, default=0.0, metavar="PPM_PER_YR")

    # --- SUMO-specific args ---
    g = p.add_argument_group("SUMO supermodel options")
    g.add_argument("--sumo", action="store_true",
                   help="Activate SUMO protocol: exchange state with CAM6 every coupling step")
    g.add_argument("--sumo_vars", nargs="+", default=["U", "V"], metavar="VAR",
                   help="Which variables to exchange (default: U V). "
                        "Can include U V T Q PS. Must be output variables of CAMulator.")
    g.add_argument("--sumo_tau", type=float, default=6.0, metavar="HOURS",
                   help="Nudging relaxation timescale τ in hours (default 6h = snapshot). "
                        "α = min(1, Δt/τ) where Δt = 6h.")
    g.add_argument("--sumo_alpha_cam", type=float, default=0.5, metavar="WEIGHT",
                   help="CAMulator weight in combined state (default 0.5). "
                        "NOTE: this value must match --alpha_cam passed to sumo_coordinator.py. "
                        "It is logged here for reference only; the actual blending is done "
                        "by the coordinator before writing sumo_combined_state.nc.")
    g.add_argument("--sumo_timeout", type=float, default=7200.0, metavar="SECONDS",
                   help="Max seconds to wait for sumo_combine_done.flag before aborting (default 7200 = 2h).")

    return p.parse_args()


# =============================================================================
# SUMO nudging helper
# =============================================================================

def apply_sumo_nudging(prediction, combined_arrays, sumo_vars, alpha,
                       accessor_out, state_transformer, cam_flip, device):
    """
    Apply SUMO nudging to the prediction tensor (in-place).

    Nudging formula (normalized space):
        X_nudged = (1 - α) * X_pred + α * X_combined
    With α = 1 (snapshot, τ = Δt):  X_nudged = X_combined

    Parameters
    ----------
    prediction     : torch.Tensor  — model output (normalized), shape (1, C, 1, lat, lon)
    combined_arrays: dict[str, np.ndarray] — physical combined state, shape (nlev, nlat, nlon)
                     on CAMulator grid (ascending lat, written by coordinator)
    sumo_vars      : list[str]  — variables to nudge
    alpha          : float  — nudging weight [0, 1]
    accessor_out   : StateVariableAccessor  — output accessor
    state_transformer : object with mean_tensors / std_tensors dicts
    cam_flip       : bool  — True if native CAM grid is N->S
    device         : torch.device
    """
    for var in sumo_vars:
        if var not in combined_arrays:
            logger.warning(f"  SUMO: combined state missing '{var}' — skipping nudge for this var")
            continue

        combined_phys = combined_arrays[var]  # (nlev_comb, nlat, nlon), ascending lat

        # combined_phys is in ascending lat (written by coordinator).
        # If native CAM is N->S, flip so it matches the prediction tensor's lat order.
        if cam_flip:
            combined_phys = combined_phys[:, ::-1, :]

        # Normalize combined state using CAMulator's own statistics
        mean_v = state_transformer.mean_tensors[var]
        std_v  = state_transformer.std_tensors[var]
        if isinstance(mean_v, torch.Tensor):
            mean_v = mean_v.cpu().numpy()
            std_v  = std_v.cpu().numpy()
        # Reshape per-level stats (nlev,) → (nlev,1,1) for broadcast with (nlev,nlat,nlon)
        if hasattr(mean_v, "shape") and mean_v.ndim == 1 and mean_v.shape[0] > 1:
            mean_v = mean_v.reshape(-1, 1, 1)
            std_v  = std_v.reshape(-1, 1, 1)

        combined_norm = (combined_phys - mean_v) / std_v  # (nlev_comb, nlat, nlon)

        # Current prediction (normalized): (1, nlev_pred, 1, nlat, nlon)
        pred_norm = accessor_out.get_state_var(prediction, var)
        nlev_pred = pred_norm.shape[1]
        nlev_comb = combined_norm.shape[0]

        # Guard against vertical level mismatch (coordinator takes min of both models)
        if nlev_comb != nlev_pred:
            nlev = min(nlev_pred, nlev_comb)
            logger.warning(f"  SUMO: {var} level mismatch "
                           f"pred={nlev_pred} combined={nlev_comb} — using bottom {nlev} levels")
            # Align bottom levels (index -nlev:) — model levels are top-to-bottom
            combined_norm = combined_norm[-nlev:]
            pred_slice    = pred_norm[:, -nlev:, :, :, :]
            comb_t = (
                torch.from_numpy(combined_norm.copy())
                .float().to(device).unsqueeze(0).unsqueeze(2)
            )
            nudged_slice = (1.0 - alpha) * pred_slice + alpha * comb_t
            # Write back only the affected level slice
            full_nudged = pred_norm.clone()
            full_nudged[:, -nlev:, :, :, :] = nudged_slice
            accessor_out.set_state_var(prediction, var, full_nudged)
        else:
            comb_t = (
                torch.from_numpy(combined_norm.copy())
                .float().to(device).unsqueeze(0).unsqueeze(2)
            )  # (1, nlev, 1, nlat, nlon)
            nudged = (1.0 - alpha) * pred_norm + alpha * comb_t
            accessor_out.set_state_var(prediction, var, nudged)

    return prediction


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    rundir = Path(args.rundir)
    if not rundir.is_dir():
        logger.error(f"rundir does not exist: {rundir}")
        sys.exit(1)

    go_flag   = rundir / GO_FLAG
    done_flag = rundir / DONE_FLAG
    sst_file  = rundir / SST_FILE
    cam_file  = rundir / CAM_FILE

    # SUMO file paths (all in rundir)
    _sumo_enabled = args.sumo
    if _sumo_enabled:
        sumo_cam_state_file    = rundir / SUMO_CAM_STATE_FILE
        sumo_cam_ready_flag    = rundir / SUMO_CAM_READY_FLAG
        sumo_combine_done_flag = rundir / SUMO_COMBINE_DONE_FLAG
        sumo_combined_file     = rundir / SUMO_COMBINED_FILE
        _sumo_tau_hr   = args.sumo_tau
        _sumo_alpha    = min(1.0, 6.0 / _sumo_tau_hr)    # nudging coefficient
        _sumo_wt_cam   = args.sumo_alpha_cam              # weight of CAMulator in combined state
        _sumo_vars     = args.sumo_vars
        _sumo_timeout  = args.sumo_timeout
        _sumo_poll_max = int(_sumo_timeout / POLL_SLEEP)

    _atm_restart_archive_dir = rundir / "atm_restarts"
    _atm_restart_archive_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Initialize CAMulator
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("CAMulator SUMO server starting")
    logger.info(f"  rundir     : {rundir}")
    logger.info(f"  model_name : {args.model_name}")
    logger.info(f"  device     : {args.device}")
    logger.info(f"  SUMO       : {'ENABLED' if _sumo_enabled else 'disabled'}")
    if _sumo_enabled:
        logger.info(f"  SUMO vars  : {_sumo_vars}")
        logger.info(f"  SUMO τ     : {_sumo_tau_hr}h  → α = {_sumo_alpha:.4f}")
        logger.info(f"  SUMO wt    : CAMulator={_sumo_wt_cam:.2f}  CAM6={1-_sumo_wt_cam:.2f}")
    logger.info("=" * 65)

    config_path = args.config
    _tmp_config = None

    if args.init_cond:
        logger.info(f"Overriding init_cond_fast_climate -> {args.init_cond}")
        with open(args.config) as f:
            raw_conf = yaml.safe_load(f)
        raw_conf["predict"]["init_cond_fast_climate"] = args.init_cond
        _tmp_config = str(Path(args.config).parent / "_tmp_sumo_server_config.yml")
        with open(_tmp_config, "w") as f:
            yaml.dump(raw_conf, f)
        config_path = _tmp_config

    ctx = initialize_camulator(config_path, model_name=args.model_name, device=args.device)

    if _tmp_config and os.path.exists(_tmp_config):
        os.remove(_tmp_config)

    stepper           = ctx["stepper"]
    state             = ctx["initial_state"]
    state_transformer = ctx["state_transformer"]
    forcing_ds_norm   = ctx["forcing_dataset"]
    static_forcing    = ctx["static_forcing"]
    conf              = ctx["conf"]
    device            = ctx["device"]
    latlons           = ctx["latlons"]

    accessor_input  = StateVariableAccessor(conf, tensor_type="input")
    accessor_output = StateVariableAccessor(conf, tensor_type="output")

    # ------------------------------------------------------------------
    # 1b. Check for atmosphere restart
    # ------------------------------------------------------------------
    atm_restart_file = rundir / ATM_RESTART
    timestep_init = 0
    _expected_first_ymd = -1
    _expected_first_tod = -1
    _restart_last_ymd   = -1
    _restart_last_tod   = -1
    _restart_cam_out    = None
    if atm_restart_file.exists():
        logger.info(f"ATM restart found: {atm_restart_file}")
        _ckpt         = torch.load(atm_restart_file, map_location=device)
        state         = _ckpt["state"].to(device)
        timestep_init = int(_ckpt["timestep"])
        logger.info(f"  Resuming atmosphere from step {timestep_init}")
        _last_ymd = int(_ckpt.get("last_ymd", -1))
        _last_tod = int(_ckpt.get("last_tod", -1))
        if _last_ymd > 0:
            _restart_last_ymd   = _last_ymd
            _restart_last_tod   = _last_tod
            _next_dt            = cesm_ymd_tod_to_dt(_last_ymd, _last_tod) + timedelta(seconds=DT_SEC)
            _expected_first_ymd = _next_dt.year * 10000 + _next_dt.month * 100 + _next_dt.day
            _expected_first_tod = _next_dt.hour * 3600 + _next_dt.minute * 60 + _next_dt.second
        _restart_cam_out = _ckpt.get("cam_out", None)
    else:
        logger.info("No ATM restart — starting from IC")

    # ------------------------------------------------------------------
    # 2. Grid setup
    # ------------------------------------------------------------------
    t62_lats, t62_lons = t62_latlons()

    cam_lats_raw = latlons.latitude.values.copy()
    cam_lons     = latlons.longitude.values.copy()
    cam_lats_asc = np.sort(cam_lats_raw)
    cam_flip     = cam_lats_raw[0] > cam_lats_raw[-1]

    logger.info(f"T62 grid      : {T62_NLAT}×{T62_NLON} = {T62_NGRID} pts")
    logger.info(f"CAMulator grid: {len(cam_lats_asc)}×{len(cam_lons)}")
    logger.info(f"CAM lat order : {'N->S (will flip)' if cam_flip else 'S->N (OK)'}")

    _cam_lat_w = np.cos(np.radians(cam_lats_asc))[:, None]
    _cam_lat_w = _cam_lat_w / _cam_lat_w.mean()

    logger.info("Precomputing bilinear remap weights ...")
    _t_remap_init = time.time()
    # t62_to_cam_remap removed: SST/ICEFRAC inputs now come from CAM6 h1 on f09
    # (coordinator writes camulator_sst_in.nc on f09 = CAMulator grid directly)
    cam_to_t62_remap = BilinearRemap(cam_lats_asc, cam_lons, t62_lats, t62_lons)
    logger.info(f"  Remap weights ready in {time.time()-_t_remap_init:.2f}s")

    # ------------------------------------------------------------------
    # 3. Normalization scalars
    # ------------------------------------------------------------------
    sst_mean     = float(state_transformer.mean_tensors["SST"])
    sst_std      = float(state_transformer.std_tensors["SST"])
    icefrac_mean = float(state_transformer.mean_tensors["ICEFRAC"])
    icefrac_std  = float(state_transformer.std_tensors["ICEFRAC"])
    logger.info(f"SST scaler    : mean={sst_mean:.3f} K, std={sst_std:.3f} K")

    _co2_trend_ppm_yr = args.co2_trend
    if _co2_trend_ppm_yr != 0.0:
        co2_mean = float(state_transformer.mean_tensors["co2vmr_3d"])
        co2_std  = float(state_transformer.std_tensors["co2vmr_3d"])
        _co2_base_ppm          = co2_mean * 1e6
        _co2_trend_norm_per_yr = (_co2_trend_ppm_yr * 1e-6) / co2_std
        _co2_ref_step          = timestep_init
    else:
        _co2_trend_norm_per_yr = 0.0
        _co2_ref_step          = 0

    # ------------------------------------------------------------------
    # 3b. Bottom level height scale
    # ------------------------------------------------------------------
    _statics_path = conf["data"]["save_loc_static"]
    with nc.Dataset(_statics_path, "r") as _ds:
        _hybm_bot = float(_ds.variables["hybm"][-1])
        _hyam_bot = float(_ds.variables["hyam"][-1])
    Z_BOT_SCALE = (287.058 / 9.80616) * (-np.log(_hybm_bot))
    logger.info(f"Z_BOT_SCALE   : {Z_BOT_SCALE:.6f} m/K")

    # ------------------------------------------------------------------
    # 4. Forcing dataset navigation
    # ------------------------------------------------------------------
    df_vars    = conf["data"]["dynamic_forcing_variables"]
    dynamic_ds = forcing_ds_norm[df_vars]
    start_raw  = conf["predict"]["start_datetime"]
    model_start_dt = datetime.strptime(start_raw, "%Y-%m-%d %H:%M:%S")

    _forcing_years = sorted({t.year for t in dynamic_ds.indexes["time"]})
    _cyclic_forcing_year = _forcing_years[0] if len(_forcing_years) == 1 else None

    # For cyclic forcing (single year), map start_raw to the cyclic year for the
    # initial index lookup — start_datetime may be from a different year (e.g. 1980
    # with a year-2000 cyclic file).
    if _cyclic_forcing_year is not None:
        _start_lookup = f"{_cyclic_forcing_year}{start_raw[4:]}"
    else:
        _start_lookup = start_raw
    loc      = dynamic_ds.indexes["time"].get_loc(_start_lookup)
    start_ix = loc.start if isinstance(loc, slice) else loc
    _cftime_type = type(dynamic_ds.indexes["time"][start_ix])
    _n_forcing_steps     = len(dynamic_ds.indexes["time"])

    def _next_forcing_ix(ix):
        if _cyclic_forcing_year is not None:
            return (ix + 1) % _n_forcing_steps
        return ix + 1

    def cesm_to_forcing_ix(ymd, tod):
        real_year = (
            _cyclic_forcing_year
            if _cyclic_forcing_year is not None
            else model_start_dt.year + (ymd // 10000) - 1
        )
        real_dt = _cftime_type(
            real_year,
            (ymd % 10000) // 100,
            ymd % 100,
            tod // 3600,
        )
        idx = dynamic_ds.indexes["time"].get_loc(real_dt)
        return idx.start if isinstance(idx, slice) else int(idx)

    if _cyclic_forcing_year is not None and timestep_init > 0 and _expected_first_ymd > 0:
        _prefetch_ix = cesm_to_forcing_ix(_expected_first_ymd, _expected_first_tod)
    else:
        _prefetch_ix = start_ix + timestep_init

    _forcing_executor = ThreadPoolExecutor(max_workers=1)

    def _load_forcing_slice(ix):
        ds_slice = dynamic_ds.isel(time=ix).load()
        arr = np.stack([ds_slice[v].values for v in df_vars], axis=0)
        return torch.from_numpy(arr.copy()).float().unsqueeze(0).unsqueeze(2)

    _prefetch_future = _forcing_executor.submit(_load_forcing_slice, _prefetch_ix)

    # ------------------------------------------------------------------
    # 5. JIT trace
    # ------------------------------------------------------------------
    logger.info("Tracing model with torch.jit.trace ...")
    if timestep_init == 0:
        trace_input = state.float()
    else:
        if _cyclic_forcing_year is not None and _expected_first_ymd > 0:
            _trace_ix = cesm_to_forcing_ix(_expected_first_ymd, _expected_first_tod)
        else:
            _trace_ix = start_ix + timestep_init
        _trace_ds  = dynamic_ds.isel(time=_trace_ix).load()
        _trace_arr = np.stack([_trace_ds[v].values for v in df_vars], axis=0)
        _trace_forcing = (
            torch.from_numpy(_trace_arr.copy())
            .float().unsqueeze(0).unsqueeze(2).to(device)
        )
        trace_input = stepper.state_manager.build_input_with_forcing(
            state, _trace_forcing, static_forcing
        ).float()
    stepper.model = torch.jit.trace(stepper.model, trace_input)
    logger.info(f"Model traced  (input shape: {list(trace_input.shape)})")

    # ------------------------------------------------------------------
    # 5b. Persistent IC SST fallback
    # ------------------------------------------------------------------
    if timestep_init == 0:
        with torch.no_grad():
            sst_ic_norm = accessor_input.get_state_var(state, "SST")[0, 0, 0].cpu().numpy()
        sst_cam_persistent = sst_ic_norm * sst_std + sst_mean
        sst_cam_persistent = np.where(sst_cam_persistent < OCEAN_MIN_K,
                                      LAND_SST_FILL, sst_cam_persistent)
    else:
        sst_cam_persistent = np.full((CAM_NLAT, CAM_NLON), LAND_SST_FILL, dtype=np.float64)

    # ------------------------------------------------------------------
    # 6. Clean up stale flags
    # ------------------------------------------------------------------
    delete_flag(go_flag)
    delete_flag(done_flag)
    delete_flag(rundir / READY_FLAG)
    if _sumo_enabled:
        delete_flag(sumo_cam_ready_flag)
        delete_flag(sumo_combine_done_flag)
        # Do NOT delete sumo_cam_state.nc / sumo_combined_state.nc — coordinator handles those

    write_flag(rundir / READY_FLAG)
    logger.info("")
    logger.info("Server ready — waiting for CESM go.flag ...")
    if _sumo_enabled:
        logger.info(f"  SUMO coordinator should also be running in rundir: {rundir}")
    logger.info("")

    # ------------------------------------------------------------------
    # 7b. Atmospheric NC output pool
    # ------------------------------------------------------------------
    _save_nc     = args.save_atm_nc is not None
    _save_nc_dir = rundir / args.save_atm_nc if _save_nc else None
    _do_daily    = _save_nc and args.daily_mean
    _cam_lat     = latlons.latitude.values
    _cam_lon     = latlons.longitude.values
    pool         = None
    if _save_nc:
        _save_nc_dir.mkdir(parents=True, exist_ok=True)
        pool = mp.Pool(4)
        logger.info(f"Atm NC output      : {_save_nc_dir}")

    # ==================================================================
    # 8. Main coupling loop
    # ==================================================================
    timestep          = timestep_init
    _prev_ymd         = -1
    _prev_tod         = -1
    _prev_step_year   = -1
    _cached_cam_out   = None
    _daily_buffer     = []
    _daily_buffer_ymd = -1

    while True:

        # --- (a) wait for go.flag ---
        poll = 0
        while not go_flag.exists():
            time.sleep(POLL_SLEEP)
            poll += 1
            if poll > POLL_MAX:
                logger.error("Timed out waiting for camulator_go.flag. Exiting.")
                if pool:
                    pool.close()
                    pool.join()
                sys.exit(1)
            if poll % POLL_LOG == 0:
                logger.info(f"  Waiting for go.flag (poll {poll}/{POLL_MAX}) ...")

        t_step_start = time.time()
        logger.info(f"Step {timestep:04d}: received go.flag")
        delete_flag(go_flag)

        # --- (b) read SST from CESM ---
        sst_flat, ifrac_flat, ymd, tod = read_sst_nc(sst_file)
        sst_ocn = sst_flat[sst_flat >= OCEAN_MIN_K]
        ocn_mean = sst_ocn.mean() if len(sst_ocn) > 0 else 0.0
        logger.info(f"  SST  min={sst_flat.min():.1f} K  max={sst_flat.max():.1f} K  "
                    f"ocn_mean={ocn_mean:.1f} K  date={ymd}  tod={tod}s")

        # --- CONTINUE_RUN re-send detection ---
        if (timestep == timestep_init and _restart_last_ymd > 0
                and ymd == _restart_last_ymd and tod == _restart_last_tod):
            if _restart_cam_out is not None:
                logger.info(f"  CONTINUE_RUN re-send: re-serving saved cam_out, skipping inference")
                write_cam_nc(cam_file,
                             _restart_cam_out["u10"], _restart_cam_out["v10"],
                             _restart_cam_out["tbot"], _restart_cam_out["zbot"],
                             _restart_cam_out["tref"], _restart_cam_out["qbot"],
                             _restart_cam_out["pbot"], _restart_cam_out["fsds"],
                             _restart_cam_out["fsus"], _restart_cam_out["flds"],
                             _restart_cam_out["flus"], _restart_cam_out["prect"])
                write_flag(done_flag)
                forcing_ix   = cesm_to_forcing_ix(ymd, tod)
                _prefetch_ix = _next_forcing_ix(forcing_ix)
                _prefetch_future = _forcing_executor.submit(_load_forcing_slice, _prefetch_ix)
                torch.save({"state": state.cpu(), "timestep": timestep,
                            "last_ymd": ymd, "last_tod": tod,
                            "cam_out": _restart_cam_out}, atm_restart_file)
                _prev_ymd = ymd
                _prev_tod = tod
                continue
            else:
                logger.warning("  CONTINUE_RUN re-send but no saved cam_out — re-running inference")
                _expected_first_ymd = ymd
                _expected_first_tod = tod

        # --- repeated-date detection ---
        _date_repeated = (ymd == _prev_ymd and tod == _prev_tod)
        _prev_ymd = ymd
        _prev_tod = tod

        if _date_repeated and _cached_cam_out is not None:
            _u10, _v10, _tbot, _zbot, _tref, _qbot, _pbot, _fsds, _fsus, _flds, _flus, _prect = _cached_cam_out
            logger.info(f"  Repeated date — reusing last output, skipping inference")
            write_cam_nc(cam_file, _u10, _v10, _tbot, _zbot, _tref, _qbot, _pbot, _fsds, _fsus, _flds, _flus, _prect)
            write_flag(done_flag)
            _prefetch_ix     = _next_forcing_ix(forcing_ix)
            _prefetch_future = _forcing_executor.submit(_load_forcing_slice, _prefetch_ix)
            torch.save({"state": state.cpu(), "timestep": timestep,
                        "last_ymd": ymd, "last_tod": tod,
                        "cam_out": {"u10": _u10, "v10": _v10, "tbot": _tbot, "zbot": _zbot,
                                    "tref": _tref, "qbot": _qbot, "pbot": _pbot, "fsds": _fsds,
                                    "fsus": _fsus, "flds": _flds, "flus": _flus,
                                    "prect": _prect}}, atm_restart_file)
            continue

        # --- restart date guard ---
        if timestep == timestep_init and _expected_first_ymd > 0:
            if ymd != _expected_first_ymd or tod != _expected_first_tod:
                logger.error("RESTART DATE MISMATCH — atmosphere and CESM are out of sync!")
                logger.error(f"  Expected: ymd={_expected_first_ymd} tod={_expected_first_tod}s")
                logger.error(f"  CESM sent: ymd={ymd} tod={tod}s")
                sys.exit(1)
            logger.info(f"  Restart date check PASSED")

        # --- (c) SST on f09 -> CAMulator (coordinator writes on f09, no remap needed) ---
        sst_cam = sst_flat.reshape(CAM_NLAT, CAM_NLON)
        if cam_flip:
            sst_cam = sst_cam[::-1, :]

        ocean_pts = sst_cam >= OCEAN_MIN_K
        if not ocean_pts.any():
            logger.info("  SST: POP not ready — using persisted IC SST")
            sst_cam = sst_cam_persistent.copy()
        else:
            sst_cam = np.where(ocean_pts, sst_cam, LAND_SST_FILL)
            sst_cam_persistent = sst_cam.copy()

        sst_norm = (sst_cam - sst_mean) / sst_std

        # --- (e) forcing ---
        t_e = time.time()
        forcing_ix = cesm_to_forcing_ix(ymd, tod)

        if forcing_ix == _prefetch_ix:
            dynamic_forcing_t = _prefetch_future.result().to(device)
        else:
            logger.info(f"  Forcing idx mismatch: prefetched={_prefetch_ix} need={forcing_ix} — sync load")
            dynamic_forcing_t = _load_forcing_slice(forcing_ix).to(device)
            _prefetch_future.result()

        t_forcing_ms = (time.time() - t_e) * 1000

        # --- (f) build model input ---
        t_f = time.time()
        if timestep == 0:
            model_input = state
        else:
            model_input = stepper.state_manager.build_input_with_forcing(
                state, dynamic_forcing_t, static_forcing
            )
        t_build_ms = (time.time() - t_f) * 1000

        # --- (g) inject SST ---
        sst_tensor = (
            torch.from_numpy(sst_norm.copy()).float().to(device)
            .unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        accessor_input.set_state_var(model_input, "SST", sst_tensor)

        # --- (g2) inject ICEFRAC (on f09 from coordinator, no remap needed) ---
        if ocean_pts.any():
            ifrac_cam = ifrac_flat.reshape(CAM_NLAT, CAM_NLON)
            if cam_flip:
                ifrac_cam = ifrac_cam[::-1, :]
            ifrac_norm = (ifrac_cam - icefrac_mean) / icefrac_std
            ifrac_tensor = (
                torch.from_numpy(ifrac_norm.copy()).float().to(device)
                .unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            accessor_input.set_state_var(model_input, "ICEFRAC", ifrac_tensor)

        # --- (g3) CO2 trend ---
        if _co2_trend_norm_per_yr != 0.0:
            _elapsed_yrs = (timestep - _co2_ref_step) / 1460.0
            _co2_delta   = _co2_trend_norm_per_yr * _elapsed_yrs
            co2_current  = accessor_input.get_state_var(model_input, "co2vmr_3d")
            accessor_input.set_state_var(model_input, "co2vmr_3d", co2_current + _co2_delta)

        # --- (h) inference ---
        t_inf = time.time()
        with torch.no_grad():
            prediction = stepper.model(model_input.float())
        prediction = stepper._apply_postprocessing(prediction, model_input)
        t_inf_ms = (time.time() - t_inf) * 1000
        logger.info(f"  Inference: {t_inf_ms:.0f}ms")

        # --- (i) inverse-transform to physical ---
        t_i = time.time()
        prediction_out = state_transformer.inverse_transform(prediction)
        t_itrans_ms = (time.time() - t_i) * 1000

        # --- (j) extract coupling variables ---
        t_j = time.time()
        prediction_cpu = prediction_out.cpu()
        U_cam    = accessor_output.get_state_var(prediction_cpu, "U"   )[0, -1, 0].numpy()
        V_cam    = accessor_output.get_state_var(prediction_cpu, "V"   )[0, -1, 0].numpy()
        T_bot_cam= accessor_output.get_state_var(prediction_cpu, "T"   )[0, -1, 0].numpy()
        Qtot_cam = accessor_output.get_state_var(prediction_cpu, "Qtot")[0, -1, 0].numpy()

        Tv_bot_cam = T_bot_cam * (1.0 + 0.608 * np.clip(Qtot_cam, 0.0, 0.04))
        z_bot_cam  = Z_BOT_SCALE * Tv_bot_cam

        TREFHT_cam = accessor_output.get_state_var(prediction_cpu, "TREFHT")[0, 0, 0].numpy()
        PS_cam     = accessor_output.get_state_var(prediction_cpu, "PS"    )[0, 0, 0].numpy()

        FSDS_J_cam = accessor_output.get_state_var(prediction_cpu, "FSDS_J")[0, 0, 0].numpy() / DT_SEC
        FLDS_J_cam = accessor_output.get_state_var(prediction_cpu, "FLDS_J")[0, 0, 0].numpy() / DT_SEC
        FSUS_cam   = accessor_output.get_state_var(prediction_cpu, "FSUS"  )[0, 0, 0].numpy() / DT_SEC
        FLUS_cam   = accessor_output.get_state_var(prediction_cpu, "FLUS"  )[0, 0, 0].numpy() / DT_SEC

        PRECT_cam = accessor_output.get_state_var(prediction_cpu, "PRECT")[0, 0, 0].numpy() / DT_SEC
        t_extract_ms = (time.time() - t_j) * 1000

        # ================================================================
        # (j2) SUMO: write state + wait for combined + nudge
        # ================================================================
        if _sumo_enabled:
            t_sumo = time.time()

            # Build the dict of variables to write (physical units, all levels)
            # Arrays are in CAMulator's native lat order.
            # write_sumo_state_nc expects ascending lat, so flip if needed.
            _sumo_write = {}
            for _sv in _sumo_vars:
                try:
                    _arr = accessor_output.get_state_var(prediction_cpu, _sv)[0, :, 0].numpy()
                    # _arr shape: (nlev, nlat, nlon) or (1, nlat, nlon)
                    if cam_flip:
                        _arr = _arr[:, ::-1, :]    # native N->S -> ascending for file
                    _sumo_write[_sv] = _arr
                except ValueError:
                    logger.warning(f"  SUMO: variable '{_sv}' not in output accessor — skipping")

            if _sumo_write:
                # Number of levels: use first variable
                _n_lev_write = next(iter(_sumo_write.values())).shape[0]
                _lev_indices = list(range(_n_lev_write))

                # Write CAMulator state (ascending lat)
                write_sumo_state_nc(
                    sumo_cam_state_file, _sumo_write,
                    cam_lats_asc, cam_lons, _lev_indices, ymd, tod
                )
                write_flag(sumo_cam_ready_flag)
                logger.info(f"  SUMO: wrote cam_state ({list(_sumo_write.keys())}) + cam_ready.flag")

                # Wait for coordinator to produce combined state
                _poll_sumo = 0
                while not sumo_combine_done_flag.exists():
                    time.sleep(POLL_SLEEP)
                    _poll_sumo += 1
                    if _poll_sumo > _sumo_poll_max:
                        logger.error(
                            f"  SUMO: timeout waiting for sumo_combine_done.flag "
                            f"({_sumo_timeout:.0f}s). Skipping nudging this step."
                        )
                        break
                    if _poll_sumo % POLL_LOG == 0:
                        logger.info(f"  SUMO: waiting for combined state (poll {_poll_sumo}) ...")

                if sumo_combine_done_flag.exists():
                    delete_flag(sumo_combine_done_flag)

                    # Read combined state
                    combined_arrays, comb_ymd, comb_tod = read_sumo_combined_nc(sumo_combined_file)

                    # Diagnostics: mean |ΔU|, |ΔV|
                    for _sv in _sumo_vars:
                        if _sv in combined_arrays and _sv in _sumo_write:
                            _raw   = _sumo_write[_sv]   # CAMulator's prediction (ascending lat)
                            _comb  = combined_arrays[_sv]  # combined (ascending lat)
                            _nlev  = min(_raw.shape[0], _comb.shape[0])
                            _delta = np.abs(_comb[:_nlev] - _raw[:_nlev]).mean()
                            logger.info(f"  SUMO: |Δ{_sv}| mean = {_delta:.4f}  "
                                        f"(α = {_sumo_alpha:.3f})")

                    # Apply nudging to prediction tensor
                    prediction = apply_sumo_nudging(
                        prediction, combined_arrays,
                        _sumo_vars, _sumo_alpha,
                        accessor_output, state_transformer,
                        cam_flip, device
                    )

                    # Re-inverse-transform so cam_out.nc uses nudged values
                    prediction_out = state_transformer.inverse_transform(prediction)
                    prediction_cpu = prediction_out.cpu()

                    # Re-extract bottom-level vars (for cam_out.nc) after nudging
                    U_cam    = accessor_output.get_state_var(prediction_cpu, "U"   )[0, -1, 0].numpy()
                    V_cam    = accessor_output.get_state_var(prediction_cpu, "V"   )[0, -1, 0].numpy()
                    T_bot_cam= accessor_output.get_state_var(prediction_cpu, "T"   )[0, -1, 0].numpy()
                    Qtot_cam = accessor_output.get_state_var(prediction_cpu, "Qtot")[0, -1, 0].numpy()
                    Tv_bot_cam = T_bot_cam * (1.0 + 0.608 * np.clip(Qtot_cam, 0.0, 0.04))
                    z_bot_cam  = Z_BOT_SCALE * Tv_bot_cam

            t_sumo_ms = (time.time() - t_sumo) * 1000
            logger.info(f"  SUMO exchange: {t_sumo_ms:.0f}ms")

        # --- (k) remap all fields CAMulator -> T62 ---
        _cam_stack = np.stack([
            U_cam, V_cam, T_bot_cam, z_bot_cam, TREFHT_cam,
            Qtot_cam, PS_cam, FSDS_J_cam, FSUS_cam, FLDS_J_cam, FLUS_cam, PRECT_cam,
        ])
        if cam_flip:
            _cam_stack = _cam_stack[:, ::-1, :]

        t_remap_start = time.time()
        _t62 = cam_to_t62_remap.batch(_cam_stack)
        t_remap_ms = (time.time() - t_remap_start) * 1000

        u10, v10, tbot, zbot, tref, qbot, pbot, fsds, fsus, flds, flus, prect = (
            _t62[i].astype(np.float64) for i in range(12)
        )

        logger.info(f"  FSDS_J mean={FSDS_J_cam.mean():7.1f} W/m2  "
                    f"|U_bot| mean={np.sqrt(U_cam**2+V_cam**2).mean():5.2f} m/s  "
                    f"remap={t_remap_ms:.1f}ms")

        # --- (l) safety clamps ---
        qbot  = np.maximum(qbot,  1.0e-9)
        zbot  = np.clip(zbot, 20.0, 200.0)
        fsds  = np.maximum(fsds,  0.0)
        fsus  = np.maximum(fsus,  0.0)
        flds  = np.maximum(flds,  0.0)
        flus  = np.maximum(flus,  0.0)
        prect = np.maximum(prect, 0.0)

        # --- (m) write cam_out.nc ---
        t_m = time.time()
        write_cam_nc(cam_file, u10, v10, tbot, zbot, tref, qbot, pbot, fsds, fsus, flds, flus, prect)
        t_write_ms = (time.time() - t_m) * 1000

        # --- (n) signal CESM + prefetch + async NC save ---
        write_flag(done_flag)
        _prefetch_ix     = _next_forcing_ix(forcing_ix)
        _prefetch_future = _forcing_executor.submit(_load_forcing_slice, _prefetch_ix)

        if _save_nc:
            _real_year = model_start_dt.year + (ymd // 10000) - 1
            _real_dt   = datetime(_real_year, (ymd % 10000) // 100, ymd % 100, tod // 3600)
            upper_air, single_level = make_xarray(
                prediction_cpu, _real_dt, _cam_lat, _cam_lon, conf
            )
            if args.save_vars is not None:
                _keep_ua = [v for v in args.save_vars if v in upper_air.coords["vars"].values]
                _keep_sl = [v for v in args.save_vars if v in single_level.coords["vars"].values]
                if _keep_ua:
                    upper_air = upper_air.sel(vars=_keep_ua)
                if _keep_sl:
                    single_level = single_level.sel(vars=_keep_sl)
            _cesm_yr  = ymd // 10000
            _year_dir = _save_nc_dir / f"{_cesm_yr:04d}"
            _date_str = (f"{_cesm_yr:04d}-{(ymd % 10000) // 100:02d}"
                         f"-{ymd % 100:02d}-{tod:05d}")
            if not _do_daily:
                pool.apply_async(
                    save_camulator_step_nc,
                    (upper_air, single_level,
                     str(_year_dir / f"camulator.h1.{_date_str}.nc")),
                )
            if _do_daily:
                if _daily_buffer_ymd > 0 and ymd != _daily_buffer_ymd:
                    _prev_d_str = (f"{_daily_buffer_ymd // 10000:04d}-"
                                   f"{(_daily_buffer_ymd % 10000) // 100:02d}-"
                                   f"{_daily_buffer_ymd % 100:02d}")
                    _prev_yr = _save_nc_dir / f"{_daily_buffer_ymd // 10000:04d}"
                    pool.apply_async(
                        save_camulator_daily_nc,
                        (_daily_buffer,
                         str(_prev_yr / f"camulator.h1d.{_prev_d_str}.nc")),
                    )
                    _daily_buffer = []
                _daily_buffer.append((upper_air, single_level))
                _daily_buffer_ymd = ymd

        t_total_s = time.time() - t_step_start
        logger.info(f"  Wrote done.flag   "
                    f"forc={t_forcing_ms:.0f}ms  inf={t_inf_ms:.0f}ms  "
                    f"itrans={t_itrans_ms:.0f}ms  remap={t_remap_ms:.1f}ms  "
                    f"write={t_write_ms:.0f}ms  total={t_total_s:.2f}s")

        # --- (o) advance state ---
        _cached_cam_out = (u10, v10, tbot, zbot, tref, qbot, pbot, fsds, fsus, flds, flus, prect)
        state = stepper.state_manager.shift_state_forward(state, prediction)
        timestep += 1

        # --- (p) save atmosphere restart ---
        _current_year = ymd // 10000
        if _prev_step_year > 0 and _current_year != _prev_step_year:
            _archive_name = f"camulator_atm_restart.year{_prev_step_year:04d}.pth"
            _archive_path = _atm_restart_archive_dir / _archive_name
            if atm_restart_file.exists():
                shutil.copy2(atm_restart_file, _archive_path)
                logger.info(f"  Year {_prev_step_year}->{_current_year}: archived -> {_archive_name}")
        _prev_step_year = _current_year

        torch.save({
            "state":    state.cpu(),
            "timestep": timestep,
            "last_ymd": ymd,
            "last_tod": tod,
            "cam_out":  {
                "u10": u10, "v10": v10, "tbot": tbot, "zbot": zbot,
                "tref": tref, "qbot": qbot, "pbot": pbot, "fsds": fsds,
                "fsus": fsus, "flds": flds, "flus": flus, "prect": prect,
            },
        }, atm_restart_file)


if __name__ == "__main__":
    main()
