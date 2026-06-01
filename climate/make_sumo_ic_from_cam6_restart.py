"""
make_sumo_ic_from_cam6_restart.py
----------------------------------
Create a CAMulator initial-condition tensor whose state matches a CAM6 restart.
This is critical for the SUMO (supermodel) setup: both models must start from
the same atmospheric state so nudging increments start near zero.

Variable sources:
  cam.r  : U, V, Q+CLDLIQ+CLDICE→Qtot, PS
  cam.i  : T  (not in cam.r; auto-detected from cam.r path)
  cam.h1 : TS→SST, TREFHT  (on native CAM grid; auto-detected from cam.r path)
  base IC: everything else (SOLIN, co2vmr_3d, z_norm, LANDM_COSLAT, ICEFRAC)
           ICEFRAC is provided by CICE at each coupling step — base IC is fine.

Qtot = Q + CLDLIQ + CLDICE  (all in cam.r)

Usage:
  conda activate /glade/work/wchapman/conda-envs/credit-coupling-ud
  python make_sumo_ic_from_cam6_restart.py \\
      --config      ./camulator_config.yml \\
      --model_name  checkpoint.pt00044.pt \\
      --base_ic     /path/to/init_camulator_condition_tensor_1980-01-01T00Z.pth \\
      --cam6_restart /path/to/<case>.cam.r.1980-01-01-00000.nc \\
      --output      /path/to/sumo_init_camulator_condition_tensor_1980-01-01T00Z.pth \\
      --replace_vars U V T Q PS \\
      --device      cpu
"""

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import netCDF4 as nc

from Model_State import initialize_camulator, StateVariableAccessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Grid remap
# =============================================================================

class BilinearRemap:
    def __init__(self, src_lats, src_lons, dst_lats, dst_lons):
        nlat_src = len(src_lats)
        nlon_src = len(src_lons)

        lat_g, lon_g = np.meshgrid(dst_lats, dst_lons, indexing="ij")
        flat_lats = lat_g.ravel()
        flat_lons = lon_g.ravel() % 360.0

        i0 = np.clip(np.searchsorted(src_lats, flat_lats, side="right") - 1, 0, nlat_src - 2)
        i1 = i0 + 1
        j0 = np.clip(np.searchsorted(src_lons, flat_lons, side="right") - 1, 0, nlon_src - 1)
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

    def __call__(self, field_2d):
        return (
            self.w00 * field_2d[self.i0, self.j0]
          + self.w01 * field_2d[self.i0, self.j1]
          + self.w10 * field_2d[self.i1, self.j0]
          + self.w11 * field_2d[self.i1, self.j1]
        ).reshape(self.shape_out)

    def remap_3d(self, field_3d):
        nlev = field_3d.shape[0]
        out  = np.empty((nlev,) + self.shape_out, dtype=np.float32)
        for k in range(nlev):
            out[k] = self(field_3d[k])
        return out


# =============================================================================
# CAM6 file reader
# =============================================================================

def _read_nc_var(ds, varname, lat_flip):
    """Read one variable from an open netCDF4 Dataset; handle time dim and lat flip."""
    dims = ds.variables[varname].dimensions
    arr = ds.variables[varname][:].data.astype(np.float32)
    if dims[0] == "time":
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]      # (lat,lon) → (1,lat,lon)
    if lat_flip:
        arr = arr[:, ::-1, :].copy()
    return arr


def read_cam6_file(cam6_path, vars_needed, label="cam6"):
    """
    Read variables from any CAM6 NetCDF file (cam.r, cam.i, cam.h1 ...).

    For Q: automatically adds CLDLIQ + CLDICE if present → Qtot.

    Returns
    -------
    data  : dict[str, np.ndarray]  shape (nlev,nlat,nlon) or (1,nlat,nlon)
    lats  : 1-D ascending np.ndarray
    lons  : 1-D [0,360) np.ndarray
    """
    # CAM6 name → same name in file (identity map; TS is TS in cam.h1)
    data = {}
    with nc.Dataset(str(cam6_path), "r") as ds:
        lats_raw = ds.variables["lat"][:].data.astype(np.float64)
        lons_raw = ds.variables["lon"][:].data.astype(np.float64)
        lat_flip = lats_raw[0] > lats_raw[-1]
        lats = (lats_raw[::-1] if lat_flip else lats_raw).copy()
        lons = lons_raw % 360.0

        logger.info(f"  {label}: {len(lats)} lat × {len(lons)} lon  "
                    f"lat [{lats[0]:.2f}, {lats[-1]:.2f}]  lat_flip={lat_flip}")

        for var in vars_needed:
            if var not in ds.variables:
                logger.warning(f"  '{var}' not found in {label} — skipping")
                continue

            arr = _read_nc_var(ds, var, lat_flip)

            # Q → Qtot: add cloud liquid and ice water
            if var == "Q":
                n_added = 0
                for cv in ("CLDLIQ", "CLDICE"):
                    if cv in ds.variables:
                        arr = arr + _read_nc_var(ds, cv, lat_flip)
                        n_added += 1
                if n_added == 2:
                    logger.info(f"  Qtot = Q + CLDLIQ + CLDICE  (all from {label})")
                elif n_added > 0:
                    logger.warning(f"  Only {n_added}/2 cloud components available — Qtot may be incomplete")
                else:
                    logger.warning(f"  CLDLIQ/CLDICE not in {label} — using Q alone for Qtot")

            data[var] = arr
            _sh = "×".join(str(s) for s in arr.shape)
            logger.info(f"  Read {var}: ({_sh})  "
                        f"min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")

    return data, lats, lons


def auto_sibling(cam_r_path, suffix):
    """Derive a sibling file path by replacing .cam.r. with .cam.<suffix>."""
    p = Path(str(cam_r_path))
    candidate = p.parent / p.name.replace(".cam.r.", f".cam.{suffix}.")
    return candidate if candidate.exists() else None


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Replace CAMulator IC dynamical state with CAM6 restart data (SUMO init)"
    )
    p.add_argument("--config",       required=True,
                   help="Path to camulator_config.yml")
    p.add_argument("--model_name",   required=True,
                   help="Checkpoint filename, e.g. checkpoint.pt00044.pt")
    p.add_argument("--base_ic",      required=True,
                   help="CAMulator IC tensor (.pth) from Make_Climate_Initial_Conditions.py")
    p.add_argument("--cam6_restart", required=True,
                   help="cam.r restart file (*.cam.r.YYYY-MM-DD-SSSSS.nc). "
                        "Provides U, V, Q+CLDLIQ+CLDICE→Qtot, PS.")
    p.add_argument("--cam6_ic",      default=None,
                   help="[Optional] cam.i initial-conditions file. Auto-detected alongside "
                        "cam.r if not given. Provides T.")
    p.add_argument("--cam6_h1",      default=None,
                   help="[Optional] cam.h1 history file. Auto-detected alongside cam.r if "
                        "not given. Provides TS→SST and TREFHT on the native CAM grid.")
    p.add_argument("--output",       required=True,
                   help="Output path for SUMO IC tensor (.pth)")
    p.add_argument("--replace_vars", nargs="+", default=["U", "V", "T", "Q", "PS"],
                   help="CAM6 variable names to replace (default: U V T Q PS). "
                        "Q→Qtot and TS→SST mappings are applied automatically. "
                        "TS and TREFHT are always added from cam.h1 if available.")
    p.add_argument("--device",       default="cpu")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Resolve sibling files
    cam_i_path  = Path(args.cam6_ic)  if args.cam6_ic  else auto_sibling(args.cam6_restart, "i")
    cam_h1_path = Path(args.cam6_h1)  if args.cam6_h1  else auto_sibling(args.cam6_restart, "h1")

    logger.info("=" * 65)
    logger.info("SUMO IC initialization: replacing dynamical state from CAM6")
    logger.info(f"  Base IC      : {args.base_ic}")
    logger.info(f"  cam.r        : {args.cam6_restart}")
    logger.info(f"  cam.i        : {cam_i_path  or 'not found (T will be skipped)'}")
    logger.info(f"  cam.h1       : {cam_h1_path or 'not found (TS/TREFHT will be skipped)'}")
    logger.info(f"  Output       : {args.output}")
    logger.info(f"  Replace vars : {args.replace_vars}")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 1. Initialize CAMulator
    # ------------------------------------------------------------------
    logger.info("Loading CAMulator (for normalization statistics) ...")
    ctx = initialize_camulator(args.config, model_name=args.model_name, device=args.device)
    state_transformer = ctx["state_transformer"]
    conf              = ctx["conf"]
    latlons           = ctx["latlons"]

    accessor_input = StateVariableAccessor(conf, tensor_type="input")

    cam_lats_raw = latlons.latitude.values.copy()
    cam_lons     = latlons.longitude.values.copy()
    cam_lats_asc = np.sort(cam_lats_raw)
    cam_flip     = cam_lats_raw[0] > cam_lats_raw[-1]

    logger.info(f"CAMulator grid: {len(cam_lats_asc)}×{len(cam_lons)}  "
                f"lat [{cam_lats_asc[0]:.3f}, {cam_lats_asc[-1]:.3f}]  lat_flip={cam_flip}")

    # ------------------------------------------------------------------
    # 2. Load base CAMulator IC tensor
    # ------------------------------------------------------------------
    logger.info(f"Loading base CAMulator IC: {args.base_ic}")
    state = torch.load(args.base_ic, map_location=args.device)
    logger.info(f"  Base IC shape: {list(state.shape)}  dtype: {state.dtype}")

    # ------------------------------------------------------------------
    # 3. Collect all CAM6 data
    #    cam.r  → U, V, Q→Qtot (with CLDLIQ+CLDICE), PS
    #    cam.i  → T
    #    cam.h1 → TS (→SST), TREFHT
    # ------------------------------------------------------------------
    # Build per-file request lists
    r_vars  = [v for v in args.replace_vars if v not in ("T",)]        # cam.r has everything except T
    i_vars  = [v for v in args.replace_vars if v == "T"]               # T from cam.i only
    h1_vars = ["TS", "TREFHT"]                                         # always from cam.h1

    logger.info(f"Reading cam.r: {args.cam6_restart}")
    cam6_data, cam6_lats, cam6_lons = read_cam6_file(
        args.cam6_restart, r_vars, label="cam.r")

    # T from cam.i
    if "T" in args.replace_vars:
        if cam_i_path:
            logger.info(f"Reading cam.i for T: {cam_i_path}")
            i_data, _, _ = read_cam6_file(cam_i_path, ["T"], label="cam.i")
            if "T" in i_data:
                cam6_data["T"] = i_data["T"]
            else:
                logger.warning("T not found in cam.i — will be skipped")
        else:
            logger.warning("cam.i not found — T will be skipped (pass --cam6_ic to override)")

    # TS→SST and TREFHT from cam.h1
    if cam_h1_path:
        logger.info(f"Reading cam.h1 for TS/TREFHT: {cam_h1_path}")
        h1_data, h1_lats, h1_lons = read_cam6_file(cam_h1_path, h1_vars, label="cam.h1")
        for v in h1_vars:
            if v in h1_data:
                cam6_data[v] = h1_data[v]
    else:
        logger.warning("cam.h1 not found — TS (→SST) and TREFHT will be skipped")
        h1_lats, h1_lons = cam6_lats, cam6_lons   # unused

    # ------------------------------------------------------------------
    # 4. Build remap for cam.r/cam.i grid (same grid for all atm files)
    # ------------------------------------------------------------------
    def make_remap(src_lats, src_lons, label):
        grids_match = (
            len(src_lats) == len(cam_lats_asc)
            and np.allclose(src_lats, cam_lats_asc, atol=0.01)
            and len(src_lons) == len(cam_lons)
            and np.allclose(src_lons % 360.0, cam_lons % 360.0, atol=0.01)
        )
        if grids_match:
            logger.info(f"{label} grid matches CAMulator — no remap needed")
            return None
        logger.info(f"Building bilinear remap {label} {len(src_lats)}×{len(src_lons)} "
                    f"→ CAMulator {len(cam_lats_asc)}×{len(cam_lons)}")
        return BilinearRemap(src_lats, src_lons, cam_lats_asc, cam_lons)

    remap_atm = make_remap(cam6_lats, cam6_lons, "cam.r/cam.i")
    remap_h1  = (make_remap(h1_lats,  h1_lons,  "cam.h1")
                 if cam_h1_path else None)

    # ------------------------------------------------------------------
    # 5. Replace channels in IC tensor
    #    Ordered replace list = user vars + TS + TREFHT (h1-sourced)
    # ------------------------------------------------------------------
    # CAM6 file variable → CAMulator tensor variable
    CAM6_TO_CAMULATOR = {
        "Q":      "Qtot",   # Q+CLDLIQ+CLDICE already summed in read_cam6_file
        "TS":     "SST",    # cam.h1 surface temperature → CAMulator SST channel
    }

    all_replace = list(args.replace_vars)
    for v in ["TS", "TREFHT"]:
        if v in cam6_data and v not in all_replace:
            all_replace.append(v)

    replaced = []
    skipped  = []

    for var in all_replace:
        cam_var = CAM6_TO_CAMULATOR.get(var, var)

        if var not in cam6_data:
            skipped.append(var)
            logger.warning(f"  Skipping '{var}' — not available")
            continue

        var_info = accessor_input.get_var_info(cam_var)
        if not var_info.get("available", False):
            skipped.append(var)
            logger.warning(f"  Skipping '{var}' (→'{cam_var}') — not in CAMulator input "
                           f"({var_info.get('reason', '?')})")
            continue

        cam6_arr = cam6_data[var]

        # Choose correct remap (h1 vars may be on different grid in principle)
        _remap = remap_h1 if var in ("TS", "TREFHT") else remap_atm
        cam6_arr_cam = _remap.remap_3d(cam6_arr) if _remap else cam6_arr.copy()

        if cam_flip:
            cam6_arr_cam = cam6_arr_cam[:, ::-1, :].copy()

        # Normalize
        mean_v = state_transformer.mean_tensors[cam_var]
        std_v  = state_transformer.std_tensors[cam_var]
        if isinstance(mean_v, torch.Tensor):
            mean_v = mean_v.cpu().numpy()
            std_v  = std_v.cpu().numpy()
        if hasattr(mean_v, "shape") and mean_v.ndim == 1 and mean_v.shape[0] > 1:
            mean_v = mean_v.reshape(-1, 1, 1)
            std_v  = std_v.reshape(-1, 1, 1)
        cam6_norm = (cam6_arr_cam - mean_v) / std_v

        # Level-count check
        cam_nlev = var_info["n_channels"]
        if cam6_norm.shape[0] != cam_nlev:
            if cam6_norm.shape[0] < cam_nlev:
                logger.warning(f"  '{var}': {cam6_norm.shape[0]} levels < {cam_nlev} expected — "
                               f"padding with zeros at top")
                pad = np.zeros((cam_nlev - cam6_norm.shape[0],) + cam6_norm.shape[1:], np.float32)
                cam6_norm = np.concatenate([pad, cam6_norm], axis=0)
            else:
                logger.warning(f"  '{var}': {cam6_norm.shape[0]} levels > {cam_nlev} expected — "
                               f"truncating to bottom {cam_nlev}")
                cam6_norm = cam6_norm[-cam_nlev:]

        n_time = state.shape[2]
        cam6_tensor = (
            torch.from_numpy(cam6_norm.copy()).float()
            .unsqueeze(0).unsqueeze(2)
            .expand(-1, -1, n_time, -1, -1)
        )

        accessor_input.set_state_var(state, cam_var, cam6_tensor)
        replaced.append(cam_var)

        label = f"'{var}'" if cam_var == var else f"'{var}'→'{cam_var}'"
        logger.info(f"  Replaced {label}: {cam6_norm.shape}  "
                    f"norm_mean={cam6_norm.mean():.4f}  norm_std={cam6_norm.std():.4f}")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(output_path))

    logger.info("=" * 65)
    logger.info(f"SUMO IC saved: {output_path}")
    logger.info(f"  Replaced  : {replaced}")
    if skipped:
        logger.info(f"  Skipped   : {skipped}")
    logger.info(f"  IC shape  : {list(state.shape)}")
    logger.info("=" * 65)
    logger.info("Next step: use this tensor as --init_cond in camulator_sumo_server.py")


if __name__ == "__main__":
    main()
