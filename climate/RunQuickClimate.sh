#!/bin/bash
#-*- coding: utf-8 -*-
#PBS -N CAMulator_Climate
#PBS -A NAML0001
#PBS -l walltime=12:00:00
#PBS -o RUN_Climate.out
#PBS -e RUN_Climate.out
#PBS -q casper
#PBS -l select=1:ncpus=32:ngpus=1:mem=250GB
#PBS -l gpu_type=a100
#PBS -m a
#PBS -M wchapman@ucar.edu

# ============================================================================
# CAMulator climate inference  —  end-to-end driver
# ============================================================================
# Two output modes, both driven by ONE yaml file (camulator_config.yml):
#
#   AVG=none  (default)  full 6-hourly resolution -> YEARLY zarr
#       1. Quick_Climate.py   roll the model forward -> per-step pred_*.nc
#       2. netcdf_to_zarr.py  consolidate pred_*.nc  -> <prefix>_<year>.zarr
#
#   AVG=daily | monthly     compact time-averaged NetCDF (no zarr step)
#       1. Quick_Climate.py --{daily,monthly}_mean -> one averaged pred_*.nc
#          per day/month  (this replaces the old Post_Process.py)
#
# All model inputs are read from ./assets/ (see stage_assets.sh / README.md).
#
# Run it either as a PBS job  (qsub RunQuickClimate.sh)  or directly on an
# interactive GPU node        (bash RunQuickClimate.sh).
#
# BEFORE RUNNING:
#   - conda activate your CREDIT env (set CONDA_ENV below)
#   - ./stage_assets.sh           (NCAR) or download assets per README.md
#   - edit FOLD_OUT / MODEL_NAME / AVG below as desired
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------- user settings --------------------------------
CONFIG=./camulator_config.yml
CONDA_ENV=/glade/work/wchapman/conda-envs/credit-coupling   # <-- your env
FOLD_OUT=run_default                 # experiment subfolder under save_forecast
MODEL_NAME=checkpoint.pt             # checkpoint file inside save_loc (./assets)
ZARR_PREFIX=camulator                # output zarr name: <prefix>_<year>.zarr
AVG=none                             # none -> yearly zarr | daily | monthly
# ----------------------------------------------------------------------------

case "$AVG" in
  none)    AVG_FLAG="" ;;
  daily)   AVG_FLAG="--daily_mean" ;;
  monthly) AVG_FLAG="--monthly_mean" ;;
  *) echo "AVG must be one of: none, daily, monthly (got '$AVG')"; exit 2 ;;
esac

if command -v module >/dev/null 2>&1; then module load conda || true; fi
# shellcheck disable=SC1091
conda activate "$CONDA_ENV"

# Derive save_forecast dir + init-time string straight from the yaml so the
# downstream zarr step knows exactly where Quick_Climate wrote its files.
read -r SAVE_FORECAST INIT_STR < <(python3 - "$CONFIG" <<'PY'
import sys, yaml, datetime
c = yaml.safe_load(open(sys.argv[1]))
sf = c["predict"]["save_forecast"]
dt = c["predict"]["start_datetime"]
if isinstance(dt, str):
    dt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
print(sf, dt.strftime("%Y-%m-%dT%HZ"))
PY
)
PRED_DIR="${SAVE_FORECAST%/}/${FOLD_OUT}/${INIT_STR}"
ZARR_DIR="${SAVE_FORECAST%/}/${FOLD_OUT}/zarr"

echo "============================================================"
echo " CAMulator inference"
echo "   config         : $CONFIG"
echo "   checkpoint     : $MODEL_NAME"
echo "   experiment     : $FOLD_OUT"
echo "   output mode    : $AVG"
echo "   pred dir       : $PRED_DIR"
[ "$AVG" = "none" ] && echo "   zarr out       : $ZARR_DIR"
echo "   start          : $(date)"
echo "============================================================"

# --------------------------- 1. roll the model ------------------------------
echo "[1/2] Quick_Climate.py $AVG_FLAG ..."
python ./Quick_Climate.py \
  --config "$CONFIG" \
  --model_name "$MODEL_NAME" \
  --device cuda \
  --save_append "$FOLD_OUT" \
  $AVG_FLAG

# --------------------------- 2. consolidate ---------------------------------
if [ "$AVG" = "none" ]; then
  echo "[2/2] netcdf_to_zarr.py  (-> yearly zarr) ..."
  python ./netcdf_to_zarr.py \
    --input_dir "$PRED_DIR" \
    --out_dir   "$ZARR_DIR" \
    --prefix    "$ZARR_PREFIX" \
    --statics   ./assets/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc
  echo "    -> $(ls -d "$ZARR_DIR"/*.zarr 2>/dev/null | wc -l) yearly zarr store(s) in $ZARR_DIR"
else
  echo "[2/2] zarr step skipped — $AVG-mean NetCDF written to $PRED_DIR"
fi

echo "============================================================"
echo " Done at $(date)."
[ "$AVG" = "none" ] && echo " Yearly zarr: $ZARR_DIR" || echo " $AVG means: $PRED_DIR"
echo "============================================================"
