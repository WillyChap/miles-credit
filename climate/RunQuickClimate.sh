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
# CAMulator climate inference  —  driver
# ============================================================================
# Roll a trained CAMulator checkpoint forward and write NetCDF, driven by ONE
# yaml file (camulator_config.yml). Output granularity is set by AVG:
#
#   AVG=none  (default)  one NetCDF per 6-hourly step   -> pred_*.nc
#   AVG=daily            one daily-mean NetCDF per day   (--daily_mean)
#   AVG=monthly          one monthly-mean NetCDF per month (--monthly_mean)
#
# Files land in:  <save_forecast>/<FOLD_OUT>/<init_time>/pred_*.nc
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
# Each of these can be overridden from the environment, e.g.
#   CONDA_ENV=camulator AVG=monthly bash RunQuickClimate.sh
CONFIG=${CONFIG:-./camulator_config.yml}
# Conda env to activate. Default is the name created by environment.yml.
# NCAR users can point this at /glade/work/wchapman/conda-envs/credit-coupling-ud.
# Set CONDA_ENV="" to skip activation (e.g. you already activated it).
CONDA_ENV=${CONDA_ENV:-camulator}
FOLD_OUT=${FOLD_OUT:-run_default}     # experiment subfolder under save_forecast
MODEL_NAME=${MODEL_NAME:-checkpoint.pt}  # checkpoint file inside save_loc (./assets)
AVG=${AVG:-none}                      # none | daily | monthly
# ----------------------------------------------------------------------------

case "$AVG" in
  none)    AVG_FLAG="" ;;
  daily)   AVG_FLAG="--daily_mean" ;;
  monthly) AVG_FLAG="--monthly_mean" ;;
  *) echo "AVG must be one of: none, daily, monthly (got '$AVG')"; exit 2 ;;
esac

# Activate a conda env if requested and available (skipped on non-conda systems
# or when CONDA_ENV is empty). The #PBS lines above are ignored unless this file
# is submitted with qsub, so `bash RunQuickClimate.sh` works anywhere.
if [ -n "$CONDA_ENV" ] && command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then module load conda 2>/dev/null || true; fi
  # shellcheck disable=SC1091
  conda activate "$CONDA_ENV" || echo "WARN: could not activate '$CONDA_ENV'; using current env"
fi

echo "============================================================"
echo " CAMulator inference"
echo "   config         : $CONFIG"
echo "   checkpoint     : $MODEL_NAME"
echo "   experiment     : $FOLD_OUT"
echo "   output mode    : $AVG"
echo "   start          : $(date)"
echo "============================================================"

python ./Quick_Climate.py \
  --config "$CONFIG" \
  --model_name "$MODEL_NAME" \
  --device cuda \
  --save_append "$FOLD_OUT" \
  $AVG_FLAG

echo "============================================================"
echo " Done at $(date).  Output under <save_forecast>/$FOLD_OUT/"
echo "============================================================"
