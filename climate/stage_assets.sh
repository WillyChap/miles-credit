#!/bin/bash
# ============================================================================
# stage_assets.sh
# ----------------------------------------------------------------------------
# Populate ./assets/ with every file CAMulator inference needs.
#
# The config (camulator_config.yml) refers to all runtime inputs by the
# relative path ./assets/<file>, so the toolbox is self-contained: whatever
# lands in ./assets/ is what the model uses.
#
# Two ways to fill ./assets/:
#
#   1. ON NCAR (Derecho/Casper): run this script. It SYMLINKS the canonical
#      GLADE copies into ./assets/ (no data is duplicated).
#
#   2. ANYWHERE ELSE (e.g. a HuggingFace download): skip this script and copy
#      the files listed in README.md ("Asset manifest") into ./assets/.
#
# Usage:
#   ./stage_assets.sh            # symlink the default checkpoint + assets
#   ./stage_assets.sh --copy     # hard-copy instead of symlink (portable)
# ============================================================================
set -euo pipefail

MODE="symlink"
[ "${1:-}" = "--copy" ] && MODE="copy"

ASSETS="$(cd "$(dirname "$0")" && pwd)/assets"
mkdir -p "$ASSETS"

# --- canonical GLADE sources -> assets/ basename ----------------------------
# checkpoint (the trained model weights)
CKPT_DIR="/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2"

declare -a SRC=(
  # normalization
  "/glade/derecho/scratch/wchapman/b_credit_runs/mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc"
  "/glade/derecho/scratch/wchapman/b_credit_runs/std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc"
  # statics (data block + post-block conservation fixers)
  "/glade/derecho/scratch/wchapman/b_credit_runs/statics_b_credit_runs_f32_02.nc"
  "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
  # latitude weights (loss block)
  "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/f.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
  # cyclic 1-yr forcing (SOLIN, SST, ICEFRAC, co2vmr_3d)
  "/glade/derecho/scratch/wchapman/CAMULATOR_FORCING/b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc"
  # initial condition tensor (default start 1981-01-01T00Z)
  "/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended/init_times/init_camulator_condition_tensor_1981-01-01T00Z.pth"
  # output metadata
  "/glade/work/schreck/repos/credit/miles-credit/metadata/era5.yaml"
  # OPTIONAL: climatology + seasonal mean (only for rollout_metrics / fast-climate scores)
  "/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/ERA5_clim_1990_2019_6h_interp.nc"
  "/glade/work/wchapman/miles_branchs/CESM_Spatial_PS/truth_be21_tensor_2013-01-01T00Z.pth"
)

link_one () {
  local src="$1" dst="$ASSETS/$(basename "$1")"
  if [ ! -e "$src" ]; then
    echo "  !! MISSING source: $src"; return
  fi
  if [ "$MODE" = "copy" ]; then
    cp -n "$src" "$dst" && echo "  copied  $(basename "$src")"
  else
    ln -sfn "$src" "$dst" && echo "  linked  $(basename "$src")"
  fi
}

echo "Staging assets into $ASSETS (mode: $MODE)"
for s in "${SRC[@]}"; do link_one "$s"; done

# checkpoint.pt -> assets/checkpoint.pt  (config save_loc points at ./assets/)
if [ -e "$CKPT_DIR/checkpoint.pt" ]; then
  if [ "$MODE" = "copy" ]; then cp -n "$CKPT_DIR/checkpoint.pt" "$ASSETS/checkpoint.pt"
  else ln -sfn "$CKPT_DIR/checkpoint.pt" "$ASSETS/checkpoint.pt"; fi
  echo "  ${MODE/symlink/linked}  checkpoint.pt"
else
  echo "  !! MISSING checkpoint: $CKPT_DIR/checkpoint.pt"
fi

echo "Done. Verify with:  ls -lh assets/"
