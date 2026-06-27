#!/bin/bash
# ============================================================================
# run_upload.sh  (maintainer tool)
# ----------------------------------------------------------------------------
# Walk-away upload of the CAMulator assets to HuggingFace via nohup.
#
# Run this on an NCAR LOGIN or DATA-ACCESS node (e.g. casper-login, derecho-login,
# or a `qinteractive`-free shell) -- PBS COMPUTE nodes have NO outbound internet
# and the upload will hang/fail there.
#
# It stages the shared assets (symlinks), then uploads:
#   model card -> README.md, config -> inference_config.yaml, figs/,
#   normalization, forcing (cyclic + slim progressive), all 69 initial
#   conditions, and the top-10 checkpoints (65,63,70,48,47,66,76,68,51,43)
#   from the run dir. ~75 GB total. Safe to re-run: HF LFS de-dups uploaded content.
#
# Usage (detach and walk away), from climate/maintainer/ :
#   huggingface-cli login            # once, paste willychap write token
#   cd climate/maintainer
#   nohup ./run_upload.sh > upload.out 2>&1 &
#   disown; tail -f upload.out       # watch; Ctrl-C the tail anytime
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

REPO_ID="${REPO_ID:-willychap/camulator}"
CONDA_ENV="${CONDA_ENV:-/glade/work/wchapman/conda-envs/credit-coupling-ud}"
CKPT_DIR="${CKPT_DIR:-/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2}"
CKPTS="${CKPTS:-checkpoint.pt00065.pt checkpoint.pt00063.pt checkpoint.pt00070.pt checkpoint.pt00048.pt checkpoint.pt00047.pt checkpoint.pt00066.pt checkpoint.pt00076.pt checkpoint.pt00068.pt checkpoint.pt00051.pt checkpoint.pt00043.pt}"
INIT_DIR="${INIT_DIR:-/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/init_times}"
LOG="${LOG:-$HERE/upload_$(date +%Y%m%d_%H%M%S).log}"

# activate env (works whether or not conda is already initialised in this shell)
CONDA_ROOT="$(dirname "$CONDA_ENV")"
CONDA_ROOT="$(dirname "$CONDA_ROOT")"
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV" 2>/dev/null || source activate "$CONDA_ENV"

{
  echo "=== CAMulator HF upload  $(date) ==="
  echo "repo:   $REPO_ID"
  echo "ckpts:  $CKPTS"
  echo "log:    $LOG"
  echo "host:   $(hostname)"

  # fail fast if not logged in
  if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "ERROR: not logged in. Run 'huggingface-cli login' first." >&2
    exit 1
  fi
  echo "user:   $(huggingface-cli whoami)"

  echo "--- staging shared assets (symlinks) ---"
  ./stage_assets.sh

  echo "--- uploading (this is the long part) ---"
  # shellcheck disable=SC2086
  python upload_assets.py \
    --repo_id "$REPO_ID" --create \
    --checkpoints $CKPTS \
    --checkpoint_dir "$CKPT_DIR" \
    --init_dir "$INIT_DIR"

  echo "=== DONE  $(date) ==="
} 2>&1 | tee -a "$LOG"
