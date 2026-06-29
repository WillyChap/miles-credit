#!/bin/bash
# =====================================================================================
# PHASE 1 · E3 — controlled v1-vs-v2 + weight sweep, submitter
# -------------------------------------------------------------------------------------
# Generates one config per (trainer, weight) from the two preserved base configs by
# editing only conservation_loss_weight and save_loc, then qsubs phase1_e3_train.pbs
# for each. The single sweep knob:
#   v1 base (trainer era5):     weight 0.0 = BROKEN control, >0 = the fix
#   v2 base (trainer era5-v2):  weight 0.0 .. 0.5 = the untested penalty-only path
#
# PREREQUISITE (clean common start): each generated run must load the SAME baseline
# checkpoint (baseline_no_fixers, fixers off) so the only difference is the sweep knob.
# Seed each run dir with that checkpoint, or set the load path in the base config, before
# launching. This script does NOT copy checkpoints for you — see the TODO below.
#
# Usage:  bash experiments/pbs/phase1_e3_sweep_submit.sh            # generate + submit
#         DRYRUN=1 bash experiments/pbs/phase1_e3_sweep_submit.sh   # generate only
# =====================================================================================
set -euo pipefail

REPO=/glade/work/wchapman/Roman_Coupling/train_johns
SCRATCH=/glade/derecho/scratch/wchapman
GEN="$REPO/experiments/configs/generated"
PBS="$REPO/experiments/pbs/phase1_e3_train.pbs"
mkdir -p "$GEN"

# Common start = the PRE-FIXER, well-trained base: the surgery-extended 17-channel model
# (cp00092_extended) before any water fixer was ever active. This is the only neutral start
# for a clean sweep; the w=0.0 broken control must begin from a model that has NOT yet been
# shaped by a conservation penalty.
#   DO NOT use NEW_CLI_JOHN_CASPER_extended_v2/checkpoint.pt00079.pt -- that is the already-
#   FIXED endpoint (64 epochs of penalty training, drift ~0), which would make the sweep
#   circular. DO NOT use NEW_CLI_JOHN_CASPER_extended/ either -- that run is itself the
#   feedback configuration (water fixer on, conservation_loss_weight: 0.0; drifted to ~45%).
BASE_CKPT="$SCRATCH/CREDIT_runs/wxformermod_sharp_SpatPS_pxshf/checkpoint.pt00092_extended.pt"

declare -A BASECFG=(
    [v1]="$REPO/experiments/configs/recommended_v1_decouple.train.yml"
    [v2]="$REPO/experiments/configs/alt_v2_penalty.train.yml"
)
WEIGHTS=(0.0 0.05 0.1 0.5)

for trainer in v1 v2; do
    for w in "${WEIGHTS[@]}"; do
        name="e3_${trainer}_w${w}"
        out="$GEN/${name}.yml"
        save_loc="$SCRATCH/CREDIT_runs/${name}/"

        # edit only the two lines; preserve the rest of the config verbatim
        sed -e "s|conservation_loss_weight:.*|conservation_loss_weight: ${w}|" \
            -e "s|^save_loc:.*|save_loc: '${save_loc}'|" \
            "${BASECFG[$trainer]}" > "$out"

        echo "generated $out  (save_loc=$save_loc)"
        # TODO before launch: mkdir -p "$save_loc" && cp "$BASE_CKPT" "$save_loc/checkpoint.pt"
        #   and confirm the base config's load_weights/reload_epoch resume from it.

        if [ "${DRYRUN:-0}" != "1" ]; then
            qsub -v CONFIG="$out" -N "$name" "$PBS"
        fi
    done
done

echo "Done. 8 configs in $GEN. Set DRYRUN=1 to generate without submitting."
