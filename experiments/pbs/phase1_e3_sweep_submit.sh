#!/bin/bash
# =====================================================================================
# PHASE 1 · E3 — controlled v1-vs-v2 / weight sweep, staged-screen submitter
# -------------------------------------------------------------------------------------
# Short fine-tunes from a common pre-fixer checkpoint. The failure/fix appear within ~1
# epoch, so each cell runs only SWEEP_EPOCHS (default 3). Default grid is the coarse
# screen chosen for the paper: {v1, v2} x weight {0.0, 0.1} = 4 runs. Expand WEIGHTS to
# {0.0 0.05 0.1 0.5} once the screen warrants it.
#
# For each cell this script: generates a config (sets conservation_loss_weight, save_loc,
# reload_epoch=False so the epoch counter starts at 0, num_epoch=SWEEP_EPOCHS), seeds the
# common-start checkpoint into the run dir (COPY, not symlink, so training does not clobber
# the shared base), and qsubs phase1_e3_train.pbs (4 GPU, gpu_type in select).
#
# Usage:  bash experiments/pbs/phase1_e3_sweep_submit.sh            # seed + generate + submit
#         DRYRUN=1 bash experiments/pbs/phase1_e3_sweep_submit.sh   # generate configs only
# =====================================================================================
set -euo pipefail

REPO=/glade/work/wchapman/Roman_Coupling/train_johns
SCRATCH=/glade/derecho/scratch/wchapman
GEN="$REPO/experiments/configs/generated"
PBS="$REPO/experiments/pbs/phase1_e3_train.pbs"
mkdir -p "$GEN"

# Pre-fixer common start (17-channel surgery output, before any water fixer was active).
# See the comment block in this file's git history for why extended_v2/extended are wrong.
BASE_CKPT="$SCRATCH/CREDIT_runs/wxformermod_sharp_SpatPS_pxshf/checkpoint.pt00092_extended.pt"

SWEEP_EPOCHS=${SWEEP_EPOCHS:-4}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-150}   # halve the per-epoch cost vs the 300 default
WALLTIME=${WALLTIME:-08:00:00}                # ~41 s/step x 600 steps ~= 6.8 h + margin
# total gradient steps per cell = SWEEP_EPOCHS * BATCHES_PER_EPOCH (default 600), enough to
# show v1 recovery-and-hold and v2 oscillation onset; raise if chasing v2's NaN tail.

# Both trainers use the SAME base config (128-dim model matching cp00092_extended); the v2
# cells only flip trainer type era5 -> era5-v2. This keeps the architecture, data, rollout,
# and common-start checkpoint identical so the ONLY variables are loss placement and weight.
# (The separate alt_v2_penalty.yml is a different, smaller architecture incompatible with the
# common checkpoint, so it is NOT used for the controlled sweep.)
BASE_V1="$REPO/experiments/configs/recommended_v1_decouple.train.yml"
WEIGHTS=(${WEIGHTS:-0.0 0.1})

[ -f "$BASE_CKPT" ] || { echo "ERROR: common-start checkpoint missing: $BASE_CKPT" >&2; exit 1; }

for trainer in ${TRAINERS:-v1 v2}; do
    for w in "${WEIGHTS[@]}"; do
        name="e3_${trainer}_w${w}"
        out="$GEN/${name}.yml"
        save_loc="$SCRATCH/CREDIT_runs/${name}/"

        # v2 cells: flip only the trainer type (the line whose value is exactly 'era5')
        trainer_sed=""
        [ "$trainer" = "v2" ] && trainer_sed='s|\(type:[[:space:]]*\)era5[[:space:]]*$|\1era5-v2|'

        sed -e "s|^save_loc:.*|save_loc: '${save_loc}'|" \
            -e "s|^\([[:space:]]*\)conservation_loss_weight:.*|\1conservation_loss_weight: ${w}|" \
            -e "s|^\([[:space:]]*\)reload_epoch:.*|\1reload_epoch: False|" \
            -e "s|^\([[:space:]]*\)num_epoch:.*|\1num_epoch: ${SWEEP_EPOCHS}|" \
            -e "s|^\([[:space:]]*\)batches_per_epoch:.*|\1batches_per_epoch: ${BATCHES_PER_EPOCH}|" \
            ${trainer_sed:+-e "$trainer_sed"} \
            "$BASE_V1" > "$out"
        echo "generated $out  (weight=$w, ${SWEEP_EPOCHS} epochs, save_loc=$save_loc)"

        if [ "${DRYRUN:-0}" = "1" ]; then continue; fi

        mkdir -p "$save_loc"
        [ -f "$save_loc/checkpoint.pt" ] || cp "$BASE_CKPT" "$save_loc/checkpoint.pt"
        qsub -N "$name" -l "walltime=${WALLTIME}" -v CONFIG="$out" "$PBS"
    done
done

echo "Done. ${SWEEP_EPOCHS}-epoch cells: $(for t in v1 v2; do for w in "${WEIGHTS[@]}"; do echo -n "$t/w$w "; done; done)"
echo "Set DRYRUN=1 to generate without seeding/submitting; WEIGHTS='0.0 0.05 0.1 0.5' to expand."
