#!/bin/bash
set -e

CONFIG=/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml
CKPT_DIR=/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2
START_ITER=30

cd /glade/work/wchapman/Roman_Coupling/train_johns/climate

SAVE_BASE=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['predict']['save_forecast'])")

for ckpt in $(ls ${CKPT_DIR}/checkpoint.pt000*.pt | sort -V); do
    name=$(basename "${ckpt}" .pt)
    num=${name#checkpoint.pt}

    # Skip checkpoints before iteration 35
    if [ $((10#${num})) -lt "${START_ITER}" ]; then
        echo "=== Skipping ${name} (before iteration ${START_ITER}) ==="
        continue
    fi

    save_append="run_chkpt${num}_newforc"
    out_dir="${SAVE_BASE}/${save_append}"

    n_files=$(ls "${out_dir}"/*/*.nc 2>/dev/null | wc -l)
    if [ "${n_files}" -gt 0 ]; then
        echo "=== Skipping ${name} (${n_files} files already in ${out_dir}) ==="
        continue
    fi

    echo "=== Running ${name} -> ${save_append} ==="
    python Quick_Climate.py \
        --config "${CONFIG}" \
        --model_name "${name}.pt" \
        --save_append "${save_append}" \
        --monthly_mean
done

echo "=== All checkpoints done ==="