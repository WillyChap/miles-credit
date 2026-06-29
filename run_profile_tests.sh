#!/bin/bash
# run_profile_tests.sh
# Run on an interactive Casper node with 1 GPU (env already activated).

set -e
cd /glade/work/wchapman/Roman_Coupling/train_johns
mkdir -p profile_results

echo "========================================================"
echo " Camulator profiling — $(date)"
echo "========================================================"

python profile_camulator.py --batch_size 4 2>&1 | tee profile_results/summary.log

echo ""
echo "Done. Results in profile_results/"
