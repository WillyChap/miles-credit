#!/bin/bash
#PBS -N rechunk_zarr
#PBS -A NAML0001
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=06:00:00
#PBS -q casper
#PBS -J 0-34
#PBS -j oe
#PBS -o /glade/work/wchapman/Roman_Coupling/train_johns/rechunk_logs/

module load conda/latest
conda activate /glade/work/wchapman/conda-envs/credit-casper-mar2026

# Array index → year (1980 + index)
YEAR=$(( 1980 + PBS_ARRAY_INDEX ))

echo "Processing year: ${YEAR}"

python /glade/work/wchapman/Roman_Coupling/train_johns/rechunk_zarr.py --year ${YEAR}
