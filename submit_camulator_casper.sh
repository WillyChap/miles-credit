#!/bin/bash
#PBS -A NAML0001
#PBS -N camulator_extended
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=32:ngpus=4:mem=250GB
#PBS -q casper
#PBS -j oe
#PBS -k eod

source ~/.bashrc
conda activate /glade/work/wchapman/conda-envs/credit-casper-mar2026

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /glade/work/wchapman/Roman_Coupling/train_johns

torchrun --nproc_per_node=4 \
  credit/applications/train.py \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended.yml
