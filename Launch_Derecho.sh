module load ncarenv/24.12 gcc/12.4.0 ncarcompilers cray-mpich/8.1.29 cuda/12.3.2 conda/latest
conda activate /glade/work/wchapman/conda-envs/credit-derecho-mar2026   

#LMK if those eventually work you can pip install -e to them too. Right now we are not fully using the pbs options in the script, but I can make that the highest priority … the CLI has options for shit but now I think its kind of janky experienced users will have shit in the config down already 

credit submit --cluster derecho -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config.yml --dry-run

# That's it. No flags needed. The job plan will print your account so you can confirm before anything submits:

# ====================================================
#   Job plan
# ====================================================
#   Cluster  : derecho
#   Account  : MYACCOUNT
#   Config   : my_config.yml
#   GPUs     : 4 GPU(s) × 2 nodes (8 total)
#   Walltime : 06:00:00 per job
#   Chain    : 1 job (no chaining)
# ====================================================
# Use --dry-run to see the full PBS script before committing. Override anything one-off at the CLI — e.g. --account DIFFERENT_ACCOUNT --walltime 01:00:00 for a quick test run without touching the config.
# And use the conda env I listed before that … it will work now for you be default as its set. You can also control using CLI options 


# And use the conda env I listed before that … it will work now for you be default as its set. You can also control using CLI options 

# Git pull from the branch and hopefully it should freaking work for you 