# CAMulator Climate Run Guide

---

Optimal settings for starting an interactive job on CASPER: 

 - 8 CPUS 
 - 250 GB memory 
 - 1 GPU 
 - Type: H100 80GB for fastest runs 

## Environment

Always activate the correct conda environment before running any of these scripts:

```bash
conda activate credit-casper-mar2026
```

Install this toolbox as an editable package (required once per environment, or after pulling new changes):

```bash
cd /glade/work/wchapman/Roman_Coupling/train_johns
pip install -e . --no-deps
```

Get an interactive Casper node with a GPU (required for all steps below):

```bash
qsub -I -A NAML0001 -l select=1:ncpus=32:ngpus=1:mem=250GB -l walltime=8:00:00 -q casper -l gpu_type=h100
```

All climate scripts live in:
```
/glade/work/wchapman/Roman_Coupling/train_johns/climate/
```

---

## Config File

The primary config for the extended (17-diagnostic) run is:

```
/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml
```

**Key fields to check/update before each run:**

```yaml
predict:
  start_datetime: '1981-01-01 00:00:00'   # Must match forecasts section exactly
  forecasts:
    start_year: 1981                        # Non-leap year (CESM uses no-leap calendar!)
    start_month: 1
    start_day: 1
    start_hours: [0]
    duration: 1
  save_forecast: '/glade/derecho/scratch/wchapman/CREDIT/climate_extended/'
  timesteps_fast_climate: 14600            # 6-hr steps (14600 = 10 years)
  init_cond_fast_climate: '...init_camulator_condition_tensor_1981-01-01T00Z.pth'
```

> **IMPORTANT**: The training data uses a **no-leap calendar** (365 days/year, no Feb 29).
> Always use a non-leap start year (e.g. 1981, 1983, 1985, 1987) or you will get a
> `ValueError: <timestamp> is not in list` crash on Feb 29.

---

## Step 1 — Generate Initial Conditions
Q
Only needed once per start date. Reads the zarr training data, runs one forward pass,
and saves a `.pth` tensor that Step 2 uses to initialize the climate integration.

```bash
cd /glade/work/wchapman/Roman_Coupling/train_johns/climate

torchrun Make_Climate_Initial_Conditions.py \
  -c /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml \
  --model_name checkpoint.pt00055.pt
```

**Output** is saved to:
```
<save_loc>/init_times/init_camulator_condition_tensor_YYYY-MM-DDTHhz.pth
# e.g.:
/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended/init_times/init_camulator_condition_tensor_1981-01-01T00Z.pth
```

Then update `init_cond_fast_climate` in the config to point to this file before running Step 2.

**Available extended checkpoints** (in `NEW_CLI_JOHN_CASPER_extended/`):
- `best_checkpoint.pt` — lowest validation loss (recommended)
- `checkpoint.pt00055.pt` — most recent epoch
- `checkpoint.pt000XX.pt` — specific epoch checkpoints

---

## Step 2 — Run Climate Integration

Runs the full autoregressive time-stepping loop.
`timesteps_fast_climate` controls run length (1460 steps = 1 year at 6-hr steps).

```bash
cd /glade/work/wchapman/Roman_Coupling/train_johns/climate

python Quick_Climate.py \
  --config /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml \
  --model_name checkpoint.pt00055.pt \
  --save_append run_extended_001
```

**Key optional flags:**
| Flag | Purpose |
|---|---|
| `--save_append <name>` | Subfolder name appended to `save_forecast` path |
| `--monthly_mean` | Save monthly means only (much smaller output files) |
| `--track_moisture` | Write global mean Qtot per step to `moisture_tracking.csv` |
| `--init_noise <float>` | Add Gaussian noise to IC (for ensemble runs) |

**Output** lands in:
```
<save_forecast>/<save_append>/
# e.g.:
/glade/derecho/scratch/wchapman/CREDIT/climate_extended/run_extended_001/
```

for example, with our runs to find the best model: 


```bash
cd /glade/work/wchapman/Roman_Coupling/train_johns/climate/

python Quick_Climate.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml \
  --model_name checkpoint.pt00055.pt \
  --save_append run_chkpt0055
  --monthly_mean

```

forecasts will be saved to the specified config path, for example, in /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended/config_run_quickclimate.yml: 

```bash 
save_forecast: /glade/derecho/scratch/wchapman/CREDIT/climate_extended/
```

do this and then evaluate the models:
```bash
python Quick_Climate.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml --model_name checkpoint.pt00055.pt --save_append run_chkpt0055 --monthly_mean
python Quick_Climate.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml --model_name checkpoint.pt00054.pt --save_append run_chkpt0054 --monthly_mean
python Quick_Climate.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml --model_name checkpoint.pt00053.pt --save_append run_chkpt0053 --monthly_mean
python Quick_Climate.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml --model_name checkpoint.pt00052.pt --save_append run_chkpt0052 --monthly_mean
python Quick_Climate.py --config /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_run_quickclimate.yml --model_name checkpoint.pt00051.pt --save_append run_chkpt0051 --monthly_mean

```

They are initialised in 2001, so compare them to: 
```
/glade/derecho/scratch/wchapman/b_credit_runs/b.e21.CREDIT_climate_branch_1980_2001_zmdata_ERA5scaled_zmdata_Qtot.zarr
...
/glade/derecho/scratch/wchapman/b_credit_runs/b.e21.CREDIT_climate_branch_1980_2011_zmdata_ERA5scaled_zmdata_Qtot.zarr
```

---

## Step 3 — Post-Process to Daily/Monthly Means

Aggregates 6-hourly NetCDF output into coarser time means.

```bash
cd /glade/work/wchapman/Roman_Coupling/train_johns/climate

python Post_Process.py \
  /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended/config_run_quickclimate.yml \
  1D \
  --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY FSDS_J FLDS_J FSUS FLUS SHFLX LHFLX FSUTOA FLUT \
  --reset_times False --dask_do False \
  --name_string daily_processed \
  --rescale_it False \
  --save_append run_extended_001
```

---

## Checkpoints and Configs

| File | Purpose |
|---|---|
| `NEW_CLI_JOHN_CASPER_extended/config_run_quickclimate.yml` | Primary run config (17 diag vars) |
| `NEW_CLI_JOHN_CASPER_extended/best_checkpoint.pt` | Best fine-tuned weights |
| `NEW_CLI_JOHN_CASPER_extended/training_log.csv` | Per-epoch training/validation loss |
| `b_credit_runs/camulator_config_extended.yml` | Training config (used for fine-tuning, not inference) |

The extended model predicts 17 diagnostic variables vs the original 15:
- **Removed**: FSNS, FLNS, FSNT, FLNT (net fluxes)
- **Added**: FSDS_J, FLDS_J, FSUS, FLUS, FSUTOA, FLUT (up/down components for energy fixer)
- **Kept**: SHFLX, LHFLX (reordered to channels 9-10)

---

## Troubleshooting

**`ValueError: <timestamp> is not in list`**
→ Leap year issue. Change `start_year` to a non-leap year (1981, 1983, 1985...).

**`KeyError: 'tracer_inds'`**
→ `credit_main_parser` not called before loading the model. This is handled automatically
  by `train.py` but custom eval scripts must call it manually.

**`prefetch_factor` crash with `num_workers=0`**
→ Set `valid_thread_workers: 2` (or higher) in the trainer config section.

**`thread_workers > 1` causes silent data duplication with `ERA5_MultiStep_Batcher`**
→ `ERA5_MultiStep_Batcher` manages batch sequencing via an internal `batch_call_count`
  counter. PyTorch DataLoader forks the dataset into each worker, so all workers start
  with the same `batch_call_count`. Every worker independently returns `batch_indices[0]`
  on its first call — meaning each unique batch is repeated `thread_workers × prefetch_factor`
  times before the counter advances.

  **Effect**: with `thread_workers: 4` and `prefetch_factor: 4`, only
  `batches_per_epoch / 16 ≈ 9` unique batches are seen per epoch instead of 150.
  This causes severe under-training and a systematic even/odd epoch oscillation in the
  water budget drift (the first batch of each epoch dominates, and its moisture content
  alternates with the epoch seed).

  **Fix**: always set `thread_workers: 1` and `valid_thread_workers: 1` when using
  `ERA5_MultiStep_Batcher`. You can safely increase `prefetch_factor` (e.g. 4) to
  compensate — the single worker queues batches ahead to hide I/O latency.

**`index 0 is out of bounds for axis 0 with size 0`** in era5_multistep
→ `history_len: 0` is invalid. Set `history_len: 1`.

**`ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group`**
→ Happens when `trainable_layers` changes between runs (e.g. unfreezing all layers after
  training only `up_block4`). The saved optimizer state only has Adam momentum for the
  previously-trainable parameters — it can't be loaded into a larger parameter group.

  **Fix**: set `load_optimizer: False` and `load_scaler: False` whenever you change
  `trainable_layers`. Remember to update both the original config AND `config_reload.yml`
  in the save directory. After the first epoch with the new parameter group, you can
  set them back to `True` for subsequent chained jobs.

**Config changes mid-chain don't take effect on jobs 2–N**
→ When a chain is submitted, `credit submit` writes a `config_reload.yml` into the
  `save_loc` directory. Jobs 2–N load from that file, not the original config.
  Any change made to the original config after submission is ignored by the running chain.

  **Fix**: always update **both** files when changing config mid-chain:
  ```bash
  # 1. Edit the original config
  vi /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended_v2.yml

  # 2. Apply the same change to the reload config
  vi /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_reload.yml
  ```
  Changes take effect at the start of the next job in the chain (not mid-job).

---

## Training the Model

### Configs

| Config | Purpose | Cluster |
|---|---|---|
| `b_credit_runs/camulator_config_extended.yml` | Fine-tuning (15→17 diag vars, fresh optimizer) | Casper |
| `b_credit_runs/camulator_config.yml` | Full training from scratch | Derecho |
| `NEW_CLI_JOHN_CASPER_extended/config_run_quickclimate.yml` | Inference / climate runs only | Casper |

### Key training config fields

```yaml
save_loc: '/glade/derecho/scratch/wchapman/CREDIT_runs/<run_name>/'

trainer:
  load_weights: True        # resume from checkpoint.pt in save_loc
  load_optimizer: False     # False = fresh Adam state (for fine-tuning)
  load_scaler: False        # False = fresh AMP grad scaler
  reload_epoch: True        # continue epoch counter from checkpoint (for chained jobs)

  learning_rate: 5.0e-05    # half of original 1e-4 for fine-tuning
  train_batch_size: 8       # per-GPU; effective = batch_size × n_gpus
  num_epoch: 4              # epochs THIS job runs (~2.3h/epoch × 4 ≈ 9h, fits 12h walltime)
  epochs: 100               # total training budget across all chained jobs
```

> **Checkpointing**: every epoch saves `checkpoint.pt00XXX.pt`.
> `best_checkpoint.pt` is updated whenever validation loss improves.
> `backup_checkpoint.pt` is the start-of-epoch snapshot (useful if a job crashes mid-epoch).

---

### Interactive Training (Casper, 4 GPUs)

Best for debugging, testing config changes, or short runs.

```bash
# Get an interactive node with 4 GPUs
qsub -I -A NAML0001 -l select=1:ncpus=32:ngpus=4:mem=250GB \
     -l walltime=8:00:00 -q casper -l gpu_type=a100

conda activate /glade/work/wchapman/conda-envs/credit-casper-mar2026

cd /glade/work/wchapman/Roman_Coupling/train_johns

torchrun --nproc_per_node=4 \
  credit/applications/train.py \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended.yml
```

Monitor progress:
```bash
tail -f /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended/training_log.csv
```

---

### PBS Submission via `credit submit` (Casper)

The `credit-casper-mar2026` env includes a full `credit submit` CLI that reads
PBS settings from the config's `pbs:` section and generates the script for you.

**Dry-run first** (always do this before submitting):
```bash
cd /glade/work/wchapman/Roman_Coupling/train_johns

credit submit --cluster casper \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended.yml \
  --gpus 4 --dry-run
```

**Submit a fresh run:**
```bash
credit submit --cluster casper \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended.yml \
  --gpus 4
```

**Resume from the latest checkpoint** (`--reload` patches `load_weights/optimizer/scaler/reload_epoch` automatically):
```bash
credit submit --cluster casper \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended.yml \
  --gpus 4 --reload
```

**Chain N back-to-back jobs** (PBS `afterok` dependency — each job auto-reloads):
```bash
# Auto-calculate chain length from ceil(trainer.epochs / trainer.num_epoch):
credit submit --cluster casper \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended.yml \
  --gpus 4 --chain

# Or specify explicitly (e.g., 10 jobs × 4 epochs = 40 epochs total):
credit submit --cluster casper \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended.yml \
  --gpus 4 --chain 10
```

> **How chaining works**: job 1 uses the base config; jobs 2–N are automatic reload
> jobs. Each job's checkpoint becomes the next job's starting point.
> The `pbs:` block in the config sets defaults; all flags can override them one-off.

---

### PBS Submission — Derecho (2 nodes, 8 GPUs, full training)

For full retraining from scratch. Uses `camulator_config.yml`.

```bash
# Dry-run to check the script:
credit submit --cluster derecho \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config.yml \
  --gpus 8 --nodes 2 --dry-run

# Submit with auto-chaining for full training budget:
credit submit --cluster derecho \
  -c /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config.yml \
  --gpus 8 --nodes 2 --chain
```

---

### Monitoring a Training Run

```bash
# Watch validation loss in real time
tail -f /glade/derecho/scratch/wchapman/CREDIT_runs/<run_name>/training_log.csv

# Check job status
qstat -u wchapman

# List checkpoints saved so far
ls -lth /glade/derecho/scratch/wchapman/CREDIT_runs/<run_name>/checkpoint.pt*.pt
```
