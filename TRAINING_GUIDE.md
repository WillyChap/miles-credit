# CAMulator Training Guide
**One-stop reference for training, monitoring, and validating CAMulator on NCAR HPC**

---

## Table of Contents
1. [Setup: Clone the Repo and Environment](#1-setup)
2. [Data and Config Overview](#2-data-and-config)
3. [Submitting Training Jobs](#3-submitting-training-jobs)
4. [Monitoring Runs](#4-monitoring-runs)
5. [Resuming / Chaining Jobs](#5-resuming--chaining-jobs)
6. [Picking Up Will's Existing Run](#6-picking-up-wills-existing-run)
7. [Post-Epoch-2: Stepping Out to Longer Horizons](#7-post-epoch-2-stepping-out-to-longer-horizons)
8. [Post-Epoch-30: Enabling Physics Post-Blocks](#8-post-epoch-30-enabling-physics-post-blocks)
9. [Offline Validation](#9-offline-validation)
10. [Key Files Reference](#10-key-files-reference)
11. [Common Failures and Fixes](#11-common-failures-and-fixes)

---

## 1. Setup

### Copy the repo
```bash
# Log in to Derecho or Casper
ssh <username>@derecho.hpc.ucar.edu   # or casper.hpc.ucar.edu

# Copy the training repo to your work directory
cp -r /glade/work/wchapman/Roman_Coupling/train_johns \
      /glade/work/<username>/Roman_Coupling/train_johns
cd /glade/work/<username>/Roman_Coupling/train_johns
```

### Clone the conda environment

**Casper (A100, recommended for first runs and inference):**
```bash
module load conda
conda create --name credit-casper --clone /glade/work/wchapman/conda-envs/credit-casper-mar2026
conda activate credit-casper
pip install -e .   # editable install from your repo copy
```

**Derecho (H100, for serious multi-node training):**
```bash
# These exact modules must be loaded every time you log in to Derecho before activating the env
module load ncarenv/24.12 gcc/12.4.0 ncarcompilers cray-mpich/8.1.29 cuda/12.3.2 conda/latest

conda create --name credit-derecho --clone /glade/work/wchapman/conda-envs/credit-derecho-mar2026
conda activate /glade/work/<username>/conda-envs/credit-derecho
cd /glade/work/<username>/Roman_Coupling/train_johns
pip install -e .
```

> **Note:** The `-e` (editable) install means any edits you make to `.py` files take effect immediately — no reinstall needed. You only run `pip install -e .` once after cloning.

### Copy and update the config files

**Copy Will's configs to your own scratch space:**
```bash
mkdir -p /glade/derecho/scratch/<username>/b_credit_runs

cp /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_casper.yml \
   /glade/derecho/scratch/<username>/b_credit_runs/camulator_config_casper.yml

cp /glade/derecho/scratch/wchapman/b_credit_runs/camulator_config.yml \
   /glade/derecho/scratch/<username>/b_credit_runs/camulator_config.yml
```

**Every path you MUST change in `camulator_config_casper.yml` (Casper):**
```yaml
# Top of file
save_loc: '/glade/derecho/scratch/<username>/CREDIT_runs/my_experiment/'

# pbs section (bottom of file)
pbs:
    conda: "/glade/work/<username>/conda-envs/credit-casper"
    project: "P93300042"
```
Everything else (data paths, mean/std, statics) points to Will's scratch/campaign and is **shared — do not change**.

**Every path you MUST change in `camulator_config.yml` (Derecho):**
```yaml
# Top of file
save_loc: '/glade/derecho/scratch/<username>/CREDIT_runs/my_experiment_derecho/'

# pbs section (bottom of file)
pbs:
    conda: "/glade/work/<username>/conda-envs/credit-derecho"
    project: "P93300042"
```
Again, all data paths remain pointing to Will's scratch — they are shared read-only.

**Quick sanity check — confirm the data paths are correct before submitting:**
```bash
# These should all return files, not errors:
ls /glade/derecho/scratch/wchapman/b_credit_runs_f32_02/ | head -3
ls /glade/derecho/scratch/wchapman/b_credit_runs/mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc
ls /glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc
```

---

## 2. Data and Config Overview

### Training data
The rechunked, NaN-safe zarr files used for training live here (do not modify):
```
/glade/derecho/scratch/wchapman/b_credit_runs_f32_02/
  b.e21.CREDIT_climate_branch_1980_????_zmdata_ERA5scaled_zmdata_Qtot.zarr
```
- **Train period:** 1980–2012
- **Validation period:** 2013–2014

### Normalization files
```
/glade/derecho/scratch/wchapman/b_credit_runs/
  mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc
  std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc
```

### Static file
```
/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/
  b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc
```

### Model architecture (small model — start here)
```yaml
model:
    dim: [128, 256, 512, 1024]
    depth: [2, 2, 12, 2]          # ~183M parameters
    pad_lat: 24
    pad_lon: 24
    cross_embed_kernel_sizes:
    - [4, 8, 16, 24]
```

### Key trainer settings (already tuned — do not change without reason)
```yaml
trainer:
    activation_checkpoint: False   # not needed for 183M model on 80GB GPU
    update_learning_rate: False    # MUST be False — True bypasses warmup and causes NaN
    use_compile: False             # spectral norm is incompatible with torch.compile
    amp: True
    mixed_precision:
        param_dtype: "bfloat16"    # 3× speedup over float32
    thread_workers: 4
    valid_thread_workers: 4
    prefetch_factor: 2             # DO NOT increase to 4+ without checking memory
    warmup_steps: 1200             # ~2 epochs on Casper 4-GPU
    total_steps: 18000             # ~30 epochs on Casper 4-GPU
```

> **OOM warning:** Real memory per sample is ~63 MB (not the ~9 MB the preflight log shows — that estimate has a known bug). With `workers=4, prefetch=2, batch=20`, each rank uses ~10 GB. With 4 ranks that's ~40 GB on a 256 GB node — safe. `prefetch=4` doubles this to 80 GB, which is still OK but tighter. **Never use prefetch ≥ 8 with workers ≥ 4.**

---

## 3. Submitting Training Jobs

### Casper (4 GPUs, 1 node, ~1.5 hr/epoch)
```bash
module load conda
conda activate /glade/work/<username>/conda-envs/credit-casper
cd /glade/work/<username>/Roman_Coupling/train_johns

credit submit --cluster casper -c /glade/derecho/scratch/<username>/camulator_config_casper.yml
```

Dry-run first to preview the PBS script:
```bash
credit submit --cluster casper -c /path/to/config.yml --dry-run
```

### Derecho (8 GPUs, 2 nodes, ~30 min/epoch)
```bash
# Must load these modules first — every session on Derecho
module load ncarenv/24.12 gcc/12.4.0 ncarcompilers cray-mpich/8.1.29 cuda/12.3.2 conda/latest
conda activate /glade/work/<username>/conda-envs/credit-derecho
cd /glade/work/<username>/Roman_Coupling/train_johns

credit submit --cluster derecho -c /glade/derecho/scratch/<username>/b_credit_runs/camulator_config.yml
```

### Interactive node (for debugging only — single GPU)
```bash
# Casper interactive
qsub -I -A P93300042 -l select=1:ncpus=16:ngpus=1:mem=128GB -l walltime=4:00:00 -q casper -l gpu_type=a100

# Then run directly (single GPU — NCCL will error if you try DDP)
module load conda && conda activate /glade/work/<username>/conda-envs/credit-casper
cd /glade/work/<username>/Roman_Coupling/train_johns
CUDA_VISIBLE_DEVICES=0 python applications/train.py -c /path/to/config.yml
```

---

## 4. Monitoring Runs

### Watch the live log
Job logs go to `<repo_dir>/ps_wxformer_CESM_6h.o<jobid>` (job name set in PBS config):
```bash
tail -f /glade/work/<username>/Roman_Coupling/train_johns/ps_wxformer_CESM_6h.o<jobid>
```

### What healthy output looks like
```
Model parameters: 182.00M total, 182.00M trainable
Epoch: 0 train_loss: 1.88  train_acc: 0.0001  lr: 0.000001  4.4s/it
Epoch: 0 train_loss: 1.55  train_acc: 0.0117  lr: 0.000025  4.4s/it
...
Epoch: 1 valid_loss: 2.10  valid_acc: 0.05    valid_mae: 0.95
```

> **Why valid_loss > train_loss in early epochs:** Validation uses EMA (exponentially smoothed) weights with `decay=0.9999`. The EMA window is ~10,000 steps, so before epoch ~16 it still contains a large fraction of early random-init weights. This makes valid_loss look bad. **It is not overfitting.** The actual saved checkpoints are from the current training weights, not EMA. Trust the training loss curve and validate offline.

### Check saved checkpoints
```bash
ls /glade/derecho/scratch/<username>/CREDIT_runs/my_experiment/
# checkpoint.pt          ← latest epoch (used for resuming)
# checkpoint.pt000XX.pt  ← per-epoch saves
# best_checkpoint.pt     ← best valid_loss epoch (unreliable early — see above)
# backup_checkpoint.pt   ← previous epoch backup
# training_log.csv       ← full loss/metric history
```

### Plot training progress
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/glade/derecho/scratch/<username>/CREDIT_runs/my_experiment/training_log.csv')
fig, ax = plt.subplots()
ax.plot(df['train_loss'], label='train')
ax.plot(df['valid_loss'], label='valid (EMA — ignore until epoch ~16)')
ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend()
plt.show()
```

### Check queue / job status
```bash
qstat -u $USER          # all your jobs
qstat -f <jobid>        # details for one job
qdel <jobid>            # cancel a job
gladequota              # check remaining project hours
```

---

## 5. Resuming / Chaining Jobs

The config is set to auto-resume. When a job finishes (or hits walltime), just resubmit:

```bash
# Config already has load_weights/optimizer/scheduler: True
# reload_epoch: True means it reads the epoch from checkpoint.pt automatically
credit submit --cluster casper -c /path/to/config.yml
```

**Important:** `num_epoch` in the config is the per-job cap, not total epochs. It is set conservatively to fit within walltime:
- Casper (1.5 hr/epoch, 12 hr wall): `num_epoch: 7`
- Derecho (~30 min/epoch, 12 hr wall): `num_epoch: 22`

`epochs: 30` is the total training target. Once `checkpoint["epoch"] + 1 >= epochs`, training stops.

---

## 6. Picking Up Will's Existing Run

Will has an ongoing Casper training run. Rather than starting from scratch, you can copy his checkpoints to your own scratch directory and continue from where he left off. **Always copy — never write directly into Will's directory.**

### Step 1: Copy the checkpoints

```bash
# Create your own experiment directory
mkdir -p /glade/derecho/scratch/<username>/CREDIT_runs/casper_resume/

# Copy the three files needed to resume
cp /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_small/checkpoint.pt \
   /glade/derecho/scratch/<username>/CREDIT_runs/casper_resume/

cp /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_small/checkpoint_ema.pt \
   /glade/derecho/scratch/<username>/CREDIT_runs/casper_resume/

cp /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_small/training_log.csv \
   /glade/derecho/scratch/<username>/CREDIT_runs/casper_resume/
```

You can also copy the per-epoch checkpoints if you want a full history:
```bash
cp /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_small/checkpoint.pt*.pt \
   /glade/derecho/scratch/<username>/CREDIT_runs/casper_resume/
```

### Step 2: Confirm what epoch you're resuming from

```bash
python -c "
import torch
ckpt = torch.load('/glade/derecho/scratch/<username>/CREDIT_runs/casper_resume/checkpoint.pt',
                  map_location='cpu', weights_only=False)
print('Resuming from epoch:', ckpt['epoch'])
"
```

### Step 3: Update your config

In your copy of `camulator_config_casper.yml`, make these changes:

```yaml
# Point save_loc at YOUR copy of the checkpoints
save_loc: '/glade/derecho/scratch/<username>/CREDIT_runs/casper_resume/'

trainer:
    # Load all states to resume seamlessly
    load_weights: True
    load_optimizer: True
    load_scaler: True
    load_scheduler: True
    reload_epoch: True    # reads epoch number from checkpoint.pt automatically

    # Total target and per-job cap
    epochs: 30            # keep training until epoch 30 total
    num_epoch: 7          # stop after 7 epochs per job (~10.5 hr on Casper)

    # These must stay False
    update_learning_rate: False
    use_compile: False
    activation_checkpoint: False
```

Everything else in the config (data paths, model architecture, AMP settings) stays exactly as-is — the checkpoint was trained with those settings and they must match.

### Step 4: Submit

```bash
module load conda
conda activate /glade/work/<username>/conda-envs/credit-casper
cd /glade/work/<username>/Roman_Coupling/train_johns

credit submit --cluster casper \
  -c /glade/derecho/scratch/<username>/b_credit_runs/camulator_config_casper.yml
```

### Step 5: Verify it picked up correctly

In the job log, you should see within the first few lines:
```
INFO: Resuming from epoch X    ← should be Will's last epoch + 1
Model parameters: 182.00M total
Preflight: first batch ready in 25s
Epoch: X train_loss: ...       ← loss should start where Will left off, not at ~6.0
```

If `train_loss` starts near 6.0, the checkpoint was not loaded — check that `save_loc` points to the directory containing `checkpoint.pt`.

---

## 7. Post-Epoch-2: Stepping Out to Longer Horizons

After ~2 epochs (once the model is no longer completely random), begin validating on longer rollouts. This does **not** require retraining — run offline.

### Get an interactive node
```bash
qsub -I -A P93300042 -l select=1:ncpus=32:ngpus=1:mem=250GB -l walltime=8:00:00 -q casper -l gpu_type=a100
module load conda && conda activate /glade/work/<username>/conda-envs/credit-casper
cd /glade/work/<username>/Roman_Coupling/train_johns/climate
```

### Step 1: Make initial conditions (once per start date)
Pre-made ICs for common dates live here — use them if your date is covered:
```
/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/
```

To make your own:
```bash
python Make_Climate_Initial_Conditions.py \
  -c ./camulator_config.yml \
  --model_name checkpoint.pt00002.pt   # use epoch-2 checkpoint
```

### Step 2: Run a short rollout (e.g., 2 weeks = 56 steps)
```bash
python Quick_Climate.py \
  --config ./camulator_config.yml \
  --model_name checkpoint.pt00002.pt \
  --save_append epoch2_test
```

In your config's `predict` section, control length:
```yaml
predict:
    timesteps_fast_climate: 56      # 56 × 6h = 14 days
    # 1460 = 1 year, 7300 = 5 years
    start_datetime: '2000-01-01 00:00:00'
    init_cond_fast_climate: '/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/init_be21_condition_tensor_2000-01-01T00Z.pth'
    save_forecast: '/glade/derecho/scratch/<username>/CREDIT/epoch2_test/'
```

### Step 3: Post-process
```bash
python Post_Process.py ./camulator_config.yml 1D \
  --variables U V T Qtot PS PRECT TREFHT TS TAUX TAUY \
  --reset_times False \
  --dask_do False \
  --name_string epoch2_test \
  --rescale_it False \
  --save_append epoch2_test
```

### Progression schedule
| After epoch | Rollout to test | What to look for |
|-------------|-----------------|------------------|
| 2 | 2 weeks | No NaN, loss decreasing, fields look meteorological |
| 5 | 1 month | Seasonal cycle emerging, PRECT plausible |
| 15 | 3 months | EMA valid_loss improving, T/U patterns realistic |
| 30 | 1 year | Full annual cycle, compare to CAM6 climatology |

---

## 8. Post-Epoch-30: Enabling Physics Post-Blocks

After epoch 30 with a stable, well-trained model, enable the physics fixers to enforce conservation laws. These are off during initial training to avoid interfering with the loss landscape.

Open your config and change the following in the `post_conf` section:

### Recommended order: enable one at a time, retrain a few epochs, verify

**Step A — Tracer fixer** (prevents negative humidity/cloud fractions):
```yaml
        tracer_fixer:
            activate: True             # changed from False
            activate_outside_model: True
            tracer_name: ['Qtot', 'PRECT', 'U10', 'CLDHGH', 'CLDLOW', 'CLDMED']
            tracer_thres: [0, 0, 0, 0, 0, 0]
```

**Step B — Global mass fixer** (dry air mass conservation):
```yaml
        global_mass_fixer:
            activate: True             # changed from False
            activate_outside_model: True
```

**Step C — Global water fixer** (P-E balance):
```yaml
        global_water_fixer:
            activate: True             # changed from False
            activate_outside_model: True
```

**Step D — Global energy fixer** (TOA + surface energy balance):
```yaml
        global_energy_fixer_updown:
            activate: True             # changed from False
            activate_outside_model: True
```

### Also update trainer for fine-tuning phase
```yaml
trainer:
    load_weights: True
    load_optimizer: False    # reset optimizer — start fresh from lower LR
    load_scheduler: False
    update_learning_rate: True   # only time this is safe — setting a new lower LR
    learning_rate: 1.0e-05       # 10× lower than initial training
    warmup_steps: 0
    epochs: 50                   # extend target
    num_epoch: 7                 # per-job cap (same as before)
```

---

## 9. Offline Validation

Run these on an interactive Casper node after training.

### Short deterministic forecast skill (2-week RMSE)
```bash
# Uses the CREDIT rollout evaluation tool
python applications/rollout_metrics.py \
  -c /path/to/config.yml \
  --model_name checkpoint.pt00029.pt \
  --start_datetime "2013-01-01 00:00:00" \
  --n_steps 56
```

### Long climate run (1 year) for climatology comparison
```bash
# Update config predict section first (timesteps_fast_climate: 1460)
python climate/Quick_Climate.py \
  --config /path/to/config.yml \
  --model_name checkpoint.pt00029.pt \
  --save_append validation_yr2013
```

Then compare against the CAM6 zarr training data or ERA5 climatology:
```
/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/ERA5_clim_1990_2019_6h_interp.nc
```

### Key variables to check
| Variable | What bad looks like |
|----------|-------------------|
| `TREFHT` | Drift > 2K after 1 year |
| `PRECT` | Negative values (need tracer fixer) or global mean drift |
| `PS` | Oscillating or drifting globally (mass fixer needed) |
| `U`, `T` | Checkerboard noise at model top (levels 0-5) |
| `ICEFRAC` | Large jumps in sea-ice edge (check ICEFRAC coupling) |

---

## 10. Key Files Reference

| File | Purpose |
|------|---------|
| `camulator_config_casper.yml` | Casper training config (1 node, 4 GPUs) |
| `camulator_config.yml` | Derecho training config (2 nodes, 8 GPUs) |
| `credit/trainers/trainerERA5v2.py` | Main training loop |
| `credit/trainers/base_trainer.py` | Epoch loop, EMA, checkpointing |
| `credit/trainers/preflight.py` | Pre-training memory + hang checks |
| `credit/models/camulator.py` | Crossformer model architecture |
| `climate/Quick_Climate.py` | Long rollout inference script |
| `climate/Make_Climate_Initial_Conditions.py` | Create IC tensors |
| `climate/Post_Process.py` | Aggregate 6-hourly output to daily/monthly |
| `applications/train.py` | Training entrypoint |

**Pre-trained checkpoint (epoch 91, well-validated):**
```
/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/CAMulator_models/checkpoint.pt00091.pt
```

**Pre-made initial conditions:**
```
/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/
```

---

## 11. Common Failures and Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Job killed at step ~10, no error | OOM — Linux silently kills rank 0 | Reduce `prefetch_factor` or `thread_workers` |
| `NaN` at step 2 | `update_learning_rate: True` bypasses warmup | Set `update_learning_rate: False` |
| `NaN` at step 2 with `use_compile: True` | Spectral norm + torch.compile incompatible | Set `use_compile: False` |
| DDP hangs at step 10 | Metrics calling `dist.all_reduce` 588×, ranks diverge | Already fixed in trainer — don't revert |
| `valid_loss >> train_loss` in early epochs | EMA lag — not real overfitting | Ignore until epoch ~16; validate offline |
| `CUDA out of memory` during forward | Big model + large padding + batch=20 | Enable `activation_checkpoint: True` |
| `einops` shape error on startup | Invalid padding: `(192 + 2*pad) / 2` must be divisible by 3 | Only use `pad=0`, `pad=24`, or `pad=48` |
| First batch takes > 5 minutes | zarr cold-cache on Lustre | Normal — subsequent batches are fast |
| ACC stuck at 0.000 | bfloat16 tensors passed to metric | Already fixed (`.float()` cast) — don't revert |
| `NCCL: duplicate GPU` error | DDP launched on interactive node with 1 GPU | Use `CUDA_VISIBLE_DEVICES=0 python ...` instead of `torchrun` |
