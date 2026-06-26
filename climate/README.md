# CAMulator — climate inference toolbox

Roll a trained **CAMulator** (CREDIT AI atmosphere) checkpoint forward for
climate-length runs and get **NetCDF** output. Everything is driven by a single
YAML file and reads its inputs from a local `./assets/` folder, so the whole
thing is self-contained and easy to host (e.g. on HuggingFace).

```
config (camulator_config.yml)
        │
        ▼
Quick_Climate.py ──► NetCDF in <save_forecast>/<run>/<init_time>/pred_*.nc
                     • default      : one file per 6-hourly step
                     • --daily_mean : one daily-mean file per day
                     • --monthly_mean: one monthly-mean file per month
```

---

## 1. What's in this folder

| File | Role |
|------|------|
| `camulator_config.yml` | **The driver.** All paths, variables, physics, run length. |
| `Quick_Climate.py` | Autoregressive rollout. Default: per-step `pred_*.nc`. `--daily_mean` / `--monthly_mean`: time-averaged NetCDF. |
| `Model_State.py` | State container + `CAMulatorStepper` (model step + conservation fixers). |
| `WindPP.py` | Wind-artifact post-filter (called automatically by `Model_State`). |
| `Make_Climate_Initial_Conditions.py` | *Optional* — build a new initial-condition tensor for a custom start date (NCAR data needed). |
| `RunQuickClimate.sh` | End-to-end driver (rollout → NetCDF). PBS or interactive. |
| `download_assets.py` | Pull model + inputs from a HuggingFace repo into `./assets/`. |
| `upload_assets.py` | *Maintainer:* create + populate that HuggingFace repo from `./assets/`. |
| `check_setup.py` | Preflight: verify deps, CREDIT, assets, GPU before a run. |
| `stage_assets.sh` | NCAR-only alternative: symlink the GLADE copies into `./assets/`. |
| `assets/` | All model inputs live here (see manifest below). |
| `output/` | Default run output directory. |

---

## 2. Quick start

Requirements: a CUDA GPU (~6 GB free; the rollout JIT-traces the model at run
time) and ~6 GB of asset downloads (~4.5 GB checkpoint + 1.3 GB forcing +
statics). Host RAM ~16 GB is plenty (the `#PBS mem` in `RunQuickClimate.sh` is
deliberately over-provisioned for NCAR).

The default model is **`checkpoint.pt00065.pt`** (training epoch 65). The HF repo
hosts many epochs; pick one with `download_assets.py --checkpoint <name>` and set
`MODEL_NAME` to match — see *Choosing a checkpoint* below.

```bash
# 0. install CREDIT (this repo) + its environment — do this ONCE.
#    `climate/` depends on the parent repo's `credit/` package, so install from
#    the repo ROOT, and use THIS repo's credit (do not `pip install miles-credit`).
git clone -b camulator_huggingface \
    https://github.com/WillyChap/miles-credit.git camulator && cd camulator
conda env create -f environment.yml -n camulator   # PyTorch + CREDIT deps
conda activate camulator
pip install -e . --no-deps                          # from repo root: installs ./credit
cd climate
#  NOTE: use --no-deps. environment.yml already pins the exact runtime stack
#  (torch 2.4.1 + torch_harmonics 0.7.2 + numpy<2); a plain `pip install -e .`
#  re-resolves and upgrades numpy/torch_harmonics, which breaks `import credit`.
#    (NCAR users may instead: conda activate /glade/work/wchapman/conda-envs/credit-coupling-ud)

# 1. get the model + inputs into ./assets/
#    Pass the real HuggingFace repo id (or set CAMULATOR_HF_REPO). See "Getting
#    the assets" below — there is no runnable default until the repo is published.
python download_assets.py --repo_id <org>/camulator    # off-NCAR (HuggingFace)
#   ... or, on NCAR, symlink the GLADE copies instead (no download):
./stage_assets.sh

# 2. verify everything is in place (deps, CREDIT, assets, GPU) — takes seconds
python check_setup.py

# 3. run the rollout
bash RunQuickClimate.sh      # interactive GPU node
#   ... or:  qsub RunQuickClimate.sh   (NCAR PBS; edit CONDA_ENV first)

# result:
ls output/run_default/1981-01-01T00Z/   # pred_*.nc
```

`RunQuickClimate.sh` reads overridable env vars, so you can avoid editing it:

```bash
CONDA_ENV=camulator FOLD_OUT=myrun AVG=monthly bash RunQuickClimate.sh
# CONDA_ENV="" skips conda activation (if you already activated your env)
```

**Output modes** — set `AVG` in `RunQuickClimate.sh`:

| `AVG` | What you get |
|-------|--------------|
| `none` (default) | one NetCDF per 6-hourly step (full resolution) |
| `daily` | one daily-mean NetCDF per day (`Quick_Climate.py --daily_mean`) |
| `monthly` | one monthly-mean NetCDF per month (`Quick_Climate.py --monthly_mean`) |

Averaging happens *inside the rollout* (no separate post-processing step), so
`daily`/`monthly` produce far less data.

Need a GPU node on Casper:

```bash
qsub -I -A <PROJECT> -l select=1:ncpus=32:ngpus=1:mem=250GB \
     -l walltime=8:00:00 -q casper -l gpu_type=a100
```

---

## 3. Configure your run

Edit `camulator_config.yml` → `predict:`

| Key | Meaning |
|-----|---------|
| `start_datetime` | Initial time. **Must match the IC tensor** (`init_cond_fast_climate`). |
| `init_cond_fast_climate` | Initial-condition tensor (`.pth`). Default: `1981-01-01T00Z`. |
| `forcing_file` | Cyclic 1-yr forcing (SOLIN, SST, ICEFRAC, CO₂). Wrapped automatically for multi-year runs. |
| `timesteps_fast_climate` | Number of 6-hourly steps. `1460` = one no-leap year; `1460*N` = N years. |
| `save_forecast` | Output root (default `./output/`). |

…and the experiment knobs in `RunQuickClimate.sh`: `FOLD_OUT` (run name),
`MODEL_NAME` (checkpoint file in `./assets/`, default `checkpoint.pt00065.pt`),
and `AVG` (`none` → 6-hourly, `daily`/`monthly` → averaged).

### Choosing a checkpoint

Many training epochs are hosted (`checkpoint.pt000NN.pt`). Default is epoch 65.
To use a different one, download it and set `MODEL_NAME` to the same name:

```bash
python download_assets.py --repo_id <org>/camulator --checkpoint checkpoint.pt00058.pt
MODEL_NAME=checkpoint.pt00058.pt bash RunQuickClimate.sh
# (download several at once: --checkpoints checkpoint.pt000{50..60}.pt)
```

`save_loc` (`./assets/`) is the checkpoint *directory*; `MODEL_NAME` selects the
file within it, so multiple checkpoints can coexist in `./assets/`.

Output NetCDF is written per step/day/month to
`<save_forecast>/<FOLD_OUT>/<init_time>/pred_*.nc`, in physical units, with
variable metadata from `era5.yaml`.

---

## 4. Asset manifest (what `./assets/` must contain)

### Getting the assets

All config paths are `./assets/<file>`. Fill `./assets/` one of two ways:

- **NCAR users:** `./stage_assets.sh` symlinks the GLADE copies (no download).
- **Everyone else:** `python download_assets.py --repo_id <org>/camulator` pulls
  them from HuggingFace.

> ⚠️ **The public HuggingFace asset repo is not published yet.** Until a
> maintainer creates it (see *Hosting layout* below) and updates the
> `--repo_id`, off-NCAR users cannot auto-download — request the assets from the
> maintainers. `download_assets.py` deliberately errors on the placeholder id
> rather than failing silently. Set `CAMULATOR_HF_REPO=<org>/camulator` to avoid
> passing `--repo_id` each time.

**Required for a basic run** (the "origin" column is the NCAR-internal source
`stage_assets.sh` links from; off-NCAR users get these from the HF repo):

| File in `assets/` | ~Size | Used for | NCAR origin |
|---|---|---|---|
| `checkpoint.pt00065.pt` | 4.5 G | the trained model (default epoch; other epochs hosted too) | `…/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/checkpoint.pt00065.pt` |
| `mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc` | 0.5 M | normalization mean | `…/b_credit_runs/` |
| `std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc` | 0.5 M | normalization std | `…/b_credit_runs/` |
| `statics_b_credit_runs_f32_02.nc` | 50 M | static inputs + mass/water fixers | `…/b_credit_runs/` |
| `b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc` | 50 M | hybrid-sigma coeffs (conservation post-blocks) | `…/MLWPS/STAGING/` |
| `f.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc` | 50 M | latitude weights | `…/MLWPS/STAGING/` |
| `b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc` | 1.3 G | cyclic 1-yr forcing | `…/CAMULATOR_FORCING/` |
| `init_camulator_condition_tensor_1981-01-01T00Z.pth` | 29 M | initial condition | `…/NEW_CLI_JOHN_CASPER_extended/init_times/` |
| `era5.yaml` | <1 K | output variable metadata | `/glade/work/schreck/repos/credit/miles-credit/metadata/` |

**Optional (only for `rollout_metrics` / fast-climate scores, not for a plain run)**

| File in `assets/` | Used for |
|---|---|
| `ERA5_clim_1990_2019_6h_interp.nc` | climatology baseline |
| `truth_be21_tensor_2013-01-01T00Z.pth` | seasonal-mean reference |

> The full ERA5/CESM training zarr (`data.save_loc`, still an absolute GLADE
> path in the config) is **not** needed for inference — only
> `Make_Climate_Initial_Conditions.py` reads it, when generating a brand-new IC.

**Hosting layout.** Code (this folder) lives in git; the assets live in a
HuggingFace **model** repo with the manifest files at the repo root:

```
<org>/camulator        (HF model repo)
├── checkpoint.pt00065.pt          # default epoch (others: checkpoint.pt000NN.pt)
├── checkpoint.pt00040.pt … 00079  # additional epochs (optional)
├── mean_6h_...nc, std_6h_...nc
├── *statics*.nc
├── b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc
├── init_camulator_condition_tensor_1981-01-01T00Z.pth
└── era5.yaml
```

`download_assets.py --repo_id <org>/camulator` pulls the shared files + the
default checkpoint into `./assets/`. Maintainers publish the repo with
`upload_assets.py`, e.g. shared files + a range of epochs straight from the run
dir:

```bash
python upload_assets.py --repo_id <org>/camulator --create \
    --checkpoint_dir /glade/.../NEW_CLI_JOHN_CASPER_extended_v2 \
    --checkpoints checkpoint.pt000{40..79}.pt
```

then set that repo id as the default `--repo_id` (or `CAMULATOR_HF_REPO`).

> **Forcing file note:** `b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc` must
> contain the model's static fields (`z_norm`, `LANDM_COSLAT`) in addition to the
> dynamic forcing — the rollout reads static inputs from the forcing file. The
> shipped file already includes them; only relevant if you build your own.

---

## 5. Custom initial conditions (optional)

To start from a date other than the shipped IC, generate a new tensor (requires
the training zarr on NCAR), then point `init_cond_fast_climate` and
`start_datetime` at it:

```bash
python Make_Climate_Initial_Conditions.py -c ./camulator_config.yml \
       --model_name checkpoint.pt00065.pt
```

---

## 6. Notes

- 1° grid, 192×288, 32 hybrid-sigma levels, 6-hourly steps.
- The rollout uses a **no-leap (365-day) clock**, matching the forcing calendar,
  so output time stamps stay aligned over multi-year runs.
- Conservation fixers (mass / water / energy-updown) and the wind-artifact
  filter are configured in the `post_conf` and `postprocessing` blocks of the
  YAML and run automatically.
