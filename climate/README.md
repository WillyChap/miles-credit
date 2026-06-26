# CAMulator — climate inference toolbox

Roll a trained **CAMulator** (CREDIT AI atmosphere) checkpoint forward for
climate-length runs and get **per-year zarr** output. Everything is driven by a
single YAML file and reads its inputs from a local `./assets/` folder, so the
whole thing is self-contained and easy to host (e.g. on HuggingFace).

```
config (camulator_config.yml)
        │
        ▼
Quick_Climate.py ──► pred_*.nc (6-hourly steps)
        │
        ▼
netcdf_to_zarr.py ─► camulator_<year>.zarr   ← the product (yearly zarr)

        ── or, for compact output ──
Quick_Climate.py --daily_mean / --monthly_mean ─► time-averaged NetCDF
```

---

## 1. What's in this folder

| File | Role |
|------|------|
| `camulator_config.yml` | **The driver.** All paths, variables, physics, run length. |
| `Quick_Climate.py` | Autoregressive rollout. Default: per-step `pred_*.nc`. `--daily_mean` / `--monthly_mean`: time-averaged NetCDF. |
| `Model_State.py` | State container + `CAMulatorStepper` (model step + conservation fixers). |
| `WindPP.py` | Wind-artifact post-filter (called automatically by `Model_State`). |
| `netcdf_to_zarr.py` | Consolidate 6-hourly `pred_*.nc` → `camulator_<year>.zarr`. |
| `Make_Climate_Initial_Conditions.py` | *Optional* — build a new initial-condition tensor for a custom start date (NCAR data needed). |
| `RunQuickClimate.sh` | End-to-end driver (rollout → zarr, or averaged NetCDF). PBS or interactive. |
| `download_assets.py` | Pull model + inputs from a HuggingFace repo into `./assets/`. |
| `stage_assets.sh` | NCAR alternative: symlink the GLADE copies into `./assets/`. |
| `assets/` | All model inputs live here (see manifest below). |
| `output/` | Default run output directory. |

---

## 2. Quick start

```bash
# 0. environment (CREDIT / miles-credit installed, PyTorch 2.4 + CUDA)
conda activate /glade/work/wchapman/conda-envs/credit-coupling

# 1. get the assets into ./assets/
python download_assets.py --repo_id <user>/camulator   # anywhere (HuggingFace)
#   ... or, on NCAR, symlink the GLADE copies instead:
./stage_assets.sh

# 2. run everything (rollout → yearly zarr)
bash RunQuickClimate.sh      # interactive GPU node
#   ... or:  qsub RunQuickClimate.sh

# result:
ls output/run_default/zarr/  # camulator_1981.zarr, camulator_1982.zarr, ...
```

**Output modes** — set `AVG` in `RunQuickClimate.sh`:

| `AVG` | What you get |
|-------|--------------|
| `none` (default) | full 6-hourly fields, consolidated into **yearly zarr** |
| `daily` | one daily-mean NetCDF per day (`Quick_Climate.py --daily_mean`) |
| `monthly` | one monthly-mean NetCDF per month (`Quick_Climate.py --monthly_mean`) |

Averaging happens *inside the rollout* (no separate post-processing step), so
`daily`/`monthly` are far smaller and skip the zarr consolidation.

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
`MODEL_NAME` (checkpoint file in `./assets/`), `ZARR_PREFIX`, `RUN_POST`.

Output zarr matches the CREDIT training layout: dims `[time, level, lat, lon]`,
3-D vars chunked `time=1`, float32, hybrid-sigma coefficients attached so the
store is self-sufficient for vertical integration.

---

## 4. Asset manifest (what `./assets/` must contain)

All paths in the config are `./assets/<file>`. On NCAR, `stage_assets.sh`
symlinks these from GLADE. To host elsewhere, ship these files.

**Required for a basic run**

| File in `assets/` | ~Size | Used for | GLADE source |
|---|---|---|---|
| `checkpoint.pt` | 4.8 G | the trained model | `…/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/checkpoint.pt` |
| `mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc` | 0.5 M | normalization mean | `…/b_credit_runs/` |
| `std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc` | 0.5 M | normalization std | `…/b_credit_runs/` |
| `statics_b_credit_runs_f32_02.nc` | 50 M | static inputs + mass/water fixers | `…/b_credit_runs/` |
| `b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc` | 50 M | hybrid-sigma coeffs (post-blocks + zarr) | `…/MLWPS/STAGING/` |
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
<user>/camulator        (HF model repo)
├── checkpoint.pt
├── mean_6h_...nc, std_6h_...nc
├── *statics*.nc
├── b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc
├── init_camulator_condition_tensor_1981-01-01T00Z.pth
└── era5.yaml
```

`download_assets.py --repo_id <user>/camulator` pulls them into `./assets/`.

---

## 5. Custom initial conditions (optional)

To start from a date other than the shipped IC, generate a new tensor (requires
the training zarr on NCAR), then point `init_cond_fast_climate` and
`start_datetime` at it:

```bash
python Make_Climate_Initial_Conditions.py -c ./camulator_config.yml \
       --model_name checkpoint.pt
```

---

## 6. Notes

- 1° grid, 192×288, 32 hybrid-sigma levels, 6-hourly steps.
- The rollout uses a **no-leap (365-day) clock**, matching the forcing calendar,
  so output time stamps stay aligned over multi-year runs.
- Conservation fixers (mass / water / energy-updown) and the wind-artifact
  filter are configured in the `post_conf` and `postprocessing` blocks of the
  YAML and run automatically.
