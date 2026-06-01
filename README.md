# SUMO Supermodel — CAM6 (CESM2.1.5) steered by GraphCast / CAMulator

> Run CESM2.1.5 with CAM6 nudged at 6-hour intervals toward a neural-surrogate atmospheric prediction (GraphCast and/or CAMulator). Designed to be portable to any HPC that can build CESM and host a single NVIDIA A100 GPU. **Operational details: see [`climate/To_Run_SUMO.md`](./climate/To_Run_SUMO.md).**

---

## Overview

**SUMO** ("supermodel") couples CESM2.1.5 / CAM6 with one or two neural-network atmospheric models that run as **external GPU services**. The cycle at every 6 h CAM6 coupling boundary:

1. CAM6 writes its instantaneous atmospheric state to an `h1` history file, then pauses at a Fortran barrier (`sumo_barrier_after_h1` in `cam_comp.F90`).
2. A neural surrogate — **GraphCast** (JAX, 0.25° ERA5 grid), **CAMulator** (PyTorch, CAM6 native grid), or both blended — reads the h1, predicts the state at T+6 h, and writes a nudge-target file in the CESM run directory.
3. CAM6 reads that nudge file via the standard CESM nudging toolbox, relaxes its U/V (and optionally T, Q, PS) toward the target, and integrates forward.

The coupling is **file-handshake**, not in-process: CESM and the neural surrogates are independent batch jobs that synchronize through lifecycle flag files on a shared scratch filesystem. **The only modification to CAM6 is the barrier subroutine, gated on a sentinel file** (`sumo_active.flag`) in the run directory. The same `cesm.exe` runs in SUMO and non-SUMO mode unchanged — without the sentinel the barrier is a single integer compare per call.

## Architecture

```
   Compute partition                                    GPU partition
   (CESM compute nodes)                                 (single A100 node)
   ┌──────────────────────────┐                       ┌──────────────────────────┐
   │  CESM2.1.5 / CAM6        │                       │  Neural-surrogate server │
   │   ├─ writes h1 every 6h  │     cesm_h1_ready ───►│   ├─ reads h1            │
   │   ├─ sumo_barrier_after  │     ◄─── nudge.nc     │   ├─ runs inference      │
   │   │   _h1 (cam_comp.F90) │                       │   ├─ reverse-translates  │
   │   │   gated on           │     coord_done.flag◄──┤   └─ writes nudge.nc     │
   │   │   sumo_active.flag   │                       │                          │
   │   └─ CESM nudging        │                       │  Coordinator             │
   │      toolbox reads       │                       │   ├─ flag protocol       │
   │      sumo_cam6_nudge.*nc │                       │   └─ orchestrates server │
   └──────────────────────────┘                       └──────────────────────────┘
              ▲                                                  ▲
              └─────── shared /scratch filesystem ───────────────┘
                       (h1 files, nudge files, lifecycle flags)
```

## Pick your mode

| Mode | Surrogate | Where it runs | Status |
|---|---|---|---|
| `graphcast` | GraphCast (DeepMind, JAX) at 0.25° ERA5 grid | 1× A100 (40 GB OK warm, 80 GB cold) | Validated (Jun 2026) |
| `camulator` | CAMulator (CREDIT, PyTorch) on CAM6 native grid | 1× A100 (40 GB) | Validated (May 2026) |
| `both` | CAMulator + GraphCast blended on CAM6 grid | 2× A100 | Servers + reverse translator done; blend coordinator TODO |

---

## Prerequisites

You should already know how to:
- Build and submit a CESM case (`create_newcase`, `case.setup`, `case.build`, `case.submit`)
- Submit a batch job on your HPC
- Manage conda environments

Your HPC must have:
- A compute partition that can build CESM2.1.5 at `f09_g17` resolution (~290 PEs typical)
- At least one NVIDIA A100 GPU node (80 GB recommended for the first cold JAX compile of GraphCast; 40 GB is fine for steady-state inference and for CAMulator)
- A **shared scratch filesystem** visible from both partitions with POSIX semantics (the file-flag handshake assumes new files become visible within seconds across both jobs)
- The usual CESM build dependencies (Intel/GCC, MPI, NetCDF, PnetCDF, PIO)

You do **not** need FTorch. `USE_FTORCH` defaults to `FALSE` in this CESM fork; the SUMO neural surrogates run out-of-process.

---

## Install on a new HPC

### 1. Clone the CESM2.1.5 fork

This fork is CESM2.1.5 + the SUMO barrier in CAM6 + the SST=0 fix for DATA ATM + a portable Makefile (FTorch made optional).

```bash
git clone -b release-cesm2.1.5-camulator git@github.com:WillyChap/CESM.git
cd CESM
./manage_externals/checkout_externals
```

`manage_externals` pulls the matching CAM and CIME forks automatically (`WillyChap/CAM:coupled_camulator`, `WillyChap/cime:coupled_camulator`).

### 2. Clone this CREDIT fork

```bash
git clone -b camulator-sumo git@github.com:WillyChap/miles-credit.git
cd miles-credit
```

### 3. Build the conda environment

A single env (`supermodel`) serves the whole stack — GraphCast (JAX), CAMulator (PyTorch), and the CREDIT package itself. The `environment.yml` at the repo root is **curated from a verified working build on NCAR Casper** (Python 3.10, PyTorch 2.6 + CUDA 12.4, JAX 0.6 + CUDA 12); major versions are pinned, the rest is left to the solver so it ports cleanly to other HPCs.

```bash
conda env create -f environment.yml -n supermodel
conda activate supermodel
pip install -e .                 # install CREDIT (this repo) editable
pip install -e ./graphcast       # install vendored DeepMind GraphCast
```

The env builds for **CUDA 12.x on linux-x86_64**. On a host with a different CUDA major (e.g. 11.x), edit the pip section of `environment.yml`: bump the `--extra-index-url` to the matching PyTorch wheel index and swap the `jax-cuda12-*` plugins for the `jax-cuda11-*` equivalents.

For an exact lockfile-style reproduction of the NCAR Casper build (less portable, more reproducible), see the comment at the bottom of `environment.yml`. The legacy `credit-coupling` env (PyTorch only, no JAX) also still works for the CAMulator-only path — it's a strict subset of what `supermodel` provides.

### 4. Stage the model weights and initial conditions

**GraphCast** weights are not in git (140 MB params exceeds GitHub limits). Download from DeepMind's public bucket:

```bash
mkdir -p graphcast/params graphcast/stats
# Public bucket: https://console.cloud.google.com/storage/browser/dm_graphcast
# Required files:
#   params/  : graphcast_params_GraphCast - ERA5 1979-2017 - resolution 0.25 -
#              pressure levels 37 - mesh 2to6 - precipitation input and output.npz
#   stats/   : diffs_stddev_by_level.nc, mean_by_level.nc, stddev_by_level.nc
```

**CAMulator** weights and the matching normalization/static/IC files live on NCAR's glade. If you are not on glade, request copies from the maintainers and place them at any local paths; update `climate/camulator_config.yml` accordingly. Originals:

| Asset | Source path on glade |
|---|---|
| Checkpoint (`extended_v2`, 3.3 GB) | `/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/checkpoint.pt00044.pt` |
| Training config (sub-paths inside) | `/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/camulator_config_extended_v2.yml` |
| Mean / std stats | `/glade/derecho/scratch/wchapman/b_credit_runs/{mean,std}_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc` |
| Static fields (orography, land mask) | `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc` |
| Initial-condition tensors | `/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/init_times/init_camulator_condition_tensor_YYYY-MM-DDTHHz.pth` (plus a `sumo_init_…` variant matched to a CAM6 restart, generated by `climate/make_sumo_ic_from_cam6_restart.py`) |

### 5. Per-machine config

- On Cray/MPICH systems, add to `env_mach_specific.xml`: `MPICH_GPU_SUPPORT_ENABLED=0`, `FI_CXI_DISABLE_HOST_REGISTER=1`, `MPICH_SMP_SINGLE_COPY_MODE=NONE`. (The example case-setup script `climate/setup_SUMO_B_case.sh` does this automatically for Derecho — adapt for your machine.)
- For GraphCast, set a persistent JAX XLA compilation cache directory (`JAX_COMPILATION_CACHE_DIR` env var, defaulted by `regrid/graphcast_server.py`). Cold compile ~18 min on first call; warm cache ~10 s per inference after.
- Edit the absolute paths near the top of `climate/setup_SUMO_B_case.sh` (`CESM_ROOT`, `CASE_DIR`, `REFDIR`, `PROJECT`, `MACH`, `COMPILER`) to match your install.

---

## Set up the CESM case

The case is shared between modes — same CAM6 build, same `user_nl_cam` nudging setup, same `fincl2` history variables. Mode selection happens on the GPU side.

```bash
cd miles-credit/climate
bash setup_SUMO_B_case.sh         # create + setup + build (one-shot, ~25-50 min)
```

This script does, in order: `create_newcase` (B compset, `f09_g17`, hybrid run from a CESM2-LE refdate), all the SUMO-specific `xmlchange` and `user_nl_*` writes, restart symlinking from your `REFDIR`, the MPICH env-var patches, and `case.build`. It also creates `sumo_active.flag` in the run directory to arm the CAM6 barrier.

Read the script — every section is annotated with the physics rationale (FV CFL, Force_Opt, Nudge_Vwin, fincl2 fields required by GraphCast vs. CAMulator, etc.).

---

## Run a SUMO experiment

Pick the mode you want. **Both modes require the CESM case from the previous step, both use the same `sumo_cam6_nudge.*.nc` file format, both share `climate/reset_run_dir.sh` between attempts.** What differs is which GPU-side server you launch.

### Mode A — GraphCast

```bash
# (1) Reset the run directory (one-time per attempt)
bash climate/reset_run_dir.sh

# (2) Submit the GPU-side stack (one A100; starts server + coordinator)
qsub -v MODE=graphcast,RUNDIR=/path/to/case/run \
     climate/submit_supermodel.pbs

# (3) Submit CESM (separate job, any time after step 2 — coordinator polls)
cd /path/to/case && ./case.submit
```

The GPU job starts `regrid/graphcast_server.py` (long-lived JAX inference server) and `climate/gc_coordinator.py` (orchestrates the per-step file handshake). The very first coupling boundary is free-running because GraphCast needs a 12 h input bundle (T-6h and T) which CAM6 does not yet have at step 0.

### Mode B — CAMulator

```bash
# (1) Reset
bash climate/reset_run_dir.sh

# (2) Start the CAMulator server (Casper / GPU node)
conda activate supermodel  # or: credit-coupling
python climate/camulator_sumo_server.py \
    --config     climate/camulator_config.yml \
    --model_name checkpoint.pt00044.pt \
    --rundir     /path/to/case/run \
    --init_cond  /path/to/sumo_init_YYYY-MM-DDTHHz.pth \
    --sumo --sumo_vars U V --sumo_tau 6.0 \
    --save_atm_nc camulator_out --daily_mean

# (3) Start the coordinator (any CPU / login node)
python climate/sumo_coordinator.py \
    --rundir   /path/to/case/run \
    --cam6dir  /path/to/case/run \
    --cam6_case <CASENAME> \
    --vars U V --alpha_cam 0.5 \
    --start_ymd YYYYMMDD

# (4) Submit CESM
cd /path/to/case && ./case.submit
```

Generate the `sumo_init_*.pth` IC from a CAM6 restart and a CAMulator-format IC using `climate/make_sumo_ic_from_cam6_restart.py` — call signature in the `setup_SUMO_B_case.sh` epilogue.

### Smoke-test the GraphCast server in isolation

```bash
conda activate supermodel
python -m regrid.smoke_test_graphcast_server
```
Three phases (lifecycle, SIGTERM, SIGKILL). ~20 min cold, ~3 min subsequent. Populates the JAX cache.

---

## Where to read next

| For… | See |
|---|---|
| **Full operational guide** — per-mode launch sequence, file-handshake protocol step-by-step, status / stop commands, troubleshooting matrix | **[`climate/To_Run_SUMO.md`](./climate/To_Run_SUMO.md)** |
| Reference case setup (commented, every namelist explained) | `climate/setup_SUMO_B_case.sh` |
| GraphCast server internals (warmup, bundle build, reverse-translate) | `regrid/graphcast_server.py` + `regrid/run_graphcast_inference.py` |
| CAMulator server internals (state injection, SST/ICEFRAC reads) | `climate/camulator_sumo_server.py` + `climate/Model_State.py` |
| Coordinator internals (flag protocol) | `climate/gc_coordinator.py` (GraphCast) and `climate/sumo_coordinator.py` (CAMulator) |
| GraphCast ↔ CAM6 stencil and vertical interpolation | `regrid/` directory (translator blocks, hybrid pressure, blender) |

If you hit an issue not in the troubleshooting matrix, please open an issue against this fork.

---

# NSF NCAR MILES Community Research Earth Digital Intelligence Twin (CREDIT)

[![DOI](https://zenodo.org/badge/710968229.svg)](https://doi.org/10.5281/zenodo.14361005)

[PyPI](https://pypi.org/project/miles-credit/)

[CREDIT npj Climate and Atmospheric Science Article](nature.com/articles/s41612-025-01125-6)

## About
CREDIT is an open software platform to train and deploy AI atmospheric prediction models. CREDIT offers fast models 
that can be flexibly configured both in terms of input data and neural network architecture. The interface is designed
to be user-friendly and enable fast spin-up and iteration. CREDIT is backed by the AI and atmospheric science expertise
of the MILES group and the NSF National Center for Atmospheric Research, leading to design choices that balance advanced
AI/ML with our physical knowledge of the atmosphere.

CREDIT has reached its first stable release with a full set of models, training, and deployment options. It continues
to be under active development. Please contact [the MILES group](mailto:milescore@ucar.edu) if you have any questions about CREDIT.

MILES CREDIT also provides more detailed [documentation](https://miles-credit.readthedocs.io/en/latest/) with installation
instructions, how to get started training and deploying models, how to interpret the config files, and full API docs. 

## Citing CREDIT
If you are interested in using CREDIT as part of your research, please cite the following paper:
Schreck, J.S., Sha, Y., Chapman, W. et al. Community Research Earth Digital Intelligence Twin: a scalable framework 
for AI-driven Earth System Modeling. npj Clim Atmos Sci 8, 239 (2025). https://doi.org/10.1038/s41612-025-01125-6

# Model Weights and Data
Model weights for the CREDIT 6-hour WXFormer and FuXi models and the 1-hour WXFormer are available on huggingface.

* [6-Hour WXFormer](https://huggingface.co/djgagne2/wxformer_6h)
* [1-Hour WXFormer](https://huggingface.co/djgagne2/wxformer_1h)
* [6-Hour FuXi](https://huggingface.co/djgagne2/fuxi_6h)

Processed ERA5 Zarr Data are available for download through Globus (requires free account) through the [CREDIT ERA5 Zarr Files](https://app.globus.org/file-manager/collections/2fc90d8f-10b7-44e1-a6a5-cf844112822e/overview) collection.

Scaling/transform values for normalizing the data are available through Globus [here](https://app.globus.org/file-manager/collections/c5a23e21-1bee-4d1e-bb59-77c5dcee7c76). 

CREDIT also supports realtime runs generated from deterministic [Google Cloud GFS files](https://console.cloud.google.com/marketplace/product/noaa-public/gfs)
and raw cube sphere [GEFS files](https://console.cloud.google.com/marketplace/product/noaa-public/gfs-ensemble-forecast-system).

# Support
This software is based upon work supported by the NSF National Center for Atmospheric Research, a major facility sponsored by the 
U.S. National Science Foundation  under Cooperative Agreement No. 1852977 and managed by the University Corporation for Atmospheric Research. Any opinions, findings and conclusions or recommendations 
expressed in this material do not necessarily reflect the views of NSF. Additional support for development was provided by 
The NSF AI Institute for Research on Trustworthy AI for Weather, Climate, and Coastal Oceanography (AI2ES)  with grant
number RISE-2019758 and by Schmidt Sciences, LLC. 
