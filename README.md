# SUMO / GraphCast / CAMulator + CESM2.1.5 — porting guide

This is a customized fork of NSF NCAR MILES CREDIT that adds the **SUMO supermodel coupling layer**: CAM6 (CESM2.1.5) nudged at 6-hour intervals toward a consensus of neural-surrogate atmospheric models (GraphCast and/or CAMulator) via a file-based handshake on a shared filesystem. The upstream CREDIT documentation begins below; this section is what you need to **port the full coupled stack to a new HPC**.

## Architecture

CESM (compute partition) and the neural-surrogate server + coordinator (GPU partition) run as **independent batch jobs** that synchronize through lifecycle flag files in the CESM run directory. The Fortran barrier lives in `cam_comp.F90:sumo_barrier_after_h1` and activates only when `sumo_active.flag` is present; absent the flag, CESM runs free with zero overhead.

## What to port

| Component | How to get it |
|---|---|
| CESM2.1.5 fork (CAM6 + DATM-camulator + SST=0 fix) | `git clone -b release-cesm2.1.5-camulator git@github.com:WillyChap/CESM.git && cd CESM && ./manage_externals/checkout_externals` |
| This repo (SUMO orchestration + CAMulator + GraphCast Python) | `git clone -b camulator-sumo git@github.com:WillyChap/miles-credit.git` then `pip install -e .` inside the active conda env |
| `credit-coupling` conda env | PyTorch 2.4.1+cu121 base + this repo; used by the CAMulator server and all `climate/` scripts |
| `supermodel` conda env | JAX + PyTorch; used only by the GraphCast coordinator/server |
| GraphCast source code | Vendored at `./graphcast/` in this repo (DeepMind release, Apache 2.0) |
| GraphCast weights + stats | **NOT in git** (140 MB params file exceeds GitHub limits). Download `params/*.npz` and `stats/*.nc` from DeepMind's public bucket: https://console.cloud.google.com/storage/browser/dm_graphcast — drop into `./graphcast/params/` and `./graphcast/stats/` |
| CAMulator checkpoint (active model: `extended_v2`) | `/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/checkpoint.pt00044.pt` (3.3 GB) |
| CAMulator config referenced by the checkpoint | `/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/camulator_config_extended_v2.yml` — all sub-paths below are listed inside it |
| CAMulator normalization stats (mean / std) | `/glade/derecho/scratch/wchapman/b_credit_runs/mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc` and the matching `std_6h_…` file in the same directory |
| CAMulator static fields (statics / physics) | `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc` |
| CAMulator initial-condition tensors | `/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/init_times/init_camulator_condition_tensor_YYYY-MM-DDTHHz.pth` (and the `sumo_init_…` variant used by SUMO restart) |
| `env_mach_specific.xml` | per-machine MPI/compiler config; edit in the CESM case after `create_newcase` |

## Quick-start

1. Build a CESM case the usual way against the fork above. `USE_FTORCH` defaults to `FALSE` — **no FTorch install needed for SUMO**.
2. Activate `credit-coupling` (or `supermodel` for GraphCast) and `pip install -e .`.
3. See **[`climate/To_Run_SUMO.md`](./climate/To_Run_SUMO.md)** for the full three-mode launch protocol (camulator-only, graphcast-only, both-consensus) and the file-handshake details.
4. Drop `sumo_active.flag` into the CESM run directory before `case.submit` to arm the CAM6 barrier.

## Per-machine notes

- On Cray/MPICH systems set in `env_mach_specific.xml`: `MPICH_GPU_SUPPORT_ENABLED=0`, `FI_CXI_DISABLE_HOST_REGISTER=1`, `MPICH_SMP_SINGLE_COPY_MODE=NONE`.
- For GraphCast, configure a persistent JAX XLA compilation cache directory (warm cache ~60 s vs cold ~18 min on first run).
- CESM runs on the compute partition; the coordinator + neural-surrogate server run on a GPU node sharing the **same scratch filesystem** as CESM's run directory (lifecycle flags assume POSIX visibility within seconds across both jobs).

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
