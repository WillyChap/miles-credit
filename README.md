# MILES-CREDIT — CAMulator climate inference

This repository is a fork of **MILES CREDIT** (Community Research Earth Digital
Intelligence Twin), NSF NCAR's framework for training and running AI atmospheric
models. It is configured to run **CAMulator** — the CREDIT AI atmosphere — as a
climate-length rollout that produces NetCDF output.

Two parts:

- **`credit/`** — the installable `miles-credit` Python package (models,
  datasets, transforms, conservation post-blocks). Install it with
  `pip install -e .` from this directory.
- **`climate/`** — the CAMulator inference toolbox: a single YAML driver plus a
  handful of scripts that roll a trained checkpoint forward and write NetCDF.
  **This is the main entry point — see [`climate/README.md`](./climate/README.md).**

## Quick start

```bash
# install the package + its environment (once)
conda env create -f environment.yml -n camulator
conda activate camulator
pip install -e . --no-deps      # --no-deps: environment.yml already pins the runtime stack

# run CAMulator inference (full instructions in climate/README.md)
cd climate
# NOTE: the public HuggingFace asset repo is not published yet, so this download
# is not yet runnable off-NCAR — see climate/README.md "Getting the assets".
python download_assets.py --repo_id <org>/camulator   # or ./stage_assets.sh on NCAR
python check_setup.py                                  # preflight
bash RunQuickClimate.sh                                # rollout -> output/<run>/<init>/pred_*.nc
```

Requirements: a CUDA GPU (~6 GB free) and ~6.5 GB of model/forcing assets. The
environment (`environment.yml`) is PyTorch + CREDIT only — no JAX. It was built
and tested on NCAR Casper (Python 3.11, PyTorch 2.6 + CUDA 12.4); adjust the
torch wheel for a different host CUDA.

## What's here

| Path | What it is |
|------|------------|
| `climate/` | CAMulator inference toolbox (start here) |
| `credit/` | the `miles-credit` framework package (models, trainers, datasets) |
| `config/` | example CREDIT training/inference configs |
| `applications/` | CREDIT CLI entry points (train, rollout, metrics) |
| `tests/` | pytest suite for the framework |

## Documentation

- [`climate/README.md`](./climate/README.md) — the inference workflow, config,
  and asset manifest.
- [`climate/HF_READINESS.md`](./climate/HF_READINESS.md) — notes on packaging the
  toolbox + assets for distribution (e.g. HuggingFace).
- `config/README.md` — every CREDIT config key, for the underlying framework.

## License

See [`LICENSE`](./LICENSE). CREDIT is developed by NSF NCAR.
