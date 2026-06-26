---
license: apache-2.0
library_name: miles-credit
tags:
  - climate
  - weather
  - atmosphere
  - emulator
  - earth-system-model
  - pytorch
  - CAMulator
  - CREDIT
pipeline_tag: other
---

# CAMulator

**CAMulator** is an AI emulator of NSF NCAR's CAM6 atmosphere, trained and run
within the [CREDIT](https://github.com/WillyChap/miles-credit) framework. It
rolls a 1° (192×288), 32-level, 6-hourly atmospheric state forward for
climate-length simulations (years to decades), driven by prescribed SST, sea-ice,
solar, and CO₂ forcing.

> ⚠️ CAMulator is a research tool. It emulates a specific CAM6 configuration and
> is not a substitute for an operational forecast or a full Earth-system model.

### Quick links

- 📦 **Inference toolbox (code):** https://github.com/WillyChap/miles-credit (branch `camulator_huggingface`, dir `climate/`)
- 📖 **CREDIT framework:** https://github.com/NCAR/miles-credit
- 📄 **Paper:** CAMulator (Chapman et al.) — *add DOI/link*
- 🤗 **This model + data:** https://huggingface.co/willychap/camulator

### Inference quickstart

```bash
# 1. get the toolbox
git clone -b camulator_huggingface https://github.com/WillyChap/miles-credit.git camulator
cd camulator

# 2. environment (PyTorch 2.4.1 + CUDA 12.1; pinned in environment.yml)
conda env create -f environment.yml -n camulator
conda activate camulator
pip install -e . --no-deps        # --no-deps: environment.yml pins the exact stack

# 3. pull this model + its inputs into ./assets/
cd climate
python download_assets.py --repo_id willychap/camulator     # default checkpoint = epoch 65

# 4. verify + run (writes monthly-mean NetCDF by default)
python check_setup.py
bash RunQuickClimate.sh
```

Full instructions, configuration, and the asset manifest are in
[`climate/README.md`](https://github.com/WillyChap/miles-credit/blob/camulator_huggingface/climate/README.md).

### Strengths and weaknesses

**Strengths**
- **Conserves mass, water, and energy** via CREDIT's physics post-blocks (applied every step).
- **Stable multi-year / multi-decadal rollouts** with a no-leap calendar that stays aligned to the forcing.
- **Fast** — a year of 6-hourly climate runs on a single GPU in minutes.
- **Cyclic or transient forcing** — repeat a climatological year, or follow the real 1980–2014 SST/ICE/CO₂ record.
- **Multiple checkpoints** (training epochs) hosted so you can study sensitivity to training stage.

**Known limitations**
- Research use only; emulates one CAM6 configuration at 1°.
- Small near-surface temperature bias (~1–2 K) typical of the architecture.
- A single deterministic realization (no built-in ensemble).
- Skill depends on the fidelity of the prescribed forcing.

### Repository layout

```
willychap/camulator
├── README.md                          # this model card
├── inference_config.yaml              # ready-to-run config (= camulator_config.yml)
├── checkpoint.pt00065.pt              # default model (epoch 65); other epochs alongside
├── forcing_data/
│   ├── b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc      # cyclic (default)
│   └── b.e21.CREDIT_climate_branch_1980_2014.nc          # progressive/transient
├── initial_conditions/
│   └── init_camulator_condition_tensor_1981-01-01T00Z.pth
├── normalization/
│   ├── mean_*.nc, std_*.nc                                # z-score
│   └── *statics*.nc                                       # statics + latitude weights
└── metadata/
    └── era5.yaml
```

`download_assets.py` pulls these into the toolbox's `./assets/` for you.

### Training data

CAMulator was trained on a CAM6 / ERA5-scaled climate dataset (1980–2014). The
full training archive is not hosted here; the inputs needed to *run* the model
(forcing, initial conditions, normalization, statics) are.

### Citation

```bibtex
@misc{camulator,
  title  = {CAMulator: an AI emulator of the CAM6 atmosphere},
  author = {Chapman, William E. and others},
  note   = {CREDIT framework, NSF NCAR},
  year   = {2025}
}
```
*(Replace with the published reference / DOI when available.)*
