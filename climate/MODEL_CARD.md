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

CAMulator is an auto-regressive machine-learned emulator of NSF NCAR's CAM6
atmosphere, trained and run within the
[CREDIT](https://github.com/WillyChap/miles-credit) framework. Given prescribed
sea-surface temperature, sea-ice, incoming solar radiation, and CO2, it rolls a
1 degree (192x288), 32-level, 6-hourly atmospheric state forward for
climate-length simulations (years to decades). It conserves global dry-air mass,
moisture, and total atmospheric energy, remains numerically stable over decadal
rollouts, and reproduces the annual climatology together with major modes of
variability such as ENSO and the NAO -- at roughly a 350x speedup over CAM6,
making it an efficient way to generate large climate ensembles.

The model and method are described in Chapman et al. (2025), *CAMulator: Fast
Emulation of the Community Atmosphere Model*
([arXiv:2504.06007](https://arxiv.org/abs/2504.06007)).

CAMulator is a research tool. It emulates a specific CAM6 configuration and is
not a substitute for an operational forecast or a full Earth-system model.

### Quick links

- Inference toolbox (code): https://github.com/WillyChap/miles-credit (branch `camulator_huggingface`, dir `climate/`)
- CREDIT framework: https://github.com/NCAR/miles-credit
- Paper: Chapman et al. (2025), *CAMulator: Fast Emulation of the Community Atmosphere Model*, [arXiv:2504.06007](https://arxiv.org/abs/2504.06007)
- This model + data: https://huggingface.co/willychap/camulator

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

### Evaluation

We evaluate many training checkpoints by running each as a free-running,
autoregressive 35-year rollout (1980-2014, 6-hourly, no-leap) and scoring it
against the CREDIT ERA5-scaled training target on the same 1 degree grid
(latitude-weighted), looking at both the monthly-mean climatology and the
6-hourly distribution.

Checkpoint 65 is selected as the default. The other top checkpoints
(epochs 63, 70, 48, 47, 66, 76, 68, 51, 43) are provided as well, so you can
evaluate them yourself, build cheap checkpoint ensembles, or study sensitivity to
training stage — pick one with `download_assets.py --checkpoint checkpoint.pt000NN.pt`.

Checkpoint 65 climatology (latitude-weighted, full 35-yr record):

| Field | Spatiotemporal RMSE | Global-mean bias | Decadal-trend error | Global-mean monthly RMSE | Annual corr. |
|---|---|---|---|---|---|
| TREFHT | 1.564 K | +0.033 K | -0.007 K/decade | 0.159 K | 0.985 |
| PRECT  | 6.09e-4 (2.4 mm/day) | -3.9e-6 (essentially neutral) | -- | clim. RMSE 7.49e-5 | -- |

The figures below summarize the metrics: a skill scorecard across the top
checkpoints, checkpoint 65's annual-mean bias maps, the global-mean temperature
evolution against the training target, and the 6-hourly precipitation
distribution.

![Checkpoint skill scorecard (green = better)](figs/monthly_scorecard.png)

![Checkpoint 65 annual-mean bias maps](figs/monthly_ckpt65_biasmaps.png)

![Global-mean TREFHT, 1980-2014: checkpoints vs training target](figs/monthly_gmt_timeseries.png)

![6-hourly PRECT distribution](figs/extremes_pdf_PRECT.png)

> Note on PRECT units: native values are metres of liquid-water equivalent per
> 6-hourly step (ERA5 `tp` convention); mm/day = native x 4000. Checkpoint 65's
> global-mean precipitation is 2.92 mm/day vs. truth 2.93.

### Repository layout

```
willychap/camulator
├── README.md                          # this model card
├── inference_config.yaml              # ready-to-run config (= camulator_config.yml)
├── checkpoint.pt00065.pt              # default model (epoch 65); top-10 epochs alongside
├── forcing_data/
│   ├── b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc      # cyclic (default)
│   └── b.e21.CREDIT_climate_branch_1980_2014.nc          # progressive/transient
├── initial_conditions/
│   └── init_camulator_condition_tensor_*.pth            # 69 ICs (Jan 1 & Jul 1, 1980/1981-2014)
├── normalization/
│   ├── mean_*.nc, std_*.nc                               # z-score
│   └── *statics*.nc                                      # statics + latitude weights
├── metadata/
│   └── camulator_metadata.yaml                           # output variable units / long-names
└── figs/                                                 # model-card figures
```

(The inference toolbox actually reads its units from the in-repo copy
`climate/camulator_metadata.yaml`; the `metadata/` copy here is for reference.)

`download_assets.py` pulls these into the toolbox's `./assets/` for you.

### Training data

CAMulator was trained on a CAM6 / ERA5-scaled climate dataset (1980-2014). The
full training archive is not hosted here; the inputs needed to *run* the model
(forcing, initial conditions, normalization, statics) are.

If you would like access to our training Zarr datasets, please email
wchapman [at] colorado.edu.

### Citation

```bibtex
@article{chapman2025camulator,
  title   = {CAMulator: Fast Emulation of the Community Atmosphere Model},
  author  = {Chapman, William E. and Schreck, John S. and Sha, Yingkai and
             Gagne II, David John and Kimpara, Dhamma and Zanna, Laure and
             Mayer, Kirsten J. and Berner, Judith},
  journal = {arXiv preprint arXiv:2504.06007},
  year    = {2025},
  doi     = {10.48550/arXiv.2504.06007},
  url     = {https://arxiv.org/abs/2504.06007}
}
```
