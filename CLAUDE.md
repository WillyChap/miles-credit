# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a **fork of [NCAR/miles-credit](https://github.com/ncar/miles-credit)** (the CREDIT AI
atmospheric prediction framework), customized for the **CAMulator** project — an AI emulator of the
CESM/CAM6 climate model on a 1.0° / 32-sigma-level grid, run under the "Roman Coupling" effort.
The package name is still `miles-credit` and the importable package is `credit`.

Because it is a fork, the physics/conservation code here has **diverged from upstream**. When
reasoning about the conservation post-blocks, compare against the upstream reference:
- https://github.com/NCAR/miles-credit/blob/main/credit/postblock/gen1.py (upstream lives in `gen1.py`;
  here the equivalent classes live in `credit/postblock/_postblock.py`).
The most important local divergences are the **float64 budget accumulation** in `physics_core.py`
and the **conservation-loss training coupling** (see "Conservation post-blocks" below).

The two main task docs written for this fork — read them before touching training or physics:
- `TRAINING_GUIDE.md` — end-to-end CAMulator training on NCAR HPC (Casper/Derecho), config knobs, common failures.
- `WaterFixer_Feedback_Bug.md` — the canonical write-up of the conservation-loss bug and its fix (Option 3).
- `Running_Guide.md` — running long climate rollouts from a trained checkpoint via `climate/`.

## Commands

Editable install (do this once per conda env; `--no-deps` when the env is already populated):
```bash
pip install -e .            # full deps
pip install -e . --no-deps  # when cloning an existing credit-* env
```

Lint (ruff, line-length 120, google docstring convention; config in `pyproject.toml`):
```bash
ruff check                  # lint
ruff format                 # format
```

Tests (pytest; tests live in `tests/`):
```bash
pytest                                  # full suite (what CI runs)
pytest tests/test_postblock.py          # one file — conservation fixers
pytest tests/test_physics.py            # physics_core budget math
pytest tests/test_trainer_components.py # trainer unit pieces
pytest tests/test_postblock.py -k water -q   # single test by name
```
CI (`.github/workflows/python-package-conda.yml`) installs CPU-only torch, runs `ruff check` (non-blocking,
`--exit-zero`) then `pytest`. Postblock/physics tests use a `simple_demo` config path with synthetic
lat/lon/levels so they run without data files.

### CLI

The unified entrypoint is `credit` (`credit/cli.py` → `console_scripts` in `pyproject.toml`):
```bash
credit train -c config.yml
credit submit --cluster casper  -c config.yml --dry-run   # generate + preview PBS script
credit submit --cluster derecho -c config.yml             # submit to PBS
credit rollout  -c config.yml
credit init --grid 0.25deg -o my_config.yml
```
There are **v2** and **legacy v1** entry points (`credit_train_v2`, `credit_train`, etc.). New work uses v2.
Direct single-GPU debugging bypasses the CLI: `CUDA_VISIBLE_DEVICES=0 python applications/train.py -c config.yml`.

## Architecture

Everything is **config-driven**: a single YAML (see `config/example-v2026.1.0.yml`, `climate/camulator_config.yml`)
declares data sources, variable→channel layout, model, trainer, loss, and the `model.post_conf` physics block.
`credit/parser.py` validates/normalizes the config; most modules take the parsed `conf` dict and read their
own sub-keys. Channel **indices** (e.g. `precip_ind`, `q_inds`, `sp_inds`, `T_inds`) are how physics code
locates variables inside the stacked `(batch, var, time, lat, lon)` tensors — get these wrong and the budget
math silently corrupts.

The forward pipeline for one step is **preblock → model → postblock**:
- `credit/preblock/` — input transforms applied before the network (`norm.py` normalization,
  `concat.py` channel concatenation, `regrid.py`). Registered/assembled per config.
- `credit/models/` — network architectures (CAMulator uses the Crossformer/wxformer-family model).
- `credit/postblock/_postblock.py` — `PostBlock` dispatches the conservation fixers (see below).
- `credit/datasets/` — zarr/ERA5 loaders. `era5_multistep_batcher.py` produces multi-step
  (rollout) batches; `multi_source.py` merges multiple data sources. `load_dataset_and_dataloader.py`
  is the factory. Note: source lookups must not be hardcoded to `source['ERA5']` (recently fixed — see git log).
- `credit/trainers/` — one trainer class per regime. `trainerERA5v2.py` is the current CAMulator trainer;
  `base_trainer.py` owns the epoch loop, EMA, and checkpointing. `preflight.py` runs pre-training
  memory/hang sanity checks. `ic_optimization.py` does initial-condition optimization.
- `credit/losses/` — `weighted_loss.py` is the main latitude-weighted loss; `__init__.py` is the loss factory.
- `climate/` — inference/rollout tooling for long climate runs: `Make_Climate_Initial_Conditions.py`,
  `Quick_Climate.py` (rollout), `Post_Process.py` (aggregate 6-hourly → daily/monthly), `Model_State.py`.

Training is multi-step: the model is rolled out `N` steps and loss is taken on the steps in
`backprop_on_timestep`. Postblocks can run "outside the model" (`activate_outside_model: True`) so
they correct `y_pred` between rollout steps.

## Conservation post-blocks and budget calculations (read carefully)

`credit/postblock/_postblock.py` holds the physics fixers, dispatched by `PostBlock` from
`model.post_conf`. Order matters: TracerFixer → GlobalMassFixer → GlobalWaterFixer → Energy fixer.
- `TracerFixer` — clamps tracers (Qtot, PRECT, clouds) to physical bounds (≥0).
- `GlobalMassFixer` — conserves global dry-air mass; corrects q and surface pressure.
- `GlobalWaterFixer` — closes the global water budget `dTWC/dt = -(E + P)` by scaling PRECT.
- `GlobalEnergyFixer` / `GlobalEnergyFixerUpDown` — close the global energy budget (TOA + surface
  fluxes). `...UpDown` is the local addition that uses **explicit up/down flux indices** (DSWRFtoa,
  USWRFtoa, ULWRFtoa, FSDS/FSUS/FLDS/FLUS/SHF/LHF in J/m²) instead of pre-computed *net* fluxes.

Budget math lives in `credit/physics_core.py` (`physics_pressure_level`, `physics_hybrid_sigma_level`):
`weighted_sum`, `total_column_water`, `total_dry_air_mass`, area weights, level integrals. **Local
divergence from upstream:** `weighted_sum` now accumulates in **float64** (cast back to input dtype) to
avoid a ~0.6% rounding error when summing ~55k float32 cells — this materially affects the budget
residual. Do not revert to float32 accumulation.

The WaterFixer (`GlobalWaterFixer.forward`, ~lines 690–803) computes, per step:
`dTWC/dt` (total water content tendency) → `E_src`, `P_sink` (area-weighted global sums) →
`residual = -TWC_sum - E_sum - P_sum` → `P_correct_ratio = (P_sum + residual) / P_sum`, then scales
PRECT by that ratio. It logs `drift_pct = 100*(ratio-1)` every 50 steps. Ground-truth drift is ≈0% (see
`experiments/compute_water_budget_truth.py`, which reproduces the *exact* same physics on raw zarr for validation).

### The conservation-loss coupling (the key fork-specific behavior)

Running the WaterFixer as an outside-model postblock **during training** originally created a
positive-feedback loop that made the model under-predict PRECT (drift ramped 2%→24%): loss was computed
*after* the fixer scaled PRECT up, so the gradient told the model to lower raw PRECT, which made the
fixer scale harder. Full write-up in `WaterFixer_Feedback_Bug.md`.

The fix is **water-only** — mass and energy fixers are correction-only (they rescale `y_pred` in
place and feed **no** loss term back). Only `GlobalWaterFixer` participates in the loss:
1. `GlobalWaterFixer.forward` stores `x["water_conservation_loss"] = (P_correct_ratio - 1)**2`
   (`_postblock.py:785`). `GlobalMassFixer` / `GlobalEnergyFixer*` compute their own correction ratios
   (`q_correct_ratio`, `sp_correct_ratio`, `E_correct_ratio`) but store no `*_conservation_loss` key.
2. The trainer adds `water_conservation_weight * water_conservation_loss` as a separate penalty term.
   The weight comes from `post_conf.global_water_fixer.conservation_loss_weight` (0.0 = safe no-op).

**The two trainers differ — check which you are editing:**
- `trainerERA5.py` (v1, the `WaterFixer_Feedback_Bug.md` "Option 3"): saves `y_pred_pre_water` and computes
  the supervised MSE on the **pre-correction** prediction (the `use_pre` gate, ~line 285), then adds the penalty.
- `trainerERA5v2.py` (v2, current CAMulator trainer): computes MSE on the **post-correction** `y_pred`
  (~line 279, after mass→water→energy have all run) plus the additive penalty. It additionally, on rollout
  steps **not** in `backprop_on_timestep`, back-props the water penalty alone (`conservation_only_loss`,
  `retain_graph=True`, ~lines 286–293) so the model can't violate conservation at intermediate steps while
  optimizing a later step. v2 does **not** save a pre-correction copy.

When changing this: preserve the `weight == 0` no-op path, keep the water penalty a separate additive term,
and don't assume v1's pre-correction-MSE behavior applies to v2. If you want mass/energy to constrain the
loss too, you'd need to add `*_conservation_loss` keys in their `forward` and corresponding penalty terms —
that work does not exist yet.

## Hard-won gotchas (from TRAINING_GUIDE.md §11)

- `update_learning_rate: True` bypasses warmup → NaN at step 2. Keep it `False` except when intentionally
  resetting LR for a fine-tune phase.
- `use_compile: True` is incompatible with spectral norm → NaN. Keep `False`.
- Padding must keep `(grid + 2*pad)/2` divisible by 3 — only `pad = 0, 24, 48` are valid for the 1° grid.
- Validation uses EMA weights (`decay=0.9999`), so `valid_loss >> train_loss` for the first ~16 epochs is
  EMA lag, **not** overfitting. Trust train loss / validate offline.
- bfloat16 tensors must be `.float()` before metrics or ACC reads 0.000.
- Physics post-blocks are enabled only **after** the model is well-trained (~epoch 30), one at a time.

## Notes for editing

- Many files in the repo root (`eval_*.py`, `plot_*.py`, `*.png`, `*.csv`, `surgery_*.py`, `submit_*.sh`,
  `rechunk_*.py`, scratch logs) are Will's analysis/run artifacts, not part of the package. The installable
  code is under `credit/`, `applications/`, and `climate/`. Don't treat root scripts as library API.
- Data, normalization (mean/std `.nc`), and statics files live on NCAR scratch/campaign and are shared
  read-only; configs point at `/glade/derecho/scratch/wchapman/...` and `/glade/campaign/...`. Copy, never
  edit in place.
</content>
</invoke>
