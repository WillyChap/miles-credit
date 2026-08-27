# CAMulator on CREDIT gen2 — fine-tuning and rollout

Companion to `camulator_gen2.yml`. This is the gen2 (v2026.2) equivalent of the gen1
`camulator_config.yml`, and it is written to be compatible with the existing CAMulator
checkpoints — the same `checkpoint.pt000NN.pt` files, the same normalization statistics,
the same conservation fixers.

Read the four landmines first. Each one fails silently.

---

## 1. Four things that fail silently

### 1.1 Surface pressure is normalized by a 2-D field, not a scalar

`PS` in `mean_6h_*_f32coords.nc` has dims `('latitude', 'longitude')` — a 192x288 field. It is
the **only** 2-D variable that does; every other one is a scalar. Any scaling path that stores
one value per channel replaces that field with its global mean (~96495.9 Pa). Nothing errors;
the model just no longer sees the inputs it was trained on, and PS feeds the mass, water and
energy fixers, so the damage propagates everywhere.

The upstream bridgescaler JSON has exactly that collapse baked in:

```
ERA5/prognostic/2d/PS    mean_x_  size=1   first=[96495.89]     # <- the field, collapsed
ERA5/prognostic/3d/T     mean_x_  size=32                       # <- per-level, fine
```

**bridgescaler can represent the field** — it just was not asked to. `BridgeScalerTransform`
(preblock and postblock) takes `spatial_variables`, which stores one statistic *per gridpoint*
instead of per channel; `_flatten_spatial_tensors` folds `(B, 1, 1, H, W)` into `(B, H*W)` so
the scaler applies cell by cell. The converter that builds the JSON from NetCDF statistics
handled only 0-D and 1-D stats, so a 2-D field fell through both branches — fixed on the PR
branch, and the scaler shipped with this config stores PS as 55296 (= 192x288) values.

Verified on a real timestep from the training zarr, in **both** directions:

- **forward** (preblock): all 27 variables, PS included, come out **bitwise identical** to the
  gen1 `Normalize_ERA5_and_Forcing` path (`max|diff| = 0.0`). A collapsed scaler gives PS a
  normalized spatial std of 11.39 where it should be 1.00.
- **inverse** (postblock): PS inverse-scales to `x*std2d + mean2d` with `max|diff| = 0.0` Pa,
  and physical → normalized → physical round-trips to within **2e-6 of field RMS** for all 23
  output variables — float32 rounding. A collapsed scaler would be off by up to **42 kPa**.

Do not read the pointwise *relative* error on cloud fractions as a problem: those fields sit
near zero, so a float32-epsilon absolute error (~6e-8) divides out to a few percent. Relative
to field RMS every variable is at 1e-6.

Two rules if you ever rebuild that JSON:

- **Convert, don't refit.** `credit preprocess` fits new statistics from data. They will not
  match `mean_6h_*.nc`, so the shipped checkpoint would be running under a normalization it
  was never trained on. Convert the existing NetCDF stats instead.
- **Check PS's stored length is 55296, not 1.** `check_camulator_gen2.py` does this for you,
  and also verifies `spatial_variables` names PS in both the pre- and postblock.

`era5_normalizer` is the alternative — it reads the NetCDF statistics directly and needs no
JSON. It also handles the 2-D field, but incidentally: via an implicit broadcast in a branch
written for per-level stats, with nothing in the code saying PS is a field. It has no inverse
postblock either, so the gen2 postblock chain cannot use it.

### 1.2 SOLIN is an input-only forcing

CAMulator does not predict TOA downwelling solar. `SOLIN` is a dynamic forcing, present in
the 136-channel input and absent from the 147-channel output. The up/down energy fixer
must read it from the **input** tensor:

```yaml
TOA_forcing_solar_ind: 132   # SOLIN, input tensor
```

Channel 132 is a consequence of `static_first: True`, which orders the input-only block as
`static, then dynamic forcing`:

```
... PS 128, TREFHT 129, z_norm 130, LANDM_COSLAT 131, SOLIN 132, SST 133, ICEFRAC 134, co2vmr_3d 135
```

Set `static_first: False` and SOLIN moves to 130 — which is `z_norm`, terrain height. The
parser does not recompute this index (there is no `TOA_forcing_solar_name` in the data
variable lists to derive it from), so nothing catches the change. The energy fixer would
close the global budget against terrain height.

Verify after any change to the variable lists:

```bash
python - <<'PY'
import yaml
from credit.parser import credit_main_parser
conf = credit_main_parser(yaml.safe_load(open('camulator_gen2.yml')),
                          parse_training=False, parse_predict=True)
pc = conf['model']['post_conf']
ud = pc['global_energy_fixer_updown']
print('SOLIN ->', pc['varname_input'][ud['TOA_forcing_solar_ind']])   # must print SOLIN
for k in ['TOA_up_solar_ind','TOA_up_OLR_ind','surf_down_solar_ind','surf_up_solar_ind',
          'surf_down_LW_ind','surf_up_LW_ind','surf_SH_ind','surf_LH_ind']:
    print(f'{k:22s}', pc['varname_output'][ud[k]])
PY
```

Expected: `SOLIN`, then `FSUTOA FLUT FSDS_J FSUS FLDS_J FLUS SHFLX LHFLX`.

### 1.3 The gen2 trainer silently deletes the gen1 conservation fixers

`credit/applications/train_gen2.py` (the `gen2` / `era5-gen2` trainer) bypasses
`credit_main_parser` and then does this:

```python
if "post_conf" in conf["model"]:
    warnings.warn("Gen 2 training does not support Gen 1 postblocks ... will be ignored")
    conf["model"].pop("post_conf", None)
```

The fixers are removed before the model is built — and until recently you could not even see
it happen: `train_gen2.py` calls `warnings.filterwarnings("ignore")` at import, so its own
`warnings.warn` never reached the log. It is a logger call now, but the behaviour is unchanged.
Independently confirmed: a gen2 run produces **zero** `... registered` lines from
`credit.postblock.gen1`, where the gen1 run produces four. Its replacement is the name-based
`postblocks:` pipeline — whose only inverse-scaling block is bridgescaler, which cannot
represent the 2-D PS statistics from §1.1. So the gen2 trainer cannot currently fine-tune
CAMulator under the physics constraint it was trained with.

`camulator_gen2.yml` therefore uses **`trainer.type: era5-gen1`**, which routes to
`credit/applications/train_gen1.py`. That path runs `credit_main_parser`, honors
`model.post_conf`, and applies the fixers. It is still the gen2 codebase — only the trainer
and the dataset layer are the gen1 ones, and the config carries the flat `data:` lists they
need. Switching to the gen2 trainer is a one-line change once the 2-D PS problem is solved.

### 1.4 `activate_outside_model` decides whether the fixers run at all

This flag flips *where* a conservation fixer runs, and the two settings are for two
different jobs:

| value | who applies the fixer | use for |
|---|---|---|
| `False` | `PostBlock` inside `model.forward()`, inside the autograd graph | **training / fine-tuning** |
| `True`  | `climate/Model_State.py::_apply_postprocessing`, after the forward pass | **Quick_Climate rollout** |

`credit/postblock/gen1.py::PostBlock.__init__` skips registering any fixer whose
`activate_outside_model` is `True`. So fine-tuning with the shipped rollout settings trains
the model with the fixers **off** — no error, no warning, just a different physics
constraint than the one you meant. `camulator_gen2.yml` ships with `True` (rollout
defaults); flip the four fixers to `False` before you train.

This gate is why the config keeps both `trainer.mode` (gen1) and `trainer.parallelism` (gen2):
the two trainers read different keys, and leaving both in place makes the switch a one-liner.

---

## 2. Energy: use the up/down fixer, only the up/down fixer

CAMulator is trained under `global_energy_fixer_updown`, which forms the TOA and surface
imbalances from explicit up/down fluxes:

```
R_T = (SOLIN - FSUTOA - FLUT) / N_seconds
F_S = (FSDS_J - FSUS + FLDS_J - FLUS - SHFLX - LHFLX) / N_seconds
```

The older `global_energy_fixer` enforces the *same* budget from pre-computed net fluxes, and
both correct it by rescaling `T`. Running both double-corrects temperature. Keep:

```yaml
global_energy_fixer:
    activate: False
global_energy_fixer_updown:
    activate: True
```

`climate/Model_State.py` now raises if it finds both active, rather than quietly
double-applying.

---

## 3. Environment

```bash
module load conda
conda activate /glade/work/wchapman/conda-envs/credit-casper-mar2026

# use the gen2 tree
export PYTHONPATH=/glade/work/wchapman/Roman_Coupling/credit_gen_02
python -c "import credit; print(credit.__file__)"   # must print .../credit_gen_02/credit/__init__.py
```

`PYTHONPATH` shadows whatever `credit` the conda env has installed editable (on this machine
that is a different checkout). Everything below runs from this repository; the configs, the
checker and the code fixes all live here and nothing reaches back into another tree. Never mix
two `credit` trees in one process.

**bridgescaler >= 0.8 is required** for anything touching a BridgeScaler JSON —
`distributed_tensor`, `load_scaler_dict` and `save_scaler_dict` do not exist in 0.7.

`credit-casper-mar2026` was upgraded 0.7.1 → **0.8.2** on 2026-08-26 (`pip install
bridgescaler==0.8.2`; nothing else changed — the dry run installed that package alone).
That took the CREDIT test suite from 83 failures to 9, and the 9 that remain are all missing
optional dependencies (`s3fs`, `obstore`) plus one hard-coded PBS project code — no code
failures. `credit-main-casper` already had 0.8.2.

`s3fs` and `obstore` were deliberately **not** installed: they pull `fsspec` from 2025.3.2 to
2026.7.0 and `aiohttp` 3.11 → 3.14, and `fsspec` is the storage layer under `zarr 3.0.6`.
They are only needed for HRRR remote datasets and downloads, which this work never touches.

The upgrade also made `tests/test_postblock.py` and `tests/test_preblock.py` collectable
again for the first time, which is how two narrowed config contracts in the ported
`GlobalEnergyFixerUpDown` were caught. Worth remembering: a module that cannot be imported
reports as an error, not a failure, and is easy to read past.

### Which gen1 tree is the baseline

Three CREDIT checkouts are installed editable into different conda envs on this machine, and
they do **not** agree on the up/down energy fixer:

| conda env | `credit` resolves to | up/down fixer |
|---|---|---|
| `credit-casper-mar2026` | `Roman_Coupling/train_johns/credit` | correct — SOLIN from the input tensor |
| `credit-casper-modern`  | `Roman_Coupling/miles-jonah/credit` | **stale** — reads SOLIN from `y_pred` |
| (PYTHONPATH override)   | `Roman_Coupling/credit_gen_02/credit` | correct, after this branch's patch |

`RunQuickClimate.sh` still names `credit-casper-modern`. That tree's parser has no literal
`*_ind` fallback, so running the current `camulator_config.yml` under it raises
`KeyError: 'TOA_down_solar_name'` — it fails loudly rather than silently, but the env line in
that script is stale. Check which tree you are on before trusting a rollout:

```bash
python -c "import credit; print(credit.__file__)"
```

Note that `Model_State.py` is imported from the *current directory*, not from the `credit`
package, so the two can and do come from different checkouts.

Interactive GPU node:

```bash
qsub -I -A NAML0001 -l select=1:ncpus=32:ngpus=1:mem=250GB -l walltime=8:00:00 \
     -q casper -l gpu_type=h100
```

---

## 4. Fine-tuning recipe

### Step 1 — copy the config

```bash
cd /glade/work/wchapman/Roman_Coupling/credit_gen_02
cp climate/camulator_gen2.yml climate/camulator_gen2_finetune.yml
```

The two CAMulator configs live in different places, each next to the thing that consumes it:

| config | path | trainer |
|---|---|---|
| gen1-compatible (verified) | `climate/camulator_gen2.yml` | `era5-gen1`, and `Quick_Climate.py` for rollout |
| gen2-native | `config/gen_2/camulator/camulator_gen2_native.yml` | `gen2`, fixers as postblocks |

### Step 2 — set the run directory and stage the checkpoint

`load_weights: True` makes the trainer look for a checkpoint inside `save_loc`. Point
`save_loc` at a fresh directory and put the checkpoint you are continuing from in it:

```yaml
save_loc: '/glade/derecho/scratch/wchapman/CREDIT_runs/camulator_gen2_finetune/'
```

```bash
FT=/glade/derecho/scratch/wchapman/CREDIT_runs/camulator_gen2_finetune
mkdir -p $FT
cp /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/checkpoint.pt00069.pt \
   $FT/checkpoint.pt
```

Copy — do not move, and do not symlink. The trainer *writes* `checkpoint.pt`, so a symlink
would overwrite `checkpoint.pt00069.pt` through it. `checkpoint.pt` is the exact filename
`load_model_states_and_optimizer` looks for.

Budget the space: each saved checkpoint is ~10.5 GB (model + optimizer + scheduler + scaler),
and with `save_every_epoch: True` plus `save_backup_weights` and `save_best_weights` you get
several.

### Step 3 — turn the fixers on for training

In `camulator_gen2_finetune.yml`, set `activate_outside_model: False` on all four:

```yaml
tracer_fixer:               { activate_outside_model: False }
global_mass_fixer:          { activate_outside_model: False }
global_water_fixer:         { activate_outside_model: False }
global_energy_fixer_updown: { activate_outside_model: False }
```

(Edit the existing keys in place; the one-line form above is just shorthand.)

Confirm at startup that the log contains `GlobalMassFixer registered`,
`GlobalWaterFixer registered` and `GlobalEnergyFixerUpDown registered`. If those lines are
absent, the fixers are off and you are training a different model than you think.

### Step 4 — trainer settings

```yaml
trainer:
    load_weights: True
    load_optimizer: False       # new fine-tune from a pretrained checkpoint
    load_scaler: False
    load_scheduler: False
    reload_epoch: False         # start the epoch counter at 0

    learning_rate: 1.0e-06      # 100x below the 1.0e-04 pretraining LR
    start_epoch: 0
    num_epoch: 5
```

Set `load_optimizer: True` and `reload_epoch: True` only when *resuming* an interrupted
fine-tune, not when starting one from a pretrained checkpoint.

### Step 5 — leave `forecast_len` alone

`forecast_len` is interpreted by the *trainer*, and the two generations index it differently:
`era5-gen1` treats `0` as a single step, `gen2` treats `1` as a single step. Since this config
uses `era5-gen1`, the gen1 value **2** is correct and gives the 3-step rollout the checkpoint
was trained with. Setting it to `3` loads a 4-step rollout instead — verified: the log then
reads `forecast length = 4`. Only bump it if you also switch `trainer.type` to `gen2`.

### Step 6 — choose the fine-tuning window

Narrow `data.start_datetime` / `end_datetime` and `validation_data` to the period you are
targeting. The full 1980-2012 training window is for pretraining.

### Step 7 — run

```bash
python -m credit.applications.train_gen1 -c camulator_gen2_finetune.yml
```

Upstream, `credit/cli/_submit.py::_train` dispatched to `train_gen2` *unconditionally*,
regardless of `trainer.type` — so `credit train` silently routed you to the trainer that
deletes your fixers. The PR branch fixes that (`era5` / `era5-gen1` now go to `train_gen1`), so
on this branch `credit train -c <config>` is also safe. Invoking `train_gen1` directly works
everywhere and does not depend on that fix, which is why it is the command given here.

Multi-GPU: `trainer.mode: ddp` under `torchrun`. Single-process: `trainer.mode: none`.

Do **not** set `trainer.type: gen2` for CAMulator — see §1.3.

Expected startup log — check for all four:

```
INFO:credit.postblock.gen1:TracerFixer registered
INFO:credit.postblock.gen1:GlobalMassFixer registered
INFO:credit.postblock.gen1:GlobalWaterFixer registered
INFO:credit.postblock.gen1:GlobalEnergyFixerUpDown registered
INFO:root:Loading model state only from <save_loc>
INFO:credit.models.checkpoint:All keys matched successfully
```

and, once stepping, the up/down fixer's own diagnostic — which only prints from inside
`PostBlock`, so it is proof the fixer ran in the forward pass:

```
EnergyFixer step 1 | residual=-6.134 W/m2 | correction dT mean=-0.0125 K | ratio 0.999949
```

Memory: batch size 1 with `activation_checkpoint: True` peaks around 30 GB, which fits a 32 GB
V100 with little headroom. Use an H100 (80 GB) for anything larger.

---

### Step 8 — train the model to need the fixers less

Enabled in `camulator_gen2_native.yml`. Each conservation fixer records the ratio it
multiplied its field by; `fixer_penalty_weight` adds
`w * mean((ratio - 1)^2 / scale^2)` to the loss, so the model is pushed to close each
budget itself rather than leaning on the fixer every step.

`fixer_penalty_scales` is the part that has to be right. Measured over 36 real training
steps on ckpt69:

| fixer | rms(ratio-1) | max&#124;ratio-1&#124; | raw term | share of an *unnormalized* penalty |
|---|---|---|---|---|
| `global_water_fixer` | 1.41e-02 | 3.73e-02 | 2.85e-04 | 99.988% |
| `global_mass_fixer` | 1.50e-04 | 3.36e-04 | 3.09e-08 | 0.011% |
| `global_energy_fixer_updown` | 4.55e-05 | 1.43e-04 | 3.46e-09 | 0.001% |

The water fixer corrects ~100x harder than mass and ~300x harder than energy, so a plain
mean over `(ratio-1)^2` is the water fixer and nothing else — mass and energy would get no
gradient at any weight you chose. Dividing each term by its own reference variance puts all
three near 1.0. Re-run with the penalty on and the shares become 45% / 42% / 13%; the
residual spread is batch-to-batch variability (the water term spans two orders across
batches), not a bias.

Note that these are **fixed reference values, not a running average**, so the incentive
persists: as the model learns to conserve, the terms fall below 1 and the penalty relaxes
on its own. Re-measure them only if you change the fixers or the model substantially.

Because a fixer at its reference scale contributes exactly 1.0, `fixer_penalty_weight` is
the penalty's contribution to the loss in absolute terms. The measured data-term loss at
ckpt69 is ~3.0e-6, so the configured `1.5e-7` makes the penalty ~5% of the loss — a
constraint, not the objective. Verified in a real run: penalty 1.60e-7 against a 3.0e-6
loss, and `train_rmse` moved 0.001758 → 0.001760.

Watch `BaseLoss.last_fixer_penalties_normalized` while training. Each entry is that fixer's
term in units of its reference, so a value above 1 means that budget is drifting worse than
ckpt69 did. `last_fixer_penalties` keeps the raw physical values. Setting a scale to `null`
drops that fixer from the penalty.

---

## 5. Rollout

The climate driver is unchanged:

```bash
python Quick_Climate.py -c camulator_gen2.yml
```

Keep `activate_outside_model: True` for rollout — `Model_State.py` applies the fixers after
the forward pass. Its startup banner now prints the up/down flag explicitly:

```
Conservation fixers: Mass=True, Water=True, Energy=False, EnergyUpDown=True
Tracer fixer: True
```

`Energy=False, EnergyUpDown=True` is the correct configuration. `EnergyUpDown=False` means
the energy constraint is off.

`predict.timesteps_fast_climate` sets the rollout length in 6-hour steps (1460 = 1 no-leap
year). `predict.init_cond_fast_climate` is the initial-condition tensor.

Note: `credit/output.py` tests for the *presence* of `predict.climate_rescale_output`, not
its value. Setting it `False` still inverse-transforms the output; delete the key to disable.

---

## 6. What the config carries that stock CREDIT does not

| item | where | why it matters |
|---|---|---|
| `global_energy_fixer_updown` reading SOLIN from the input tensor | `credit/postblock/gen1.py` | upstream read it from `y_pred`, where SOLIN does not exist |
| `TOA_forcing_solar_ind` literal index support | `credit/parser.py` | input-only variables cannot be resolved from `varname_output` |
| null handling in `tracer_fixer.tracer_thres_max` | `credit/parser.py` | `null` = no upper cap; upstream raised `TypeError` |
| gen1 postblock binding in `Model_State.py` | `climate/Model_State.py` | gen2 re-pointed `credit.postblock` at the gen2 classes; the old import disabled *all* fixers |
| `postprocessing.wind_artifact_filter` | `climate/WindPP.py` | fork-only block; anisotropic smoothing + amplitude preservation |
| per-gridpoint PS in the BridgeScaler JSON | `credit/cli/_convert.py` + `preblocks:`/`postblocks:` | the converter dropped 2-D statistics; PS now stores 55296 values instead of 1 |
| lat/lon float32 cast on the forcing dataset | `climate/Model_State.py` | without it xarray collapses latitude 192 → 2 and the rollout dies at step 1 |
| anisotropic wind filter | `climate/WindPP.py` | gen2 shipped the isotropic version, which ignores the config's zonal/meridional sigmas |
| float64 accumulation in `weighted_sum` | `credit/physics_core.py` | float32 summation of ~55k cells biased the fixer corrections by 15% (ΔT) and 36% (ΔPRECT) per step |
| per-variable `TracerFixer` denorm | `credit/postblock/gen1.py` | the old whole-tensor `inverse_transform` round-trip perturbed PS and 90% of Qtot |
| `fixer_penalty_scales` | `credit/losses/base.py` | without per-fixer normalization the penalty is the water fixer alone; mass and energy get no gradient at any weight |

All of these are staged as an upstream PR on branch
`fix/climate-gen1-postblock-imports-updown-fixer` in
`/glade/work/wchapman/Roman_Coupling/credit_gen_02`.

**Do not delete the `postprocessing.wind_artifact_filter` block to disable the filter.**
`WindPP.load_wind_filter_config` defaults `activate=True`, so removing the block re-enables
the filter with different built-in tuning (`speed_threshold` 3.019 instead of 2.8, isotropic
smoothing, no amplitude preservation). Set `activate: False` explicitly. WindPP also swallows
all exceptions, so a misconfiguration there fails silently.

---

## 7. What was actually tested

Both halves were run end to end on a V100, from `checkpoint.pt00069.pt`.

**Inference vs the gen1 stack** — same checkpoint, same initial condition, fixers on in both:

- checkpoint loads with `strict=True`: 2074/2074 tensors, 0 missing, 0 unexpected, identical
  post-load weight checksum
- the **raw model output is bitwise identical** across all 23 variable groups (Qtot: 0 of
  1,769,472 points differ)
- the post-processed output agrees to at most **2 float32 ULP** on every variable, including
  all eight flux diagnostics; PS agrees to exactly one bit in Pa
- the conservation corrections reproduce gen1 exactly:

  | correction | gen1 | gen2 |
  |---|---|---|
  | energy up/down, global mean ΔT | −0.010778 K | −0.010778 K |
  | mass fixer, mean ΔPS | −25.234682 Pa | −25.234665 Pa |
  | water fixer, mean ΔPRECT | −1.29299e-05 | −1.29299e-05 |

**Fine-tuning** — one epoch, 4 train + 2 valid batches, exit code 0:

- all four fixers registered inside `model.forward()`, and the up/down fixer's step diagnostic
  printed from inside `PostBlock` — proof it ran in the forward pass
- 2074 tensors / 875.5 M parameters loaded, `All keys matched successfully`
- losses 0.10134, 0.09004, 0.10419, 0.09016; `train_loss 0.0964`, `valid_loss 0.1061`
- a full checkpoint (model + optimizer + scheduler + scaler) was written

The losses do not fall monotonically over four steps and should not be expected to: at
`lr: 1.0e-06` on a converged checkpoint, four batch-size-1 samples are dominated by
sample-to-sample variance. What is verified is that the gradient path is live end to end.

**Config fidelity vs `camulator_config.yml`** — compared leaf by leaf:

- `model.post_conf`: 77 substantive leaves, **all identical**. The only 16 differences are
  `requires_scaling` and the `scaling_coefs` block, which are dead keys — no reference to
  either exists anywhere in `credit/`.
- `model` architecture: identical, except `pad_lat`/`pad_lon` written `[48, 48]` instead of the
  bare `48` the parser expands to the same thing.
- `postprocessing.wind_artifact_filter`: identical.
- `loss`: identical.
- every `data` variable list, `static_first`, `scaler_type`, `mean_path`, `std_path`,
  `lead_time_periods`, `history_len`, `forecast_len`, `dataset_type`: identical. `data.levels`
  is stated explicitly as `32`, where the gen1 config left the parser to inject it from
  `model.levels` — same value.

**Preblock chain** (the gen2 path, not the one your runs use): `era5_normalizer` → `concat`
puts SOLIN at channel **132**, verified bit-exact against the normalizer's own per-variable
output, with normalized PS at spatial mean −0.064 / std 0.959. Separately, the
`bridgescaler_transform` chain now shipped in the config was checked against
`Normalize_ERA5_and_Forcing` on real zarr data: **27/27 variables bitwise identical**, PS
included.

One caveat worth knowing: gen2's channel order comes from a hardcoded `FIELD_TYPE_RANK`
in `credit/datasets/gen_2/channel_utils.py`, not from `data.static_first` (gen2 never reads
that key). It happens to equal `static_first: True`. The docstring in
`credit/preblock/concat.py` claims "prognostic < dynamic_forcing < static", which would put
SOLIN at 130 — that docstring is wrong; the constant is authoritative.

---

## 8. Before you switch to gen2's conservation fixers: the Qtot decision

Gen2 has its own name-based fixers in `credit/postblock/conservation.py`, and moving to
`trainer.type: gen2` means using them instead of `model.post_conf`. They were compared against
the gen1 ones on real consecutive timesteps:

| fixer | verdict |
|---|---|
| `GlobalMassFixer` | interchangeable — ΔPS agrees to 1 float32 ULP |
| `GlobalWaterFixer` | interchangeable — ΔPRECT relative diff 4–5e-8 |
| `GlobalEnergyFixerUpDown` | interchangeable — mean ΔT relative diff 1e-9…1e-7 |
| `TracerFixer` | **different, in one place: Qtot** |

**gen1 never clips Qtot.** `parser.py` emits `tracer_var_names` as a flat per-channel list, so
`Qtot` repeats 32×, which routes `gen1.py:234-238` into a branch commented *"scalar stats
assumed"* that uses **level 0's** mean/std for all 32 channels. Qtot statistics are strongly
per-level, so the effective physical threshold is negative and grows with height:

```
lev  0: +0.000e+00      lev 25: -4.129e-02
lev 15: -6.485e-04      lev 31: -5.982e-02 kg/kg      (gen2 clips at 0.0 everywhere)
```

Measured on the ckpt69 rollout, 23,003,136 Qtot points: **0** fall below gen1's threshold, so it
truly never fires — while **1.71% of points are negative** and nothing removes them. The model
routinely produces negative moisture, and the mass and water fixers both consume Qtot.

**For a like-for-like migration, omit Qtot from gen2's `tracer_vars`.** That reproduces gen1
exactly — not approximately — and makes all four fixers interchangeable. gen2 clamps with a
scalar min per variable, so it cannot express gen1's level-dependent threshold anyway.

**Do not fold "start clipping Qtot at 0" into the migration.** With negatives present it flips
the sign of ΔPS and ΔPRECT and changes ΔT by ~7×. It is a real physics change to a constraint
the model was trained under, and deserves its own before/after rather than being confounded
with a framework switch.

## 9. If the validation metrics look absurd

`credit/metrics/base.py` scores `y_processed` against `y_target_processed`, and the trainer
builds the latter by running the **same postblock chain** on the flat target `y`. So the chain
needs four blocks, not two:

```yaml
postblocks:
  per_step:
    reconstruct:        {type: reconstruct, args: {detach: false}}
    scaler:             {type: bridgescaler_transform, args: {..., method: inverse_transform}}
    reconstruct_target: {type: reconstruct, args: {in_key: "y", out_key: "y_target_processed"}}
    scaler_target:      {type: bridgescaler_transform, args: {..., method: inverse_transform,
                                                              key: "y_target_processed"}}
```

Supply only the prediction half and the metrics compare **physical predictions against
normalized targets**. The symptom is distinctive, and misleading: the loss looks completely
normal, because it is computed elsewhere, while the metrics blow up —

```
Epoch 0 valid_loss: 0.099000
Epoch 0 valid metrics  valid_rmse: 1166044.19  valid_r2score: -6.67e12  valid_bias: 760712.03
```

A `valid_bias` larger than surface pressure itself is the tell. Both scaler blocks also need
`spatial_variables`, or the *target's* PS gets collapsed — the same bug as §1.1, but on the
half nobody looks at. `check_camulator_gen2.py` verifies both chains are complete.

## 10. Pre-flight check

`check_camulator_gen2.py` runs every check below and exits non-zero on failure:

```bash
python check_camulator_gen2.py camulator_gen2.yml            # rollout settings
python check_camulator_gen2.py camulator_gen2_finetune.yml --train   # training settings
```

It verifies the channel map (SOLIN at the right input channel, all eight flux indices), that
the up/down fixer is on and the net-flux one off, that PS statistics are still a 2-D field and
the preblock is not bridgescaler, that all four fixers actually construct, that every
referenced file exists, and that `activate_outside_model` matches the mode you asked for. In
`--train` mode it also checks `trainer.type`, `dataset_type`, `thread_workers`, `forecast_len`
and that the checkpoint is staged where the trainer will look for it.

### The same checks, by hand

- [ ] `python -c "import credit; print(credit.__file__)"` points at `credit_gen_02`
- [ ] `SOLIN` resolves to input channel 132 (§1.2 snippet)
- [ ] the eight flux indices resolve to `FSUTOA FLUT FSDS_J FSUS FLDS_J FLUS SHFLX LHFLX`
- [ ] `global_energy_fixer.activate: False`, `global_energy_fixer_updown.activate: True`
- [ ] `static_first: True`
- [ ] if the preblock is `bridgescaler_transform`, PS is in `spatial_variables` and its stored scaler holds 55296 values
- [ ] training: `trainer.type: era5-gen1` (NOT `gen2` — it deletes `post_conf`)
- [ ] training: launched via `python -m credit.applications.train_gen1`, not `credit train`
- [ ] training: `data.dataset_type: ERA5_MultiStep_Batcher` present at the top level of `data:`
- [ ] training: `thread_workers: 1` and `valid_thread_workers: >= 1`
- [ ] training: `forecast_len: 2` (gen1 indexing), not 3
- [ ] training: `activate_outside_model: False` and the `... registered` lines appear in the log
- [ ] rollout: `activate_outside_model: True` and the banner shows `EnergyUpDown=True`
