# WaterFixer Feedback Loop Bug

## Summary

Applying `GlobalWaterFixer` as an **outside-model postblock during training** causes a
positive-feedback loop that drives the model to progressively under-predict PRECT.
The drift_pct metric monotonically worsens from ~2% to ~24% over epochs 4–15 and
then stalls — the model learns a degenerate solution, not a balanced one.

---

## How to Reproduce

### 1. Ground-truth water budget (what balanced looks like)

```bash
cd /glade/work/wchapman/Roman_Coupling/train_johns

python experiments/compute_water_budget_truth.py --years 1980 --out water_budget_1980.csv
```

Script: `train_johns/experiments/compute_water_budget_truth.py`

Computes dTWC/dt, E_src, P_sink, residual, drift_pct at every 6-hourly timestep
in one year of CESM training data using the **exact same physics** as
`GlobalWaterFixer` (sigma-level TWC integral, same area weights, same N_seconds=21600).

**Result**: truth drift_pct ≈ **+0.024% ± 2.2%** — essentially zero mean, no bias.

---

### 2. Parse training logs and compare to truth

```bash
python plot_drift_comparison.py
# → drift_comparison.png
```

Script: `train_johns/plot_drift_comparison.py`

Parses all `camulator_ext_v2_finetune.o*` log files, extracts WaterFixer diagnostics
(averages duplicate 4-GPU entries at same step), and overlays them on the ground-truth
distribution.

Also produces a simple 4-panel plot of the truth data alone:

```bash
python plot_water_budget_truth.py
# → water_budget_1980.png
```

Script: `train_johns/plot_water_budget_truth.py`

---

## Observed Symptom

| Epoch | Mean drift_pct |
|-------|---------------|
| 0–3   | N/A (WaterFixer off — warmup) |
| 4     | +1.6%  ← WaterFixer just turned on |
| 5     | +9.3% |
| 6     | +11.7% |
| 7     | +12.6% |
| 8     | +14.7% |
| 9–11  | +19–21% |
| 12–15 | +22–24% (plateau) |

Ground truth: **+0.02%**. Gap at plateau: **~24 percentage points**.

P_sink in the model (~1.40×10¹² kg/s) is ~18% below the truth
(~1.72×10¹² kg/s) — the model learned to severely under-predict precipitation.

---

## Root Cause

**File**: `credit/trainers/trainerERA5.py`, lines ~238–265

```python
# WaterFixer runs on y_pred and scales PRECT upward
input_dict = self.opt_water(input_dict)
y_pred = input_dict["y_pred"]          # PRECT now multiplied by P_correct_ratio ≈ 1.23
...
loss = criterion(y.to(y_pred.dtype), y_pred).mean()   # loss sees corrected PRECT
```

The loss is computed **after** the WaterFixer has already scaled PRECT up by
`P_correct_ratio`. This creates a feedback loop:

1. Model predicts raw PRECT too low → WaterFixer boosts by ×1.23
2. Corrected PRECT > target → loss gradient says "reduce PRECT"
3. Model reduces raw PRECT → WaterFixer boosts by ×1.30
4. Repeat → diverges to ~24% drift, then stalls at a degenerate equilibrium
   where raw PRECT is very low, the ratio is very high, and corrected PRECT
   is near the target but the underlying physics is broken

The WaterFixer also stores `x["water_conservation_loss"] = (P_correct_ratio - 1)²`
but the trainer **never adds this to the total loss** — it is computed and discarded.

---

## Proposed Fixes

### Option 1 — Turn WaterFixer off during training (recommended, least risky)
Set in config (both `camulator_config_extended_v2.yml` **and** `config_reload.yml`):
```yaml
global_water_fixer:
    activate: False          # or activate_outside_model: False
```
Use WaterFixer only at inference time (`Quick_Climate.py`).
Epochs 0–3 used no fixer and already produced clean results.

### Option 2 — Detach the correction ratio
In `credit/postblock/_postblock.py` (line ~792):
```python
# BEFORE (causes feedback):
precip = precip * P_correct_ratio

# AFTER (breaks feedback, keeps corrected loss):
precip = precip * P_correct_ratio.detach()
```
Gradients flow directly to raw PRECT instead of through the ratio computation.

### Option 3 — Compute MSE on pre-correction output, add conservation penalty ✅ IMPLEMENTED
Compute the reconstruction loss **before** the WaterFixer modifies `y_pred`,
then add `conservation_loss_weight × x["water_conservation_loss"]` as a
separate term. This is the cleanest formulation:
- MSE loss drives PRECT → target correctly
- Conservation penalty drives the model toward water balance independently

Motivated by the FuXi physics paper ablation study, which showed that inference-only
conservation is measurably worse than training-with-conservation constraints — meaning
Option 1 would leave performance on the table.

#### Code changes

**`credit/trainers/trainerERA5.py` — `__init__`**

Added a default attribute and read the weight from config when the WaterFixer is active:

```python
# Default (safe no-op when WaterFixer is off)
self.water_conservation_loss_weight = 0.0

# Inside the GlobalWaterFixer activation block:
self.water_conservation_loss_weight = float(
    post_conf["global_water_fixer"].get("conservation_loss_weight", 0.0)
)
if self.water_conservation_loss_weight > 0:
    logger.info(
        "WaterFixer: MSE loss on pre-correction prediction + "
        "conservation penalty weight=%.4f", self.water_conservation_loss_weight
    )
```

**`credit/trainers/trainerERA5.py` — training loop**

Save `y_pred` before the WaterFixer call and capture the conservation loss it computes:

```python
# BEFORE (feedback loop):
if self.flag_water_conserve:
    input_dict = {"y_pred": y_pred, "x": x}
    input_dict = self.opt_water(input_dict)
    y_pred = input_dict["y_pred"]
...
loss = criterion(y.to(y_pred.dtype), y_pred).mean()

# AFTER (Option 3 fix):
water_cons_loss = None
if self.flag_water_conserve:
    y_pred_pre_water = y_pred          # save pre-correction for MSE loss
    input_dict = {"y_pred": y_pred, "x": x}
    input_dict = self.opt_water(input_dict)
    y_pred = input_dict["y_pred"]
    water_cons_loss = input_dict.get("water_conservation_loss", None)
...
with torch.autocast(enabled=self.amp, device_type="cuda"):
    use_pre = (
        self.flag_water_conserve
        and self.water_conservation_loss_weight > 0
        and water_cons_loss is not None
    )
    y_for_loss = y_pred_pre_water if use_pre else y_pred
    loss = criterion(y.to(y_for_loss.dtype), y_for_loss).mean()
    if use_pre:
        loss = loss + self.water_conservation_loss_weight * water_cons_loss
```

When `conservation_loss_weight: 0.0` the `use_pre` gate is False and behaviour is
**identical to before** — safe no-op. Only activates when the weight is nonzero.

**Config change** (both `camulator_config_extended_v2.yml` and `config_reload.yml`):

```yaml
global_water_fixer:
    conservation_loss_weight: 0.1   # was 0.0
```

---

## Result

Epoch 16, step 1 → step 51 (50 gradient steps, ~25 minutes):

| Metric | Step 1 (broken) | Step 51 (fixed) | Ground truth |
|--------|----------------|-----------------|--------------|
| drift_pct (avg 4 GPUs) | ~23% | **~0.1%** | 0.024% |
| P_sink | ~1.40×10¹³ kg/s | ~1.74×10¹³ kg/s | ~1.72×10¹³ kg/s |

The model corrected 12 epochs of degenerate PRECT learning in **50 gradient steps**
with no loss instability. Log files showing the fix:
```
train_johns/camulator_v2_consfix.o*   # epoch 16+ (fix active)
```

---

## Key Files

| File | Purpose |
|---|---|
| `train_johns/experiments/compute_water_budget_truth.py` | Compute ground-truth P_sink/E_src/dTWC/dt/residual/drift_pct from zarr |
| `train_johns/plot_water_budget_truth.py` | Plot the ground-truth CSV (4-panel) |
| `train_johns/plot_drift_comparison.py` | Parse training logs + compare model drift to truth |
| `train_johns/water_budget_1980.csv` | Ground-truth water budget output (1980, 1459 rows) |
| `train_johns/water_budget_1980.png` | Ground-truth 4-panel plot |
| `train_johns/drift_comparison.png` | Model vs truth comparison (4-panel) |
| `credit/trainers/trainerERA5.py` lines 238–265 | Where the feedback loop lives |
| `credit/postblock/_postblock.py` lines ~758–797 | WaterFixer forward: ratio + correction |

---

## Reference Configs

Training config:
```
/glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended_v2.yml
```
Reload config (used by jobs 2–N in a chain):
```
/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_reload.yml
```

Relevant log files showing the full drift trajectory:
```
train_johns/camulator_ext_v2_finetune.o3146512   # epochs 4–7  (fixer on, ramp begins)
train_johns/camulator_ext_v2_finetune.o3148281   # epochs 8–11 (ramp continues)
train_johns/camulator_ext_v2_finetune.o3148282   # epochs 12–15 (plateau at ~23%)
```
