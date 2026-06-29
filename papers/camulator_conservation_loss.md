# CAMulator training loss with the water-conservation penalty

This note writes out, mathematically, the total loss optimized when CAMulator is trained with the
`GlobalWaterFixer` post-block conservation penalty active. **Every equation below has been verified
against the source** (file:line references inline); a verification pass over all 20 claims found no
discrepancies.

Source files:
- Supervised loss: `credit/losses/weighted_loss.py` (`VariableTotalLoss2D`), `credit/losses/base_losses.py`
- Water budget + penalty: `credit/postblock/_postblock.py` (`GlobalWaterFixer.forward`, ~690–803)
- Area integration: `credit/physics_core.py` (`weighted_sum`, `total_column_water`)
- Loss assembly: `credit/trainers/trainerERA5v2.py` (~235–294), `credit/trainers/trainerERA5.py` (~238–290)

---

## Notation

Grid cells indexed $(i,j)$ over (lat, lon); $a_{ij}$ = area weight; batch samples $b=1\dots B$;
variable channels $v=1\dots V$ (3D vars expanded per level, then surface, then diagnostics).
$\Delta t$ = `N_seconds` = `lead_time_periods` $\times 3600$ s (`_postblock.py:639`);
$\rho_w$ = `RHO_WATER` (`_postblock.py:24`).

---

## 1. Supervised loss `VariableTotalLoss2D.forward`

Per-element base loss $\ell$ is selected by config `training_loss` with `reduction="none"`
(`weighted_loss.py:213,216,220`; registry in `base_losses.py:41–54` — mse, mae, msle, huber, logcosh, …;
fallback `L1Loss` at line 218). Latitude weight (`weighted_loss.py:35–37`); note the `clamp(min=1e-4)`
is applied to $\cos(\varphi)$ **before** the exponent:

$$
L^{\text{lat}}_{i} = \frac{\big[\max(\cos\varphi_i,\,10^{-4})\big]^{p}}{\overline{\big[\max(\cos\varphi,\,10^{-4})\big]^{p}}},
\qquad p = \texttt{latitude\_weight\_power}\ (\text{default }1.0)
$$

Per-variable weighted mean then averaged over variables (`weighted_loss.py:243–253`):

$$
\mathcal{L}_{\text{sup}}(y,\hat y)=\frac{1}{V}\sum_{v=1}^{V}
\underbrace{\frac{1}{B\,H\,W}\sum_{b,i,j} w_v\, L^{\text{lat}}_{i}\,\ell\!\left(y_{b,v,ij},\,\hat y_{b,v,ij}\right)}_{\texttt{loss\_dict[v].mean()}}
\;+\;\lambda_{\text{pow}}\,\mathcal{L}_{\text{pow}}\;+\;\lambda_{\text{spec}}\,\mathcal{L}_{\text{spec}}
$$

The power and spectral terms are added only in training, not validation (`weighted_loss.py:256–265`);
both $\lambda$ read from config key `spectral_lambda_reg` (lines 195, 208). Argument order is
`forward(self, target, pred)` and the trainer calls `criterion(y, y_pred)` — so $y$ is target,
$\hat y$ is prediction (`weighted_loss.py:222`, `trainerERA5v2.py:279`).

---

## 2. Water-budget terms `GlobalWaterFixer.forward`

Fluxes (`_postblock.py:738–739`) and total-column-water tendency (`:749`):

$$
\text{precip\_flux}_{ij}=\frac{P_{ij}\,\rho_w}{\Delta t},\qquad
\text{evapor\_flux}_{ij}=\frac{E_{ij}\,\rho_w}{\Delta t},\qquad
\frac{\partial \mathrm{TWC}}{\partial t}\Big|_{ij}=\frac{\mathrm{TWC}^{\text{pred}}_{ij}-\mathrm{TWC}^{\text{in}}_{ij}}{\Delta t}
$$

Global area-weighted sums via `core_compute.weighted_sum(·, axis=(-2,-1))` (`:752,755,758`).
**These sums accumulate in float64** (`physics_core.py:279–282, 501–504`: cast to `.double()`, sum,
cast back) to avoid ~0.6% float32 rounding over ~55k cells:

$$
\mathcal{T}_b=\sum_{ij}a_{ij}\,\frac{\partial \mathrm{TWC}}{\partial t}\Big|_{b,ij}
,\qquad
\mathcal{E}_b=\sum_{ij}a_{ij}\,\text{evapor\_flux}_{b,ij}
,\qquad
\mathcal{P}_b=\sum_{ij}a_{ij}\,\text{precip\_flux}_{b,ij}
$$

Residual (`:761`) and PRECT correction ratio (`:764`):

$$
R_b=-\mathcal{T}_b-\mathcal{E}_b-\mathcal{P}_b,\qquad
r_b=\frac{\mathcal{P}_b+R_b}{\mathcal{P}_b}=\frac{-\mathcal{T}_b-\mathcal{E}_b}{\mathcal{P}_b},\qquad
r_b-1=\frac{R_b}{\mathcal{P}_b}
$$

(the second and third equalities follow by substituting $R_b$ — verified algebra).

---

## 3. Water conservation penalty

Stored as `x["water_conservation_loss"]` (`_postblock.py:785`) — mean over batch of the squared
fractional imbalance:

$$
\boxed{\;\mathcal{L}_{\text{water}}=\frac{1}{B}\sum_{b=1}^{B}\bigl(r_b-1\bigr)^2
=\frac{1}{B}\sum_{b=1}^{B}\left(\frac{R_b}{\mathcal{P}_b}\right)^2\;}
$$

PRECT is then rescaled in place by $r_b$ (`:791`, written back to `y_pred` at `:795–800`); this mutates
$\hat y$ but **not** the scalar $\mathcal{L}_{\text{water}}$, which was computed beforehand.

**Two distinct "drift" diagnostics** (do not conflate):
- In the fixer log (`_postblock.py:781`): $\text{drift\_pct} = 100\,(\overline{r}-1)$ — mean of the deviation.
- In the trainer (`trainerERA5v2.py:252`): $\text{drift\_pct} = 100\,\sqrt{\mathcal{L}_{\text{water}}}
  = 100\,\operatorname{RMS}(r-1)$ — root-mean-square of the deviation.

---

## 4. Total optimized loss

Let $\lambda_w$ = `water_conservation_weight` (= config `global_water_fixer.conservation_loss_weight`,
default 0.0; `trainerERA5v2.py:53–54`, `trainerERA5.py:66–68`).

**On a backprop timestep** $t \in$ `backprop_on_timestep`:

$$
\mathcal{L} = \mathcal{L}_{\text{sup}}\!\left(y,\;\hat y^{\,\ast}\right)\;+\;\lambda_w\,\mathcal{L}_{\text{water}}
$$

The corrected prediction $\hat y^{\,\ast}$ **differs by trainer version**:

| Trainer | $\hat y^{\,\ast}$ in $\mathcal{L}_{\text{sup}}$ | Intermediate-step penalty |
|---|---|---|
| **v2** `trainerERA5v2.py:279–281` | $\hat y_{\text{corr}}$: **post-fix** (after mass→water→energy mutate `y_pred`); no pre-correction copy is saved | yes (see below) |
| **v1** `trainerERA5.py:280–288` | $\hat y_{\text{pre-water}}$: **pre-correction**, via the `use_pre` gate | none |

**On intermediate timesteps** (v2 only, $t \notin$ `backprop_on_timestep`; `trainerERA5v2.py:286–293`):

$$
\mathcal{L} = \lambda_w\,\mathcal{L}_{\text{water}}
$$

back-propagated alone with `retain_graph=True`, so the model cannot violate the water budget at an
intermediate rollout step while optimizing a later one. Gradients accumulate across timesteps via
successive `backward()` calls before the optimizer step.

---

## 5. What is *not* in the loss

Mass and energy fixers are **correction-only**. `GlobalMassFixer`, `GlobalEnergyFixer`, and
`GlobalEnergyFixerUpDown` each compute their own ratios ($q_{\text{correct}}$, $s_{p,\text{correct}}$,
$E_{\text{correct}}$) and reassign `x["y_pred"]`, but store **no** `*_conservation_loss` key, and neither
trainer adds any mass/energy penalty term. Only `GlobalWaterFixer` participates in $\mathcal{L}$.

This is the open extension point: adding a mass/energy constraint to the loss would require each fixer's
`forward` to store its own `*_conservation_loss` scalar and the trainer to add a corresponding
$\lambda\,\mathcal{L}$ term — analogous to the water path above.
</content>
