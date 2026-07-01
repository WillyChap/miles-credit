# AMS Lessons Learned — Experiment Tracker

Single source of truth for every run, config, log, and checkpoint behind the paper
`papers/lessons_learned_conservation_feedback.md`, plus the gap list (what is already
sufficient vs what still needs to be launched).

Ordered launch plan and the preserved, documented configs live in `experiments/`
(`experiments/README.md` + `experiments/configs/`). The gaps G2–G6 below are ordered there
as E1–E6.

Last updated 2026-06-30. Drift trajectories were extracted from the training logs in
`rechunk_logs/` and are filled in below; no values remain pending.

---

## 1. Runs on disk

All run directories live under `/glade/derecho/scratch/wchapman/CREDIT_runs/`.
All logs live under `/glade/work/wchapman/Roman_Coupling/train_johns/rechunk_logs/`.

| # | Role in paper | Run dir | Trainer | Water fixer | `conservation_loss_weight` | Loss target | Log family (epochs) | Drift behavior | Status |
|---|---|---|---|---|---|---|---|---|---|
| R1 | **Broken (feedback)** | `NEW_CLI_JOHN_CASPER_extended_v2/` | `era5` (v1) | on, outside | 0.0 (pre-fix code) | post-correction | `camulator_ext_v2_finetune.o*` (4–15) | 3.5% → 23.1% monotonic | ✓ have |
| R2 | **Fixed (principled, v1 decouple)** | `NEW_CLI_JOHN_CASPER_extended_v2/` | `era5` (v1) | on, outside | **0.1 (confirmed in log)** | **pre-correction** + penalty | `camulator_v2_consfix.o*` (16–79) | 22.9% → ~0 in 1 epoch; last-5 mean **+0.11%**, std **0.83%** | ✓ have |
| R1b | **Broken (2nd instance)** | (extended) | `era5` (v1) | on | n/a | post-correction | `camulator_extended_finetune.o*` (0–57) | unbounded → **~45% mean** (peaks 62%) | ✓ have |
| R3 | v2 penalty-only path | `NEW_CLI_JOHN_CASPER_small/` | `era5-v2` | on, outside | **unconfirmed** (not logged; config says 0.5) | post-correction + penalty + intermediate | `ps_wxformer_CESM_6h.o*` (0–68, drift 30–68) | **unstable**: osc. −15 to −30%, epoch-53 fork → **+389% + NaN**, restart recovers to last-5 mean +1.15%, std 2.07% | ⚠ confounded |
| R5 | energy-fixer test | `NEW_CLI_JOHN_CASPER_energy_test/` | ? | n/a | n/a | n/a | n/a (2 epochs logged) | n/a | not used |
| T1 | **Ground-truth budget** | n/a (CSV in repo) | n/a | n/a | n/a | n/a | `water_budget_1980.csv` | mean 0.024%, std 2.22% | ✓ have |
| A | **2×2 cell A** (broken) | `CREDIT_runs/e3_v1_w0.0/` | `era5` (v1) | on, outside | 0.0 | post-correction (gate) | `5032824.casper-pbs.OU` (0–4) | **runaway**, settled mean +60%, peaks 72% | ✓ have |
| B | **2×2 cell B** | `CREDIT_runs/e3_B_corr_w0.1/` | `era5` (v1) | on, outside | 0.1 | post-correction (`supervise_precorrection: False`) | `5035892.casper-pbs.OU` (0–4) | drift +0.5% ± 1.6% | ✓ have |
| C | **2×2 cell C** | `CREDIT_runs/e3_C_raw_w0.0/` | `era5` (v1) | on, outside | 0.0 | pre-correction (`supervise_precorrection: True`) | `5035893.casper-pbs.OU` (0–4) | drift +1.1% ± 3.1% | ✓ have |
| D | **2×2 cell D** (fix) | `CREDIT_runs/e3_v1_w0.1/` | `era5` (v1) | on, outside | 0.1 | pre-correction (gate) | `5032826.casper-pbs.OU` (0–4) | drift +0.1% ± 1.6% | ✓ have |

Notes (from log parse):
- **The 2×2 ablation (A/B/C/D) is complete and is now the paper's strongest causal evidence** (Fig. 1b, §5). All four warm-start from the same `cp00092_extended` checkpoint and cross supervised target (corrected vs raw, the v1 `supervise_precorrection` gate) with penalty weight (0 vs 0.1). Clean double dissociation: only cell A (corrected + no penalty) runs away; flipping either knob (B or C) pins drift near zero, both together (D) tightest. Stats over epoch>0.5. Data cached in `experiments/figure_data/screen_{A,B,C,D}.csv`; rebuild via `experiments/figure_data/build_screen_data.py` then `experiments/figures/plot_fig1.py`.
- Only R2 logs the init line `trainerERA5:WaterFixer: MSE loss on pre-correction prediction + conservation penalty weight=0.1000`, confirming both the **v1 pre-correction decouple** and the **0.1 weight**. This is direct evidence for the fix the paper recommends as principled.
- R1 and R2 share one `save_loc` and checkpoint history (`training_log.csv` epochs 0–79). R2 is the continuation of R1 after the fix.
- **R1b is a second, independent occurrence of the same feedback failure** (drifts further, to ~45%). Reproducibility of the failure, not a one-off.
- **R3 is the only run on `trainerERA5v2`, and it was unstable** (NaN losses, a divergent branch to +389%, recovery only after an epoch-53 restart). The penalty weight is not recorded in its logs, and it trained from scratch with unrelated NaN events, so its instability **cannot be cleanly attributed** to the post-correction coupling. It is suggestive, not conclusive, evidence about the v2 path.

---

## 2. Configs

| Config | Path | Trainer | Water fixer / weight | save_loc | Purpose |
|---|---|---|---|---|---|
| extended_v2 (train) | `/glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended_v2.yml` | `era5` | on/outside, 0.1 | `NEW_CLI_JOHN_CASPER_extended_v2/` | R1+R2 |
| reload (chain) | `…/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/config_reload.yml` | `era5` | on/outside, 0.1 | same | R2 resume |
| casper (train) | `/glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_casper.yml` | `era5-v2` | on/outside, 0.5 | `NEW_CLI_JOHN_CASPER_small/` | R3 |
| derecho (train) | `/glade/derecho/scratch/wchapman/b_credit_runs/camulator_config.yml` | era5-v2 | **all fixers OFF** | (derecho exp) | clean-train baseline |
| climate (infer) | `climate/camulator_config.yml` | n/a (rollout) | on/outside, no weight | n/a | inference rollout |
| climate_new (infer) | `climate/camulator_config_new.yml` | n/a | on/outside, 0.05 | n/a | inference rollout |

Reference checkpoints / data (read-only, shared):
- Ground-truth budget CSV: `water_budget_1980.csv` (repo root); generator `experiments/compute_water_budget_truth.py`.
- Pre-trained well-validated checkpoint (epoch 91): `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/CAMulator_models/checkpoint.pt00091.pt`.
- Pre-made ICs for rollouts: `/glade/campaign/cisl/aiml/wchapman/MLWPS/STAGING/init_times/`.
- Drift figures already produced: `drift_comparison_combined.png`, `_old.png`, `_new.png`, `drift_comparison.png`; script `plot_drift_comparison.py`.

---

## 3. Claim → evidence map

Status: ✓ sufficient · ⚠ partial/confounded · ✗ not in hand

| Claim in paper | Evidence | Status | Note |
|---|---|---|---|
| C1 Truth budget closes to ~0 (0.024% ± 2.22%) | `water_budget_1980.csv` | ✓ | one year only (1980) |
| C2 Feedback drives drift to ~23% (and worse) | R1 (3.5%→23.1%) **and R1b (→45%)** | ✓✓ | reproduced in two independent runs |
| C3 Penalty fix recovers drift to ~0 | R2: 22.9% → −1.08% by epoch 17 (one epoch) | ✓ | matches the "~50 gradient steps" recovery in the bug doc; that recovery is **R2 = v1**, not v2 |
| C4 Raw P_sink collapses under feedback, recovers under fix | R1/R2 logs (P_sink, P_ratio) | ✓ | derivable from logs (P_ratio = drift+1) |
| C5 v1 pre-correction decouple is the principled fix | R2 IS the v1 trainer at weight 0.1 (confirmed) | ✓ | direct, not just argued |
| C6 v2 penalty-only works empirically | R3 (`ps_wxformer`, era5-v2) was **unstable** (NaN, +389% fork) | ✗/⚠ | does NOT support "v2 works"; confounded (weight unlogged, from-scratch, other NaNs). Needs clean G3 |
| C7 Fix is stable long-horizon **in training** | R2: epochs 16–79, last-5 mean +0.11%, std 0.83%, no secular drift | ✓ | this is training-step drift; free-running rollout drift is separate (G4) |
| C8 Training-time conservation beats inference-only | none (relies on prior work) | ✗ | needs G2 or cite-only |
| C9 float64 accumulation removes ~0.6% rounding | code (`physics_core.py`) | ⚠ | could add a 5-line numerical demo (G6) |
| C10 Outcome depends on penalty weight | 3 configs at 0.05/0.1/0.5, not controlled | ⚠ | not a clean sweep (G3); but the 2×2 below cleanly contrasts weight 0 vs 0.1 |
| C12 Runaway needs corrected-supervision **and** no penalty (double dissociation) | 2×2 cells A/B/C/D, common checkpoint | ✓ | only A drifts; clean dissociation (Fig 1b, §5) |
| C13 Raw supervision alone (no penalty) prevents runaway | cell C: +1.1% ± 3.1% | ✓ | confirms the mechanism: loss sees raw amplitude |
| C14 Penalty alone stabilizes corrected-output supervision | cell B: +0.5% ± 1.6% | ✓ | establishes the alternative the draft previously only conjectured |
| C11 Fix does not degrade forecast skill | `training_log.csv` (R1+R2) | ⚠ | confounded: loss jumps at epochs 20 & 36 (rollout/fixer-set changes) → not a clean on/off comparison |
| Fig 2 schematic (gradient paths) | none | ✗ | to draw (not an experiment) |

---

## 4. Verdict: sufficient vs needed

**The core finding is already supported, and the failure is reproducible.** Two
independent runs show the feedback failure (R1 → 23%, R1b → 45%), the v1 penalty fix (R2)
recovers to near zero within one epoch and holds for 64 epochs (last-5 mean +0.11%, std
0.83%), and the ground truth (T1) sets the zero. The three planned figures can be built
from existing data. The central constraint-placement claim and the recommendation of the
v1 decouple can be written now.

**Two things the parse changed (now applied to the draft, 2026-06-29):**
1. The "recovers in ~50 gradient steps" result is **R2 (v1, weight 0.1)**, not the v2
   path. The draft now places this demonstrated recovery in §4's second response (the v1
   decouple), with the correct numbers (23% → <1% in one epoch, last-5 mean 0.11%, std 0.83%).
2. The only v2 run (R3) was **unstable** (NaN losses, a branch diverging to +389% drift,
   recovery only after a restart). The draft §4 third response now reports the v2 path as
   **untested rather than working**, noting the instability is confounded (from-scratch,
   unlogged weight, unrelated NaNs). A clean v2 experiment (G3) is still needed to say more.

The draft also now frames the fix as a **hybrid** (hard correction on the delivered field +
soft penalty on the uncorrected prediction), with hard/soft/hybrid terminology aligned to
Beucler et al. (2021) across the abstract, §1, §4, and §5.

**Experiments / actions, by priority:**

| ID | Experiment | Supports | Priority | Cost | Decision |
|---|---|---|---|---|---|
| G3 | Clean v2-vs-v1 comparison: from one common checkpoint, fine-tune both trainers with matched rollout/LR; sweep weight (0.0, 0.05, 0.1, 0.5); record drift recovery + NaN incidence + skill | C6, C10 | **high** | 4–6 short fine-tunes | launch if the paper makes any claim about v2 or about weight sensitivity |
| G4 | Long free-running climate rollout (≥1 yr) from the R2 checkpoint via `Quick_Climate.py`; measure budget drift over the rollout, not just per training step | C7 limitation | **high** | 1–2 inference rollouts | recommended; closes the main limitation (rollout vs training drift) |
| G2 | Inference-only baseline: fixer OFF in training, applied only at inference; compare drift + skill to R2 | C8 | med | 1 fine-tune | only if we want first-party evidence vs citing prior work |
| G5 | Multi-year truth budget (1980–1984) to confirm ±2.2% spread is stationary | C1 limitation | low | cheap (`experiments/compute_water_budget_truth.py --years`) | optional |
| G6 | float32-vs-float64 numerical demo of the ~0.6% rounding | C9 | low | trivial script | optional |

**Recommended minimal launch set:** G3 and G4. G3 is now the pivotal one: without a clean
v2 run, the honest framing is "v1 decouple is demonstrated and stable; the single v2
attempt was unstable and confounded." G4 converts training-step stability into the
free-running climate stability a reviewer will expect. G2/G5/G6 are optional or cheap.

---

## 5. Open reference/figure TODOs (not experiments)

- Fig 2 gradient-path schematic: draw (matplotlib or vector).
- `[CITE]` FuXi physics-constraint ablation (used for C8 if cited rather than measured).
- `[CITE]` Sha et al. 2025 JAMES volume/pages.
- Acknowledgments: funding, allocation, coauthors.
</content>
