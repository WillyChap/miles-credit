# Experiments — AMS Lessons Learned paper

Ordered plan of experiments to run, plus the preserved configs that drive them.
Companion to `papers/EXPERIMENT_TRACKER.md` (what we already have) and
`papers/lessons_learned_conservation_feedback.md` (the draft).

The configs in `configs/` were copied off purgeable scratch on 2026-06-29 and given
documentation headers. Data, normalization, and statics files still live on shared
read-only scratch/campaign storage and are referenced in place (too large to relocate).
Before launching any run, update `save_loc`, `pbs.conda`, and `pbs.project`.

## Config index

| Config | Role | Trainer | Water fixer / weight | Source (scratch) |
|---|---|---|---|---|
| `configs/recommended_v1_decouple.train.yml` | **Recommended fix** (R2) | `era5` | on / **0.1** | `b_credit_runs/camulator_config_extended_v2.yml` |
| `configs/recommended_v1_decouple.reload.yml` | R2 resume/chain | `era5` | on / 0.1 | `NEW_CLI_JOHN_CASPER_extended_v2/config_reload.yml` |
| `configs/alt_v2_penalty.train.yml` | Alternative, untested (R3) | `era5-v2` | on / 0.5 | `b_credit_runs/camulator_config_casper.yml` |
| `configs/baseline_no_fixers.train.yml` | Clean baseline / common start | `era5-v2` | **off** | `b_credit_runs/camulator_config.yml` |

Single knob for the failure/fix: in the `era5` (v1) trainer, `conservation_loss_weight: 0.0`
reproduces the broken feedback (the `use_pre` gate turns off and the loss is taken on the
corrected output); any value `> 0` is the fix (pre-correction loss + penalty).

Existing evidence already on disk (no rerun needed): broken run `camulator_ext_v2_finetune.o*`,
fix run `camulator_v2_consfix.o*`, ground truth `water_budget_1980.csv`. See the tracker.

---

## Ordered experiment plan

Ordering is by dependency and value, not just priority. Phase 0 is cheap and unblocks
trust in the baseline; Phase 1 is the scientific core; Phase 2 is conditional polish.

### Phase 0 — cheap, no GPU queue (do first)

**E1. Multi-year truth budget** (was G5)
- What: extend the ground-truth water budget from 1 year to ~5 (1980–1984) to confirm the
  reference drift (mean 0.024%, std 2.22%) and its spread are stationary across years.
- Why: the entire paper measures model drift against this baseline; one year is a weak base.
- How: `python experiments/compute_water_budget_truth.py --years 1980 1981 1982 1983 1984 --out experiments/out/water_budget_1980_1984.csv`
- Cost: minutes, CPU. Supports claim C1.

**E2. float32-vs-float64 rounding demo** (was G6)
- What: a short script summing the global water budget in float32 vs float64 to exhibit the
  ~0.6% accumulation error the fix removes.
- Why: substantiates the double-precision implementation note in §4 without hand-waving.
- How: small standalone script under `experiments/` (to write); no model needed.
- Cost: minutes, CPU. Supports claim C9.

### Phase 1 — scientific core (run in parallel; both high priority)

**E3. Controlled v1-vs-v2 + weight sweep** (was G3) — PIVOTAL
- What: from one common checkpoint (`baseline_no_fixers` checkpoint, fixers off), fine-tune
  with matched rollout length and learning rate under both trainers, sweeping
  `conservation_loss_weight` ∈ {0.0, 0.05, 0.1, 0.5}.
    - v1 (`recommended_v1_decouple.train.yml`): 0.0 = broken control, others = fix.
    - v2 (`alt_v2_penalty.train.yml`): same weights.
  Record per-epoch drift, NaN/divergence incidence, and final forecast skill (RMSE/ACC).
- Why: the current v2 evidence is a single unstable, confounded run. This is the only way
  to make any honest claim about the v2 path or about weight sensitivity. It also gives a
  clean, designed broken-vs-fixed contrast to replace the production-log trajectories.
- Deliverable: drift-vs-epoch by (trainer, weight); table of divergence/NaN events; skill
  at convergence. Supports C6, C10, and strengthens C2/C3.
- Cost: ~8 short fine-tunes (4 weights × 2 trainers). Casper, a few epochs each.

**E4. Long free-running climate rollout** (was G4)
- What: multi-step (≥1-year) rollout from the recommended-fix checkpoint (R2) via
  `climate/Quick_Climate.py`, integrating the water budget over the rollout itself.
- Why: everything we have is training-step (teacher-forced) drift. The limitation a
  reviewer will raise is whether the budget stays closed in free-running integration.
  This converts training-step stability into rollout stability. Supports/closes C7.
- How: `python climate/Quick_Climate.py --config <climate cfg> --model_name <R2 ckpt> --save_append paper_rollout`
  then integrate the budget over the saved forecast.
- Cost: 1–2 inference rollouts. Independent of E3, so run concurrently.

### Phase 2 — conditional

**E5. Inference-only baseline vs training-time penalty** (was G2)
- What: take the `baseline_no_fixers` model, apply the water fixer only at inference, and
  compare budget drift and skill to the training-time penalty (E3 winner).
- Why: the paper currently cites prior work for "training-time conservation beats
  inference-only." Run this only if we want first-party evidence for that claim.
- Cost: 1 fine-tune + 1 rollout. Supports C8. Optional.

**E6. Long rollout of the E3 winner**
- What: repeat E4 using the best checkpoint from E3 (likely a v1 weight in {0.05, 0.1}).
- Why: ties the controlled sweep to the free-running stability result. Only if E3 changes
  the recommended weight.

---

## Recommended minimal set

If running only two: **E3** (the pivotal controlled comparison the paper's "present both
honestly" framing depends on) and **E4** (free-running drift, the main limitation). Do
**E1** and **E2** alongside since they are cheap and shore up the baseline and the
implementation note. **E5** only if we choose to make the training-vs-inference claim from
our own data rather than by citation.

## Folder layout

- `configs/`      — preserved, documented training/inference configs (off purgeable scratch).
- `configs/generated/` — per-(trainer, weight) E3 sweep configs (created by the submitter).
- `pbs/`          — PBS job scripts: `phase0_truth_and_precision.pbs`,
  `phase1_e3_train.pbs` (+ `phase1_e3_sweep_submit.sh`), `phase1_e4_rollout.pbs`.
- `figures/`      — figure build scripts and rendered PDF/PNG (journal-ready, title-free).
- `figure_data/`  — small cached data behind the figures (parsed drift trajectories +
  the CAM6 truth budget), so figures rebuild without the multi-GB training logs.
  `figure_data/build_figure_data.py` regenerates these CSVs from `rechunk_logs/`.
- `float_precision_demo.py` — E2 float32-vs-float64 rounding demo.
- `out/`          — run outputs and derived CSVs (created by the jobs).

### Figure 1 (drift): how to rebuild
`python experiments/figures/plot_drift_figure.py` reads `figure_data/{drift_broken,drift_fixed,
truth_water_budget_1980}.csv` and writes `figures/drift_comparison_combined.{pdf,png}`. The two
drift CSVs were parsed from the `camulator_ext_v2_finetune` (feedback) and `camulator_v2_consfix`
(decoupled + penalty, weight 0.1) logs of the `NEW_CLI_JOHN_CASPER_extended_v2` run; to refresh
them, run `python experiments/figure_data/build_figure_data.py` (verified to reproduce the cached
CSVs byte-for-byte from the logs in `rechunk_logs/`).

## Outputs

Write run outputs and derived CSVs/figures under `experiments/out/` (create as needed) so
they are not lost on scratch. Checkpoints remain on scratch by necessity (size); record
their paths in `papers/EXPERIMENT_TRACKER.md`.
</content>
