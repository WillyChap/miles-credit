---
name: GraphCast masked backwards optimization — next phase
description: Next major task is applying masked backwards optimization to GraphCast (or equivalent differentiable weather model) to extend the Lorenz toy results to real weather forecasting
type: project
---

The next task after the Lorenz figures is to bring in GraphCast (or a differentiable weather model equivalent) and apply the same masked backwards optimization framework.

**Goal:** Show that masking over *weather fields* (e.g., temperature-only, wind-only, geopotential-only) produces more interpretable initial condition corrections, analogous to the Lorenz x/y/z masking results.

**Why:** Direct analog to Vonich & Hakim (2024), who used backprop through GraphCast to optimize ERA5 ICs for the PNW heatwave. The new contribution is the *masked/gated* version revealing which fields drive the improvement.

**How to apply:** When the user returns to this task, start by investigating what differentiable weather model is available on Derecho/Casper, check for existing GraphCast checkpoints or PyTorch ports, and plan the masking strategy over fields/levels/regions.

**Key open questions:**
- Which GraphCast implementation is accessible (JAX DeepMind original vs PyTorch port)?
- What ERA5 case study to use (PNW heatwave is natural, or a different extreme event)?
- Masking granularity: per-field (T, U, V, Z, Q), per-level, per-region, or combinations?
- Optimization budget: Vonich & Hakim needed careful tuning; gradient path through GraphCast is much longer than Lorenz
