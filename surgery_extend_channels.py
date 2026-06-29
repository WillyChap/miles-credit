"""
surgery_extend_channels.py
--------------------------
Weight surgery to extend CAMulator cp00092 from 15 → 17 diagnostic channels.

Background
----------
The old stable model (checkpoint.pt00092.pt) predicts net radiative fluxes:
  diagnostic_variables (15): PRECT TS CLDHGH CLDLOW CLDMED TAUX TAUY U10 QFLX
                              FSNS FLNS FSNT FLNT SHFLX LHFLX

The new energy-fixer-updown requires up/down flux components.  Variable ordering
is chosen to MINIMISE channel reordering — SHFLX and LHFLX keep their old
positions (13-14 in diag list, full-tensor indices 143-144):

  diagnostic_variables (17): PRECT TS CLDHGH CLDLOW CLDMED TAUX TAUY U10 QFLX
                              FSDS_J FLDS_J FSUS FLUS SHFLX LHFLX FSUTOA FLUT

Changes:
  REMOVED (4):  FSNS FLNS FSNT FLNT  (net fluxes, positions 9-12 in diag list)
  ADDED   (6):  FSDS_J FLDS_J FSUS FLUS FSUTOA FLUT (all zero-init)
  UNCHANGED:    everything else, including SHFLX/LHFLX which stay at the same
                full-tensor indices (143, 144) — no reordering of trained weights.

Total output channels: 145 → 147  (130 prog + 15 diag → 130 prog + 17 diag)

Only two layers in the whole model are affected:
  up_block4.0  Conv2d(512, 580, 3, 3)  →  Conv2d(512, 588, 3, 3)   [pre-PixelShuffle]
  up_block4.2  Conv2d(145, 145, 3, 3)  →  Conv2d(147, 147, 3, 3)   [final output]

Channel mapping (full-tensor indices)
--------------------------------------
  0–138   : prog (U,V,T,Qtot,PS,TREFHT) + PRECT–QFLX  → same positions, direct copy
  139     : FSNS → DROPPED  (new 139 = FSDS_J, zero-init)
  140     : FLNS → DROPPED  (new 140 = FLDS_J, zero-init)
  141     : FSNT → DROPPED  (new 141 = FSUS,   zero-init)
  142     : FLNT → DROPPED  (new 142 = FLUS,   zero-init)
  143     : SHFLX  →  143   (SAME POSITION — direct copy, no reorder)
  144     : LHFLX  →  144   (SAME POSITION — direct copy, no reorder)
  NEW 145 : FSUTOA (zero-init)
  NEW 146 : FLUT   (zero-init)

Usage
-----
    cd /glade/work/wchapman/Roman_Coupling/train_johns
    python surgery_extend_channels.py

Then place the output as checkpoint.pt in the new save_loc:
    mkdir -p .../NEW_CLI_JOHN_CASPER_extended/
    cp .../checkpoint.pt00092_extended.pt .../NEW_CLI_JOHN_CASPER_extended/checkpoint.pt
"""

import torch

# ── paths ─────────────────────────────────────────────────────────────────────
CKPT_IN  = (
    "/glade/derecho/scratch/wchapman/CREDIT_runs/wxformermod_sharp_SpatPS_pxshf/"
    "checkpoint.pt00092.pt"
)
CKPT_OUT = (
    "/glade/derecho/scratch/wchapman/CREDIT_runs/wxformermod_sharp_SpatPS_pxshf/"
    "checkpoint.pt00092_extended.pt"
)

# ── channel counts ────────────────────────────────────────────────────────────
N_PROG        = 130   # U(32)+V(32)+T(32)+Qtot(32)+PS(1)+TREFHT(1)
N_DIAG_OLD    = 15
N_DIAG_NEW    = 17
N_OUT_OLD     = N_PROG + N_DIAG_OLD   # 145
N_OUT_NEW     = N_PROG + N_DIAG_NEW   # 147
PIXEL_SHUFFLE = 2                      # scale factor in PixelShuffle
SCALE2        = PIXEL_SHUFFLE ** 2    # 4

# ── old→new channel map ───────────────────────────────────────────────────────
# old_to_new[k] = new channel index for old channel k (None = drop)
# Channels 0-138: identical positions
# Channels 139-142: FSNS/FLNS/FSNT/FLNT → dropped (new 139-142 = FSDS_J/FLDS_J/FSUS/FLUS)
# Channels 143-144: SHFLX/LHFLX stay at the SAME positions
# New channels 145-146: FSUTOA/FLUT (zero-initialized, no source in old model)
old_to_new = list(range(139))           # 0-138: direct copy
old_to_new += [None, None, None, None]  # 139-142: FSNS/FLNS/FSNT/FLNT → dropped
old_to_new += [143, 144]                # 143-144: SHFLX/LHFLX → same positions
assert len(old_to_new) == N_OUT_OLD

# ── load checkpoint ───────────────────────────────────────────────────────────
print(f"Loading: {CKPT_IN}")
ckpt = torch.load(CKPT_IN, map_location="cpu", weights_only=False)

raw_state = ckpt.get("model_state_dict", ckpt)
if any(k.startswith("module.") for k in raw_state):
    print("  Stripping 'module.' DDP prefix.")
    raw_state = {k.replace("module.", "", 1): v for k, v in raw_state.items()}
state = {k: v.clone() for k, v in raw_state.items()}

# ── detect spectral norm (weights stored as weight_orig / weight_u / weight_v) ──
SPEC_NORM = "up_block4.0.weight_orig" in state
W0_KEY = "up_block4.0.weight_orig" if SPEC_NORM else "up_block4.0.weight"
W2_KEY = "up_block4.2.weight_orig" if SPEC_NORM else "up_block4.2.weight"
print(f"Spectral norm detected: {SPEC_NORM}  (weight key: {'weight_orig' if SPEC_NORM else 'weight'})")

# ── verify input shapes ───────────────────────────────────────────────────────
w0_old = state[W0_KEY]    # [580, C_in, 3, 3]
b0_old = state["up_block4.0.bias"]     # [580]
w2_old = state[W2_KEY]    # [145, 145, 3, 3]
b2_old = state["up_block4.2.bias"]     # [145]

C_in = w0_old.shape[1]
assert w0_old.shape[0] == N_OUT_OLD * SCALE2, \
    f"Expected up_block4.0 out={N_OUT_OLD*SCALE2}, got {w0_old.shape[0]}"
assert w2_old.shape == (N_OUT_OLD, N_OUT_OLD, 3, 3), \
    f"Expected up_block4.2 ({N_OUT_OLD},{N_OUT_OLD},3,3), got {w2_old.shape}"

print(f"\nVerified old shapes:")
print(f"  {W0_KEY} : {tuple(w0_old.shape)}  (C_in={C_in})")
print(f"  {W2_KEY} : {tuple(w2_old.shape)}")

# ── build new weight tensors ──────────────────────────────────────────────────
new_w0 = torch.zeros(N_OUT_NEW * SCALE2, C_in, 3, 3)
new_b0 = torch.zeros(N_OUT_NEW * SCALE2)
new_w2 = torch.zeros(N_OUT_NEW, N_OUT_NEW, 3, 3)
new_b2 = torch.zeros(N_OUT_NEW)

# ── surgery: up_block4.0 (pre-PixelShuffle Conv2d, output dim only changes) ──
# PixelShuffle(scale=2): each output channel k uses pre-shuffle channels k*4:(k+1)*4.
# Since SHFLX/LHFLX stay at positions 143/144, their pre-shuffle groups also stay.
# We can use simple slice copies:
#   - Groups 0:556   (channels 0-138):   direct copy  [prog + 9 unchanged diag]
#   - Groups 556:572 (channels 139-142): DROP          [FSNS/FLNS/FSNT/FLNT]
#   - Groups 572:580 (channels 143-144): direct copy  [SHFLX/LHFLX same pos]
#   - Groups 580:588 (channels 145-146): zero-init    [FSUTOA/FLUT, new]
print("\nSurgery on up_block4.0 (pre-PixelShuffle)...")
new_w0[0:556]     = w0_old[0:556]     # channels 0–138: copy
new_b0[0:556]     = b0_old[0:556]
new_w0[572:580]   = w0_old[572:580]   # SHFLX/LHFLX groups: copy (same position)
new_b0[572:580]   = b0_old[572:580]
# groups 556:572 (FSNS/FLNS/FSNT/FLNT) stay zero — dropped
# groups 580:588 (FSUTOA/FLUT)          stay zero — new, zero-init
print(f"  Copied {556 + 8} / {N_OUT_NEW*SCALE2} pre-shuffle channels; "
      f"{N_OUT_NEW*SCALE2 - 556 - 8} zeroed (dropped + new)")

# ── surgery: up_block4.2 (final output Conv2d, both dims change) ─────────────
# SHFLX/LHFLX stay at 143/144 so we can use block copies instead of a double loop.
# For cross-terms involving SHFLX/LHFLX (rows/cols 143-144), they remain at the
# same indices — the surgery loop handles this naturally.
print("Surgery on up_block4.2 (final output layer)...")
n_copied = 0
for old_out, new_out in enumerate(old_to_new):
    if new_out is None:
        continue
    for old_in, new_in in enumerate(old_to_new):
        if new_in is None:
            continue
        new_w2[new_out, new_in] = w2_old[old_out, old_in]
        n_copied += 1
    new_b2[new_out] = b2_old[old_out]

print(f"  Copied {n_copied} / {N_OUT_NEW**2} weight entries")

# ── update state dict ─────────────────────────────────────────────────────────
state[W0_KEY] = new_w0
state["up_block4.0.bias"] = new_b0
state[W2_KEY] = new_w2
state["up_block4.2.bias"] = new_b2

# spectral norm: also resize weight_u and weight_v
# weight_u shape: (out_channels,)  — random unit vector, power iteration converges quickly
# weight_v shape: (in_channels*kH*kW,)  — for up_block4.0 in_channels unchanged;
#                                          for up_block4.2 in_channels grows to N_OUT_NEW
if SPEC_NORM:
    def _unit_vec(n):
        v = torch.randn(n); return v / v.norm()

    # up_block4.0: out grows 580→588, in_channels unchanged → weight_v unchanged
    state["up_block4.0.weight_u"] = _unit_vec(N_OUT_NEW * SCALE2)
    # weight_v stays same shape (C_in*9); just keep original
    # (state["up_block4.0.weight_v"] unchanged — C_in didn't change)

    # up_block4.2: both in and out grow 145→147
    state["up_block4.2.weight_u"] = _unit_vec(N_OUT_NEW)
    state["up_block4.2.weight_v"] = _unit_vec(N_OUT_NEW * 3 * 3)   # 147*9=1323
    print("  Spectral-norm u/v vectors re-initialized for resized layers.")

# ── verify output shapes ──────────────────────────────────────────────────────
assert state[W0_KEY].shape == (N_OUT_NEW * SCALE2, C_in, 3, 3)
assert state[W2_KEY].shape == (N_OUT_NEW, N_OUT_NEW, 3, 3)
print(f"\nNew shapes verified:")
print(f"  {W0_KEY} : {tuple(state[W0_KEY].shape)}")
print(f"  {W2_KEY} : {tuple(state[W2_KEY].shape)}")

# ── sanity checks ─────────────────────────────────────────────────────────────
# Channel 0 (first U level): identical
assert torch.allclose(state[W2_KEY][0, 0], w2_old[0, 0]), \
    "Prog channel 0 was modified!"
# SHFLX: old[143,143] → new[143,143] (same index)
assert torch.allclose(state[W2_KEY][143, 143], w2_old[143, 143]), \
    "SHFLX diagonal mismatch!"
# LHFLX: old[144,144] → new[144,144] (same index)
assert torch.allclose(state[W2_KEY][144, 144], w2_old[144, 144]), \
    "LHFLX diagonal mismatch!"
# FSDS_J (new, zero-init): row 139 should be all zeros
assert state[W2_KEY][139].abs().sum() == 0, \
    "FSDS_J row should be zero-initialized!"
print("Sanity checks passed: prog intact, SHFLX/LHFLX at same positions, new diag zeroed.")

# ── save ──────────────────────────────────────────────────────────────────────
new_ckpt = dict(ckpt)
new_ckpt["model_state_dict"] = state
if "epoch" in new_ckpt:
    print(f"\nResetting epoch {new_ckpt['epoch']} → 0")
    new_ckpt["epoch"] = 0

print(f"\nSaving to: {CKPT_OUT}")
torch.save(new_ckpt, CKPT_OUT)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Output channels: {N_OUT_OLD} → {N_OUT_NEW}")
print(f"  Dropped  (4): FSNS  FLNS  FSNT  FLNT  (were at 139-142)")
print(f"  Kept     (2): SHFLX LHFLX  (stay at 143, 144 — NO reorder)")
print(f"  Zero-init(6): FSDS_J FLDS_J FSUS FLUS FSUTOA FLUT")
print(f"\nNext:")
print(f"  mkdir -p /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended/")
print(f"  cp {CKPT_OUT} \\")
print(f"     .../NEW_CLI_JOHN_CASPER_extended/checkpoint.pt")
print(f"\n  Then launch fine-tuning with camulator_config_extended.yml")
print("="*60)
