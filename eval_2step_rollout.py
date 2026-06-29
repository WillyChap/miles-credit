"""
eval_2step_rollout.py

Autoregressive 2-step diagnostic for extended CAMulator checkpoints.

Uses the FIXED consecutive-timestep dataloader (DistributedMultiStepBatchSampler
wrapped to always pass i=0) so each pair of next(dl) calls yields genuinely
consecutive time steps T0 and T0+6h.

For N_SAMPLES pairs from 2013 validation:
  Step 1: model(x_t0_gt)   → y_hat_t1  (vs y_t1_gt)
  Step 2: model(x_t1_hat)  → y_hat_t2  (vs y_t2_gt)
          where x_t1_hat[:n_prog] = y_hat_t1[:n_prog]    (autoregressive)
          and   x_t1_hat[n_prog:] = x_t1_new[n_prog:]    (forcing from dataset at T0+6h)

Reports:
  - Step 1 vs Step 2 mean ACC, RMSE, MAE
  - Per-variable RMSE for all metrics variables at each step
  - Qtot column mean (denorm) at each step vs ground truth
  - PRECT global mean (denorm) at each step vs ground truth
  - Water drift: mean abs fractional Qtot error at step 1 vs step 2
  - Saves a pandas-readable CSV: eval_results.csv

Usage (interactive A100 node, credit-casper-mar2026 env):
    cd /glade/work/wchapman/Roman_Coupling/train_johns
    python eval_2step_rollout.py
"""

import os
import sys
import copy
import logging
import numpy as np
import pandas as pd
import torch
import yaml
import xarray as xr

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from credit.models import load_model
from credit.datasets.load_dataset_and_dataloader import load_dataset, load_dataloader
from credit.data import concat_and_reshape, reshape_only
from credit.metrics import LatWeightedMetrics
from credit.trainers.utils import cycle
from credit.parser import credit_main_parser

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

# suppress noisy sampler warning (epoch not set — irrelevant for eval)
class _SuppressEpochWarning(logging.Filter):
    def filter(self, record):
        return "set_epoch" not in record.getMessage()
logging.getLogger().addFilter(_SuppressEpochWarning())

# ── config / checkpoint ───────────────────────────────────────────────────────
CONFIG   = "/glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_extended.yml"
CKPT_DIR = "/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended"

# Every-5 sweep + best checkpoint
CKPTS = [
    "checkpoint.pt00000.pt",
    "checkpoint.pt00005.pt",
    "checkpoint.pt00010.pt",
    "checkpoint.pt00015.pt",
    "checkpoint.pt00020.pt",
    "checkpoint.pt00025.pt",
    "checkpoint.pt00030.pt",
    "checkpoint.pt00035.pt",
    "best_checkpoint.pt",
]

USE_EMA  = False
N_SAMPLES = 25
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output CSV path — named by step count so reruns don't clobber previous results
N_STEPS = 3
CSV_OUT = os.path.join(CKPT_DIR, f"eval_{N_STEPS}step_results.csv")

# ── channel layout ────────────────────────────────────────────────────────────
# x (input):  [0:128] 3D prog, [128:130] 2D prog, [130:136] forcing+static
# y (output): [0:128] 3D prog, [128:130] 2D prog, [130:147] diagnostics
N_PROG    = 130
QTOT_INDS = slice(96, 128)
PRECT_IND = 130
QFLX_IND  = 138   # PRECT TS CLDHGH CLDLOW CLDMED TAUX TAUY U10 QFLX

# ── load config ───────────────────────────────────────────────────────────────
with open(CONFIG) as f:
    conf = yaml.safe_load(f)
conf["save_loc"] = CKPT_DIR
conf = credit_main_parser(conf)

# ── normalization stats for physical diagnostics ──────────────────────────────
ds_mean = xr.open_dataset(conf["data"]["mean_path"])
ds_std  = xr.open_dataset(conf["data"]["std_path"])

# ── build model architecture once (weights loaded per checkpoint below) ───────
model_arch = load_model(copy.deepcopy(conf))

qtot_mean_t = torch.tensor(ds_mean["Qtot"].values, dtype=torch.float32).to(DEVICE).view(1, 32, 1, 1)
qtot_std_t  = torch.tensor(ds_std["Qtot"].values,  dtype=torch.float32).to(DEVICE).view(1, 32, 1, 1)
prect_mean  = float(ds_mean["PRECT"].values.flat[0])
prect_std   = float(ds_std["PRECT"].values.flat[0])

# ── validation dataset / loader ───────────────────────────────────────────────
valid_dataset = load_dataset(conf, rank=0, world_size=1, is_train=False)
valid_dataset.set_epoch(0)
valid_loader  = load_dataloader(conf, valid_dataset, rank=0, world_size=1, is_train=False)
print(f"Validation batches available: {len(valid_loader)}")

# ── batch assembly (mirrors trainerERA5 exactly) ─────────────────────────────
def prepare_xy(batch, device):
    """Assemble x and y from ERA5_MultiStep_Batcher batch keys."""
    if "x_surf" in batch:
        x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device).float()
    else:
        x = reshape_only(batch["x"]).to(device).float()

    if "x_forcing_static" in batch:
        x_forc = batch["x_forcing_static"].to(device).float().permute(0, 2, 1, 3, 4)
        x = torch.cat((x, x_forc), dim=1)

    if "y_surf" in batch:
        y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device).float()
    else:
        y = reshape_only(batch["y"]).to(device).float()

    if "y_diag" in batch:
        y_diag = batch["y_diag"].to(device).float().permute(0, 2, 1, 3, 4)
        y = torch.cat((y, y_diag), dim=1)

    return x, y

# ── metrics ───────────────────────────────────────────────────────────────────
metrics = LatWeightedMetrics(conf, training_mode=False)

# ── physical diagnostic helpers ───────────────────────────────────────────────
def qtot_column_mean(y):
    q = y[:, QTOT_INDS, :, :].float()
    return (q * qtot_std_t + qtot_mean_t).mean().item()

def qtot_drift_frac(y_pred, y_gt):
    q_pred = y_pred[:, QTOT_INDS, :, :].float() * qtot_std_t + qtot_mean_t
    q_gt   = y_gt[:,   QTOT_INDS, :, :].float() * qtot_std_t + qtot_mean_t
    return (q_pred - q_gt).abs().mean().item() / q_gt.abs().mean().item()

def prect_mean_phys(y):
    return (y[:, PRECT_IND, :, :].float() * prect_std + prect_mean).mean().item()

# ── summary rows (one per checkpoint) ────────────────────────────────────────
# Columns: checkpoint, s1_acc, s2_acc, s1_rmse, s2_rmse, s1_mae, s2_mae,
#          s1_qdrift_pct, s2_qdrift_pct,
#          s1_prect, s2_prect, gt1_prect, gt2_prect,
#          s1_qtot,  s2_qtot,  gt1_qtot,  gt2_qtot,
#          + rmse_<var>_s1 / rmse_<var>_s2 for each per-variable metric
summary_rows = []

for ckpt_name in CKPTS:
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"\nSkipping {ckpt_name} — not found")
        continue

    model = copy.deepcopy(model_arch)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model = model.to(DEVICE).eval()
    print(f"\n{'='*70}")
    print(f"  Checkpoint: {ckpt_name}")

    # ── per-checkpoint accumulators ───────────────────────────────────────────
    s1_acc,  s2_acc,  s3_acc  = [], [], []
    s1_rmse, s2_rmse, s3_rmse = [], [], []
    s1_mae,  s2_mae,  s3_mae  = [], [], []
    s1_prect, s2_prect, s3_prect = [], [], []
    gt1_prect, gt2_prect, gt3_prect = [], [], []
    s1_qtot,  s2_qtot,  s3_qtot  = [], [], []
    gt1_qtot, gt2_qtot, gt3_qtot = [], [], []
    s1_qdrift, s2_qdrift, s3_qdrift = [], [], []
    s1_per_var, s2_per_var, s3_per_var = {}, {}, {}

    dl = cycle(valid_loader)
    n_evaluated = 0

    print(f"  Running {N_SAMPLES} triplets...")
    print(f"  {'Pair':>5}  {'S1 ACC':>8}  {'S2 ACC':>8}  {'S3 ACC':>8}  "
          f"{'S1 RMSE':>9}  {'S2 RMSE':>9}  {'S3 RMSE':>9}  {'S3 Qdrift':>10}")
    print("  " + "-" * 90)

    with torch.no_grad():
        for _ in range(N_SAMPLES):
            # ── t=1: initial step ─────────────────────────────────────────────
            x_t0, y_t1_gt = prepare_xy(next(dl), DEVICE)
            y_hat_t1 = model(x_t0)

            # ── t=2: T0+6h, autoregressive ───────────────────────────────────
            x_t1_full, y_t2_gt = prepare_xy(next(dl), DEVICE)
            bs     = min(x_t1_full.shape[0], y_hat_t1.shape[0])
            n_prog = x_t1_full.shape[1] - 6   # 4 dyn forcing + 2 static

            x_t1_hat = x_t1_full[:bs].clone()
            x_t1_hat[:, :n_prog] = y_hat_t1[:bs, :n_prog]
            y_hat_t2 = model(x_t1_hat)

            # ── t=3: T0+12h, autoregressive ──────────────────────────────────
            x_t2_full, y_t3_gt = prepare_xy(next(dl), DEVICE)
            x_t2_hat = x_t2_full[:bs].clone()
            x_t2_hat[:, :n_prog] = y_hat_t2[:bs, :n_prog]
            y_hat_t3 = model(x_t2_hat)

            # trim to common batch size
            y_t1_gt = y_t1_gt[:bs];  y_hat_t1 = y_hat_t1[:bs]
            y_t2_gt = y_t2_gt[:bs];  y_hat_t2 = y_hat_t2[:bs]
            y_t3_gt = y_t3_gt[:bs];  y_hat_t3 = y_hat_t3[:bs]

            # ── metrics ───────────────────────────────────────────────────────
            r1 = metrics(y_hat_t1, y_t1_gt)
            r2 = metrics(y_hat_t2, y_t2_gt)
            r3 = metrics(y_hat_t3, y_t3_gt)

            s1_acc.append(float(r1["acc"]));   s2_acc.append(float(r2["acc"]));   s3_acc.append(float(r3["acc"]))
            s1_rmse.append(float(r1["rmse"])); s2_rmse.append(float(r2["rmse"])); s3_rmse.append(float(r3["rmse"]))
            s1_mae.append(float(r1["mae"]));   s2_mae.append(float(r2["mae"]));   s3_mae.append(float(r3["mae"]))

            for var in metrics.vars:
                s1_per_var.setdefault(var, []).append(float(r1[f"rmse_{var}"]))
                s2_per_var.setdefault(var, []).append(float(r2[f"rmse_{var}"]))
                s3_per_var.setdefault(var, []).append(float(r3[f"rmse_{var}"]))

            # ── physical diagnostics ──────────────────────────────────────────
            s1_prect.append(prect_mean_phys(y_hat_t1)); gt1_prect.append(prect_mean_phys(y_t1_gt))
            s2_prect.append(prect_mean_phys(y_hat_t2)); gt2_prect.append(prect_mean_phys(y_t2_gt))
            s3_prect.append(prect_mean_phys(y_hat_t3)); gt3_prect.append(prect_mean_phys(y_t3_gt))
            s1_qtot.append(qtot_column_mean(y_hat_t1)); gt1_qtot.append(qtot_column_mean(y_t1_gt))
            s2_qtot.append(qtot_column_mean(y_hat_t2)); gt2_qtot.append(qtot_column_mean(y_t2_gt))
            s3_qtot.append(qtot_column_mean(y_hat_t3)); gt3_qtot.append(qtot_column_mean(y_t3_gt))
            s1_qdrift.append(qtot_drift_frac(y_hat_t1, y_t1_gt))
            s2_qdrift.append(qtot_drift_frac(y_hat_t2, y_t2_gt))
            s3_qdrift.append(qtot_drift_frac(y_hat_t3, y_t3_gt))

            n_evaluated += 1
            if n_evaluated % 10 == 0 or n_evaluated == 1:
                print(f"  {n_evaluated:>5}  {np.mean(s1_acc):>8.5f}  {np.mean(s2_acc):>8.5f}  {np.mean(s3_acc):>8.5f}"
                      f"  {np.mean(s1_rmse):>9.5f}  {np.mean(s2_rmse):>9.5f}  {np.mean(s3_rmse):>9.5f}"
                      f"  {np.mean(s3_qdrift)*100:>9.2f}%")

    # ── per-checkpoint console report ─────────────────────────────────────────
    print(f"\n  {'Metric':<12}  {'Step 1 (6h)':>14}  {'Step 2 (12h)':>14}  {'Step 3 (18h)':>14}  {'Δ (S3-S1)':>12}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*12}")
    for label, s1, s2, s3 in [
        ("ACC",  np.mean(s1_acc),  np.mean(s2_acc),  np.mean(s3_acc)),
        ("RMSE", np.mean(s1_rmse), np.mean(s2_rmse), np.mean(s3_rmse)),
        ("MAE",  np.mean(s1_mae),  np.mean(s2_mae),  np.mean(s3_mae)),
    ]:
        print(f"  {label:<12}  {s1:>14.6f}  {s2:>14.6f}  {s3:>14.6f}  {s3-s1:>+12.6f}")

    print(f"\n  {'Diagnostic':<20}  {'S1 pred':>12}  {'S1 GT':>12}  {'S2 pred':>12}  {'S2 GT':>12}  {'S3 pred':>12}  {'S3 GT':>12}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
    print(f"  {'PRECT (kg/m2/s)':<20}  {np.mean(s1_prect):>12.4e}  {np.mean(gt1_prect):>12.4e}"
          f"  {np.mean(s2_prect):>12.4e}  {np.mean(gt2_prect):>12.4e}"
          f"  {np.mean(s3_prect):>12.4e}  {np.mean(gt3_prect):>12.4e}")
    print(f"  {'Qtot col mean':<20}  {np.mean(s1_qtot):>12.4e}  {np.mean(gt1_qtot):>12.4e}"
          f"  {np.mean(s2_qtot):>12.4e}  {np.mean(gt2_qtot):>12.4e}"
          f"  {np.mean(s3_qtot):>12.4e}  {np.mean(gt3_qtot):>12.4e}")
    print(f"  {'Qtot drift (%)':<20}  {np.mean(s1_qdrift)*100:>12.2f}%  {'—':>12}"
          f"  {np.mean(s2_qdrift)*100:>12.2f}%  {'—':>12}"
          f"  {np.mean(s3_qdrift)*100:>12.2f}%  {'—':>12}")

    print(f"\n  Per-variable RMSE — sorted by worst step-3 degradation:")
    print(f"  {'Variable':<24}  {'S1 RMSE':>10}  {'S2 RMSE':>10}  {'S3 RMSE':>10}  {'Δ(S3-S1)':>10}  {'Δ%':>8}")
    print(f"  {'-'*24}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    pv_rows = []
    for var in metrics.vars:
        r1v = np.mean(s1_per_var[var])
        r2v = np.mean(s2_per_var[var])
        r3v = np.mean(s3_per_var[var])
        pv_rows.append((var, r1v, r2v, r3v, r3v - r1v, (r3v - r1v) / (r1v + 1e-12) * 100))
    pv_rows.sort(key=lambda x: x[3], reverse=True)
    for var, r1v, r2v, r3v, delta, pct in pv_rows:
        print(f"  {var:<24}  {r1v:>10.5f}  {r2v:>10.5f}  {r3v:>10.5f}  {delta:>+10.5f}  {pct:>+7.1f}%")

    # ── accumulate into summary row dict ──────────────────────────────────────
    row = {
        "checkpoint":    ckpt_name,
        "s1_acc":        np.mean(s1_acc),
        "s2_acc":        np.mean(s2_acc),
        "s3_acc":        np.mean(s3_acc),
        "s1_rmse":       np.mean(s1_rmse),
        "s2_rmse":       np.mean(s2_rmse),
        "s3_rmse":       np.mean(s3_rmse),
        "s1_mae":        np.mean(s1_mae),
        "s2_mae":        np.mean(s2_mae),
        "s3_mae":        np.mean(s3_mae),
        "s1_qdrift_pct": np.mean(s1_qdrift) * 100,
        "s2_qdrift_pct": np.mean(s2_qdrift) * 100,
        "s3_qdrift_pct": np.mean(s3_qdrift) * 100,
        "s1_prect":      np.mean(s1_prect),  "gt1_prect": np.mean(gt1_prect),
        "s2_prect":      np.mean(s2_prect),  "gt2_prect": np.mean(gt2_prect),
        "s3_prect":      np.mean(s3_prect),  "gt3_prect": np.mean(gt3_prect),
        "s1_qtot":       np.mean(s1_qtot),   "gt1_qtot":  np.mean(gt1_qtot),
        "s2_qtot":       np.mean(s2_qtot),   "gt2_qtot":  np.mean(gt2_qtot),
        "s3_qtot":       np.mean(s3_qtot),   "gt3_qtot":  np.mean(gt3_qtot),
    }
    for var in metrics.vars:
        row[f"rmse_{var}_s1"] = np.mean(s1_per_var[var])
        row[f"rmse_{var}_s2"] = np.mean(s2_per_var[var])
        row[f"rmse_{var}_s3"] = np.mean(s3_per_var[var])

    summary_rows.append(row)

    # Save incrementally after each checkpoint so partial results are kept
    df_partial = pd.DataFrame(summary_rows)
    df_partial.to_csv(CSV_OUT, index=False)
    print(f"\n  [Saved incremental results → {CSV_OUT}]")

# ── final comparison table (console) ─────────────────────────────────────────
W = 100
print("\n\n" + "=" * W)
print("  SUMMARY — all checkpoints")
print("=" * W)
print(f"  {'Checkpoint':<32}  {'S1 ACC':>8}  {'S2 ACC':>8}  {'S3 ACC':>8}"
      f"  {'S1 RMSE':>9}  {'S2 RMSE':>9}  {'S3 RMSE':>9}  {'S3 Qdrift':>10}")
print("  " + "-" * (W - 2))
for row in summary_rows:
    print(f"  {row['checkpoint']:<32}  {row['s1_acc']:>8.5f}  {row['s2_acc']:>8.5f}  {row['s3_acc']:>8.5f}"
          f"  {row['s1_rmse']:>9.5f}  {row['s2_rmse']:>9.5f}  {row['s3_rmse']:>9.5f}"
          f"  {row['s3_qdrift_pct']:>9.2f}%")
print("=" * W)

# ── final CSV save ────────────────────────────────────────────────────────────
df_final = pd.DataFrame(summary_rows)
df_final.to_csv(CSV_OUT, index=False)
print(f"\nFull results saved to: {CSV_OUT}")
print(f"  Columns: {list(df_final.columns)[:10]} ... ({len(df_final.columns)} total)")
