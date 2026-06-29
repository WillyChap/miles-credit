"""
eval_3step_baseline_cp092.py

3-step autoregressive eval for the ORIGINAL checkpoint.pt00092.pt.
Uses the credit_feb182026 codebase (not train_johns) — zero changes to that repo.

Purpose: sanity-check that cp00092 (pre-surgery) is still healthy and that the
extended config / fine-tuning process hasn't regressed the baseline.

Usage (interactive A100 node, credit-casper-mar2026 env):
    cd /glade/work/wchapman/Roman_Coupling/train_johns
    python eval_3step_baseline_cp092.py
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

# ── point at the OLD codebase, not train_johns ────────────────────────────────
OLD_REPO = "/glade/work/wchapman/Roman_Coupling/credit_feb182026"
sys.path.insert(0, OLD_REPO)

from credit.models import load_model
from credit.datasets.load_dataset_and_dataloader import load_dataset, load_dataloader
from credit.data import concat_and_reshape, reshape_only
from credit.metrics import LatWeightedMetrics
from credit.trainers.utils import cycle
from credit.parser import credit_main_parser

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

# suppress noisy sampler warning
class _SuppressEpochWarning(logging.Filter):
    def filter(self, record):
        return "set_epoch" not in record.getMessage()
logging.getLogger().addFilter(_SuppressEpochWarning())

# ── config / checkpoint ───────────────────────────────────────────────────────
CONFIG   = os.path.join(OLD_REPO, "climate", "camulator_config.yml")
CKPT_DIR = "/glade/derecho/scratch/wchapman/CREDIT_runs/wxformermod_sharp_SpatPS_pxshf"
CKPT     = os.path.join(CKPT_DIR, "checkpoint.pt00092.pt")
CSV_OUT  = os.path.join(CKPT_DIR, "eval_3step_cp092_baseline.csv")

N_SAMPLES = 25
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── channel layout (same prog layout, 15 diag instead of 17) ─────────────────
# x: [0:128] upper-air, [128:130] surface, [130:136] forcing+static
# y: [0:128] upper-air, [128:130] surface, [130:145] diagnostics (15)
QTOT_INDS = slice(96, 128)
PRECT_IND = 130

# ── load + parse config ───────────────────────────────────────────────────────
with open(CONFIG) as f:
    conf = yaml.safe_load(f)
conf["save_loc"] = CKPT_DIR
conf = credit_main_parser(conf)

# ── normalization stats ───────────────────────────────────────────────────────
ds_mean = xr.open_dataset(conf["data"]["mean_path"])
ds_std  = xr.open_dataset(conf["data"]["std_path"])

qtot_mean_t = torch.tensor(ds_mean["Qtot"].values, dtype=torch.float32).to(DEVICE).view(1, 32, 1, 1)
qtot_std_t  = torch.tensor(ds_std["Qtot"].values,  dtype=torch.float32).to(DEVICE).view(1, 32, 1, 1)
prect_mean  = float(ds_mean["PRECT"].values.flat[0])
prect_std   = float(ds_std["PRECT"].values.flat[0])

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CKPT}")
model = load_model(copy.deepcopy(conf))
ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
model = model.to(DEVICE).eval()
print(f"Model loaded on {DEVICE}")

# ── validation dataset / loader ───────────────────────────────────────────────
# prefetch_factor requires num_workers > 0; override for interactive eval
conf["trainer"]["valid_thread_workers"] = 2
conf["trainer"]["prefetch_factor"] = 2

valid_dataset = load_dataset(conf, rank=0, world_size=1, is_train=False)
valid_dataset.set_epoch(0)   # initializes forecast_step_counts / batch_indices
valid_loader  = load_dataloader(conf, valid_dataset, rank=0, world_size=1, is_train=False)
print(f"Validation batches: {len(valid_loader)}")

# ── metrics ───────────────────────────────────────────────────────────────────
metrics = LatWeightedMetrics(conf, training_mode=False)

# ── batch assembly (mirrors trainerERA5 exactly) ──────────────────────────────
def prepare_xy(batch, device):
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

# ── eval loop ─────────────────────────────────────────────────────────────────
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

print(f"\nRunning {N_SAMPLES} triplets on cp00092 baseline...")
print(f"  {'N':>5}  {'S1 ACC':>8}  {'S2 ACC':>8}  {'S3 ACC':>8}  "
      f"{'S1 RMSE':>9}  {'S2 RMSE':>9}  {'S3 RMSE':>9}  {'S3 Qdrift':>10}")
print("  " + "-" * 90)

with torch.no_grad():
    for n in range(N_SAMPLES):
        # t=1
        x_t0, y_t1_gt = prepare_xy(next(dl), DEVICE)
        y_hat_t1 = model(x_t0)

        # t=2
        x_t1_full, y_t2_gt = prepare_xy(next(dl), DEVICE)
        bs     = min(x_t1_full.shape[0], y_hat_t1.shape[0])
        n_prog = x_t1_full.shape[1] - 6   # 4 dyn forcing + 2 static

        x_t1_hat = x_t1_full[:bs].clone()
        x_t1_hat[:, :n_prog] = y_hat_t1[:bs, :n_prog]
        y_hat_t2 = model(x_t1_hat)

        # t=3
        x_t2_full, y_t3_gt = prepare_xy(next(dl), DEVICE)
        x_t2_hat = x_t2_full[:bs].clone()
        x_t2_hat[:, :n_prog] = y_hat_t2[:bs, :n_prog]
        y_hat_t3 = model(x_t2_hat)

        y_t1_gt = y_t1_gt[:bs];  y_hat_t1 = y_hat_t1[:bs]
        y_t2_gt = y_t2_gt[:bs];  y_hat_t2 = y_hat_t2[:bs]
        y_t3_gt = y_t3_gt[:bs];  y_hat_t3 = y_hat_t3[:bs]

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

        s1_prect.append(prect_mean_phys(y_hat_t1)); gt1_prect.append(prect_mean_phys(y_t1_gt))
        s2_prect.append(prect_mean_phys(y_hat_t2)); gt2_prect.append(prect_mean_phys(y_t2_gt))
        s3_prect.append(prect_mean_phys(y_hat_t3)); gt3_prect.append(prect_mean_phys(y_t3_gt))
        s1_qtot.append(qtot_column_mean(y_hat_t1)); gt1_qtot.append(qtot_column_mean(y_t1_gt))
        s2_qtot.append(qtot_column_mean(y_hat_t2)); gt2_qtot.append(qtot_column_mean(y_t2_gt))
        s3_qtot.append(qtot_column_mean(y_hat_t3)); gt3_qtot.append(qtot_column_mean(y_t3_gt))
        s1_qdrift.append(qtot_drift_frac(y_hat_t1, y_t1_gt))
        s2_qdrift.append(qtot_drift_frac(y_hat_t2, y_t2_gt))
        s3_qdrift.append(qtot_drift_frac(y_hat_t3, y_t3_gt))

        nn = n + 1
        if nn % 10 == 0 or nn == 1:
            print(f"  {nn:>5}  {np.mean(s1_acc):>8.5f}  {np.mean(s2_acc):>8.5f}  {np.mean(s3_acc):>8.5f}"
                  f"  {np.mean(s1_rmse):>9.5f}  {np.mean(s2_rmse):>9.5f}  {np.mean(s3_rmse):>9.5f}"
                  f"  {np.mean(s3_qdrift)*100:>9.2f}%")

# ── report ────────────────────────────────────────────────────────────────────
print(f"\n  checkpoint.pt00092  (credit_feb182026 baseline)")
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

print(f"\n  Per-variable RMSE — sorted by step-3 degradation:")
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

# ── save CSV ──────────────────────────────────────────────────────────────────
row = {
    "checkpoint":    "checkpoint.pt00092.pt",
    "codebase":      "credit_feb182026",
    "s1_acc":        np.mean(s1_acc),   "s2_acc":  np.mean(s2_acc),   "s3_acc":  np.mean(s3_acc),
    "s1_rmse":       np.mean(s1_rmse),  "s2_rmse": np.mean(s2_rmse),  "s3_rmse": np.mean(s3_rmse),
    "s1_mae":        np.mean(s1_mae),   "s2_mae":  np.mean(s2_mae),   "s3_mae":  np.mean(s3_mae),
    "s1_qdrift_pct": np.mean(s1_qdrift) * 100,
    "s2_qdrift_pct": np.mean(s2_qdrift) * 100,
    "s3_qdrift_pct": np.mean(s3_qdrift) * 100,
    "s1_prect": np.mean(s1_prect), "gt1_prect": np.mean(gt1_prect),
    "s2_prect": np.mean(s2_prect), "gt2_prect": np.mean(gt2_prect),
    "s3_prect": np.mean(s3_prect), "gt3_prect": np.mean(gt3_prect),
    "s1_qtot":  np.mean(s1_qtot),  "gt1_qtot":  np.mean(gt1_qtot),
    "s2_qtot":  np.mean(s2_qtot),  "gt2_qtot":  np.mean(gt2_qtot),
    "s3_qtot":  np.mean(s3_qtot),  "gt3_qtot":  np.mean(gt3_qtot),
}
for var in metrics.vars:
    row[f"rmse_{var}_s1"] = np.mean(s1_per_var[var])
    row[f"rmse_{var}_s2"] = np.mean(s2_per_var[var])
    row[f"rmse_{var}_s3"] = np.mean(s3_per_var[var])

pd.DataFrame([row]).to_csv(CSV_OUT, index=False)
print(f"\nResults saved to: {CSV_OUT}")
