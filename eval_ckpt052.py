"""
Single-step evaluation over 1 year of validation data using credit's
latitude-weighted RMSE and ACC (LatWeightedMetrics).

Loads checkpoint.pt00052.pt, runs single steps over 2013,
reports per-variable and mean lat-weighted RMSE, MAE, ACC.

Usage (interactive A100 node, credit-casper-mar2026 env):
    python eval_ckpt052.py
"""

import os
import sys
import copy
import logging
import numpy as np
import torch
import yaml
import torch.nn as nn

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from credit.models import load_model
from credit.datasets.load_dataset_and_dataloader import load_dataset, load_dataloader
from credit.preblock import ERA5Normalizer, ConcatPreblock
from credit.metrics import LatWeightedMetrics

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

# ── config / checkpoint ───────────────────────────────────────────────────────
CONFIG = "/glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_casper.yml"
CKPT     = "/glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_small/checkpoint.pt00052.pt"
USE_EMA  = True   # set False to use raw checkpoint weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load config (no parser — source schema) ───────────────────────────────────
with open(CONFIG) as f:
    conf = yaml.safe_load(f)
conf["save_loc"] = os.path.dirname(CKPT)

# restrict validation to 2013 only
conf["data_valid"]["start_datetime"] = "2013-01-01"
conf["data_valid"]["end_datetime"]   = "2013-12-31"

# ── build & load model ────────────────────────────────────────────────────────
model = load_model(copy.deepcopy(conf))
ckpt  = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)

if USE_EMA:
    ema_path = os.path.join(os.path.dirname(CKPT), "checkpoint_ema.pt")
    ema_ckpt = torch.load(ema_path, map_location="cpu", weights_only=False)
    shadow = ema_ckpt["model_state_dict"]["shadow"]
    # weight_u / weight_v are spectral-norm power-iteration buffers — not true weights.
    # EMA of unit vectors loses unit-norm, corrupting spectral norm. Copy from raw ckpt.
    spectral_bufs = {k for k in shadow if k.endswith(("weight_u", "weight_v", "weight_orig"))}
    merged = dict(shadow)
    for k in spectral_bufs:
        if k in ckpt["model_state_dict"]:
            merged[k] = ckpt["model_state_dict"][k]
    model.load_state_dict(merged, strict=True)
    step = ema_ckpt["model_state_dict"].get("step", "?")
    weight_label = f"EMA+spectral_fix  (step={step})"
else:
    weight_label = os.path.basename(CKPT)

model = model.to(DEVICE).eval()
print(f"Loaded: {weight_label}  →  {DEVICE}")

# ── validation dataset & dataloader (uses credit's own loader) ────────────────
valid_dataset = load_dataset(conf, rank=0, world_size=1, is_train=False)
valid_loader  = load_dataloader(conf, valid_dataset, rank=0, world_size=1, is_train=False)
print(f"Validation steps: {len(valid_loader)}")

# ── preblock ──────────────────────────────────────────────────────────────────
preblocks = nn.ModuleDict({
    "norm":   ERA5Normalizer(conf),
    "concat": ConcatPreblock(),
}).eval()

# ── lat-weighted metrics ──────────────────────────────────────────────────────
metrics = LatWeightedMetrics(conf, training_mode=False)

# ── evaluation loop ───────────────────────────────────────────────────────────
acc_list  = []
rmse_list = []
mae_list  = []
per_var   = {}

with torch.no_grad():
    for step, batch in enumerate(valid_loader):
        for pb in preblocks.values():
            batch = pb(batch)

        x    = batch["x"].float().to(DEVICE)
        y    = batch["y"].float().to(DEVICE)
        pred = model(x)

        result = metrics(pred, y)

        acc_list.append(result["acc"])
        rmse_list.append(float(result["rmse"]))
        mae_list.append(float(result["mae"]))

        for var in metrics.vars:
            r = result[f"rmse_{var}"]
            a = result[f"acc_{var}"]
            per_var.setdefault(f"rmse_{var}", []).append(r.cpu().item() if torch.is_tensor(r) else float(r))
            per_var.setdefault(f"acc_{var}",  []).append(a.cpu().item() if torch.is_tensor(a) else float(a))

        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{len(valid_loader)}  rmse={np.mean(rmse_list):.5f}  acc={np.mean(acc_list):.5f}")

# ── report ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print(f"  Weights    : {weight_label}")
print(f"  Period     : 2013 ({len(acc_list)} steps, lat-weighted)")
print("=" * 64)
print(f"  Mean ACC   : {np.mean(acc_list):.6f}")
print(f"  Mean RMSE  : {np.mean(rmse_list):.6f}")
print(f"  Mean MAE   : {np.mean(mae_list):.6f}")
print()
print(f"  {'Variable':<24}  {'RMSE (mean)':>12}  {'ACC (mean)':>12}")
print(f"  {'-'*24}  {'-'*12}  {'-'*12}")
for var in metrics.vars:
    r = np.mean(per_var[f"rmse_{var}"])
    a = np.mean(per_var[f"acc_{var}"])
    print(f"  {var:<24}  {r:>12.6f}  {a:>12.6f}")
print("=" * 64)
