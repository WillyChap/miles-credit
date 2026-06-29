#!/usr/bin/env python3
"""Build the v2-hybrid E3 config: v2 data schema + era5-v2 trainer (from alt_v2_penalty)
with the 128-dim model block swapped in from recommended_v1_decouple so it matches the
common-start checkpoint cp00092_extended. Same variables, so channel layout is identical.

  python make_v2hybrid.py --weight 0.0 --save_loc /scratch/.../e3_v2_w0.0/ \
      --num_epoch 4 --batches_per_epoch 150 --out generated/e3_v2_w0.0.yml
  python make_v2hybrid.py --loadtest        # build model + load ckpt on CPU, no write
"""
import argparse
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

V1 = os.path.join(HERE, "recommended_v1_decouple.train.yml")
V2 = os.path.join(HERE, "alt_v2_penalty.train.yml")
CKPT = "/glade/derecho/scratch/wchapman/CREDIT_runs/wxformermod_sharp_SpatPS_pxshf/checkpoint.pt00092_extended.pt"


def build(weight, save_loc, num_epoch, batches_per_epoch):
    v1 = yaml.safe_load(open(V1))
    h = yaml.safe_load(open(V2))          # base: v2 source schema + era5-v2 trainer
    h["model"] = copy.deepcopy(v1["model"])  # 128-dim model (incl post_conf) matching ckpt
    h["save_loc"] = save_loc
    tr = h["trainer"]
    tr["load_weights"] = True
    tr["reload_epoch"] = False
    tr["num_epoch"] = int(num_epoch)
    tr["batches_per_epoch"] = int(batches_per_epoch)
    h["model"]["post_conf"]["global_water_fixer"]["conservation_loss_weight"] = float(weight)
    return h


def loadtest():
    import torch
    from credit.parser import credit_main_parser
    from credit.models import load_model

    conf = build(0.0, "/tmp/v2hybrid_test/", 1, 4)
    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)
    print("[ok] parser passed (no KeyError 'source')")
    model = load_model(conf, load_weights=False)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    n = sum(p.numel() for p in model.parameters())
    print(f"[ok] cp00092_extended loaded into the hybrid model ({n/1e6:.1f}M params, strict load)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loadtest", action="store_true")
    p.add_argument("--weight", type=float)
    p.add_argument("--save_loc")
    p.add_argument("--num_epoch", type=int, default=4)
    p.add_argument("--batches_per_epoch", type=int, default=150)
    p.add_argument("--out")
    a = p.parse_args()
    if a.loadtest:
        loadtest()
        return
    h = build(a.weight, a.save_loc, a.num_epoch, a.batches_per_epoch)
    with open(a.out, "w") as f:
        yaml.safe_dump(h, f, default_flow_style=False, sort_keys=False)
    print(f"wrote {a.out}  (weight={a.weight}, era5-v2, 128-dim, save_loc={a.save_loc})")


if __name__ == "__main__":
    main()
