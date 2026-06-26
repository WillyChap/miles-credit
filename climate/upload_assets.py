#!/usr/bin/env python3
"""
upload_assets.py  (maintainer tool)
-----------------------------------
Create/populate the HuggingFace model repo that download_assets.py pulls from.
Run this ONCE (with the real assets staged in ./assets/) to publish the model.

    # on NCAR, stage the real files first:
    ./stage_assets.sh --copy            # copy (not symlink) into ./assets/
    python upload_assets.py --repo_id <user>/camulator --create

Requires:  pip install huggingface_hub   and   `huggingface-cli login`
"""
import argparse
import os
import sys

# Same manifest as download_assets.py — keep in sync.
REQUIRED = [
    "checkpoint.pt",
    "mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc",
    "std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc",
    "statics_b_credit_runs_f32_02.nc",
    "b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc",
    "f.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc",
    "b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc",
    "init_camulator_condition_tensor_1981-01-01T00Z.pth",
    "era5.yaml",
]
OPTIONAL = [
    "ERA5_clim_1990_2019_6h_interp.nc",
    "truth_be21_tensor_2013-01-01T00Z.pth",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True, help="e.g. myorg/camulator")
    ap.add_argument("--repo_type", default="model", choices=["model", "dataset"])
    ap.add_argument("--assets_dir", default=os.path.join(os.path.dirname(__file__), "assets"))
    ap.add_argument("--create", action="store_true", help="create the repo if it does not exist")
    ap.add_argument("--private", action="store_true", help="create as a private repo")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed.  ->  pip install huggingface_hub")

    api = HfApi(token=args.token)
    if args.create:
        api.create_repo(args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
        print(f"repo ready: {args.repo_type}:{args.repo_id}")

    wanted = REQUIRED + (OPTIONAL if args.include_optional else [])
    missing = [f for f in wanted if not os.path.exists(os.path.join(args.assets_dir, f))]
    if missing:
        print("Missing locally (stage them into ./assets/ first):")
        for f in missing:
            print("  -", f)
        sys.exit(1)

    for f in wanted:
        path = os.path.join(args.assets_dir, f)
        sz = os.path.getsize(path) / 1e6
        print(f"uploading {f} ({sz:.0f} MB) ...", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=f, repo_id=args.repo_id, repo_type=args.repo_type)
    print(f"\nDone. Set download_assets.py default --repo_id to {args.repo_id}")


if __name__ == "__main__":
    main()
