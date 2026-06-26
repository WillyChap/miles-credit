#!/usr/bin/env python3
"""
upload_assets.py  (maintainer tool)
-----------------------------------
Create/populate the HuggingFace model repo that download_assets.py pulls from.
Run this ONCE (with the real assets staged in ./assets/) to publish the model.

    # on NCAR, stage the real files first:
    ./stage_assets.sh --copy            # copy (not symlink) into ./assets/
    python upload_assets.py --repo_id willychap/camulator --create

Requires:  pip install huggingface_hub   and   `huggingface-cli login`
"""
import argparse
import os
import sys

# Shared inputs (keep in sync with download_assets.py SHARED). Checkpoints are
# uploaded separately via --checkpoints since there are many (epochs).
SHARED = [
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
    "b.e21.CREDIT_climate_branch_1980_2014.nc",   # progressive/transient forcing (slim, ~10 GB)
]

HERE = os.path.dirname(os.path.abspath(__file__))


def repo_path(name):
    """Map a flat asset basename to its path in the HF repo (ACE2-style tree)."""
    if name.startswith("checkpoint.pt"):
        return name                                   # checkpoints at repo root
    if "cyclic" in name or "branch_1980_2014" in name:
        return f"forcing_data/{name}"
    if name.startswith("init_camulator_condition_tensor"):
        return f"initial_conditions/{name}"
    if name == "era5.yaml":
        return f"metadata/{name}"
    return f"normalization/{name}"                    # mean/std/statics/lat-weights/clim/truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", required=True, help="e.g. willychap/camulator")
    ap.add_argument("--repo_type", default="model", choices=["model", "dataset"])
    ap.add_argument("--assets_dir", default=os.path.join(os.path.dirname(__file__), "assets"))
    ap.add_argument("--create", action="store_true", help="create the repo if it does not exist")
    ap.add_argument("--private", action="store_true", help="create as a private repo")
    ap.add_argument("--include_optional", action="store_true")
    ap.add_argument("--checkpoints", nargs="*", default=["checkpoint.pt00065.pt"],
                    help="checkpoint files to upload (names or paths; default checkpoint.pt00065.pt). "
                         "Many epochs are fine, e.g. --checkpoints checkpoint.pt000{40..79}.pt")
    ap.add_argument("--checkpoint_dir", default=None,
                    help="directory the --checkpoints names live in (default: --assets_dir). Lets you "
                         "upload straight from the training run dir without staging into ./assets/.")
    ap.add_argument("--skip_shared", action="store_true",
                    help="upload only the checkpoints (e.g. to add more epochs to an existing repo)")
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

    # Resolve what to upload to (local_path, name_in_repo) pairs.
    uploads = []
    if not args.skip_shared:
        # model card -> README.md, config -> inference_config.yaml (ACE2-style)
        uploads.append((os.path.join(HERE, "MODEL_CARD.md"), "README.md"))
        uploads.append((os.path.join(HERE, "camulator_config.yml"), "inference_config.yaml"))
        for f in SHARED + (OPTIONAL if args.include_optional else []):
            uploads.append((os.path.join(args.assets_dir, f), repo_path(f)))
    ckpt_dir = args.checkpoint_dir or args.assets_dir
    for c in args.checkpoints or []:
        src = c if os.path.isabs(c) else os.path.join(ckpt_dir, c)
        uploads.append((src, repo_path(os.path.basename(c))))

    missing = [p for p, _ in uploads if not os.path.exists(p)]
    if missing:
        print("Missing locally:")
        for p in missing:
            print("  -", p)
        sys.exit(1)

    for path, name in uploads:
        sz = os.path.getsize(path) / 1e6
        print(f"uploading {name} ({sz:.0f} MB) ...", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=args.repo_id, repo_type=args.repo_type)
    print(f"\nDone ({len(uploads)} files). Users pick a checkpoint with "
          f"`download_assets.py --repo_id {args.repo_id} --checkpoint <name>`.")


if __name__ == "__main__":
    main()
