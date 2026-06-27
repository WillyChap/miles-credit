#!/usr/bin/env python3
"""
download_assets.py
------------------
Fetch the CAMulator inference assets (model checkpoint, normalization, statics,
forcing, initial condition, metadata) from a HuggingFace repo into ./assets/.

The config (camulator_config.yml) reads every runtime input from ./assets/<file>,
so once this finishes the toolbox is ready to run:

    python download_assets.py --repo_id <user>/<repo>
    bash RunQuickClimate.sh

On NCAR you can instead symlink the GLADE copies with ./stage_assets.sh.

Requires:  pip install huggingface_hub
"""
import argparse
import os
import sys

# Shared inputs every run needs (basenames the config points at), EXCEPT the
# model checkpoint — there are many checkpoints (epochs), so you pick which
# one(s) to download with --checkpoint/--checkpoints. Optional files are only
# needed for rollout_metrics / fast-climate scoring, not a plain run.
SHARED = [
    "mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc",
    "std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc",
    "statics_b_credit_runs_f32_02.nc",
    "b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc",
    "f.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc",
    "b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc",
    "init_camulator_condition_tensor_1981-01-01T00Z.pth",
]
OPTIONAL = [
    "ERA5_clim_1990_2019_6h_interp.nc",
    "truth_be21_tensor_2013-01-01T00Z.pth",
    # progressive/transient forcing (1980-2014, slim ~9.7 GB) — alternative to
    # the cyclic forcing; only needed for transient-climate runs.
    "b.e21.CREDIT_climate_branch_1980_2014.nc",
]

# Default HuggingFace repo holding the assets. Override with --repo_id or the
# CAMULATOR_HF_REPO environment variable.
DEFAULT_REPO = os.environ.get("CAMULATOR_HF_REPO", "willychap/camulator")


def repo_path(name):
    """Map a flat asset basename to its path in the HF repo (ACE2-style tree).
    Kept in sync with upload_assets.py."""
    if name.startswith("checkpoint.pt"):
        return name
    if "cyclic" in name or "branch_1980_2014" in name:
        return f"forcing_data/{name}"
    if name.startswith("init_camulator_condition_tensor"):
        return f"initial_conditions/{name}"
    if name == "era5.yaml":
        return f"metadata/{name}"
    return f"normalization/{name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default=DEFAULT_REPO, help=f"HuggingFace repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--repo_type", default="model", choices=["model", "dataset"], help="HF repo type")
    ap.add_argument("--revision", default=None, help="Branch/tag/commit to pull (default: main)")
    ap.add_argument("--assets_dir", default=os.path.join(os.path.dirname(__file__), "assets"),
                    help="Where to place the files (default: ./assets)")
    ap.add_argument("--checkpoint", default="checkpoint.pt00065.pt",
                    help="checkpoint file to fetch from the repo (default: checkpoint.pt00065.pt). "
                         "Set the same name as MODEL_NAME when you run.")
    ap.add_argument("--checkpoints", nargs="+", default=None,
                    help="fetch several checkpoints at once (overrides --checkpoint)")
    ap.add_argument("--no_checkpoint", action="store_true",
                    help="fetch only the shared inputs, no checkpoint")
    ap.add_argument("--include_optional", action="store_true", help="Also fetch the optional scoring files")
    ap.add_argument("--token", default=None, help="HF token for private repos (else uses cached login)")
    args = ap.parse_args()

    # Guard against the placeholder repo id (the HF repo must exist first).
    if "<" in args.repo_id:
        sys.exit(
            f"--repo_id '{args.repo_id}' is a placeholder. Pass the real HuggingFace repo, e.g.\n"
            "    python download_assets.py --repo_id willychap/camulator\n"
            "(Maintainers: create+populate it with upload_assets.py first.)"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed.  ->  pip install huggingface_hub")

    os.makedirs(args.assets_dir, exist_ok=True)
    ckpts = [] if args.no_checkpoint else (args.checkpoints or [args.checkpoint])
    wanted = ckpts + SHARED + (OPTIONAL if args.include_optional else [])
    print(f"Downloading {len(wanted)} files from {args.repo_type}:{args.repo_id} -> {args.assets_dir}")
    if ckpts:
        print(f"  checkpoint(s): {', '.join(ckpts)}  (set MODEL_NAME to match when you run)")

    import shutil
    failed = []
    for fname in wanted:
        rp = repo_path(fname)   # subdir path in the HF repo (ACE2-style tree)
        try:
            path = hf_hub_download(
                repo_id=args.repo_id, filename=rp, repo_type=args.repo_type,
                revision=args.revision, token=args.token,
                local_dir=args.assets_dir, local_dir_use_symlinks=False,
            )
            # flatten subdir -> ./assets/<basename> so the config's ./assets/<file> paths resolve
            dest = os.path.join(args.assets_dir, fname)
            if os.path.abspath(path) != os.path.abspath(dest):
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                shutil.move(path, dest)
            print(f"  ok    {fname}")
        except Exception as e:
            failed.append((fname, str(e)))
            print(f"  FAIL  {fname}: {e}")
    # tidy any now-empty subdirs the download created
    for d in ("forcing_data", "initial_conditions", "normalization", "metadata"):
        dp = os.path.join(args.assets_dir, d)
        if os.path.isdir(dp) and not os.listdir(dp):
            os.rmdir(dp)

    if failed:
        print(f"\n{len(failed)} file(s) failed. Check the repo id / filenames / access token.")
        sys.exit(1)
    print("\nAll assets present in", args.assets_dir, "- ready to run RunQuickClimate.sh")


if __name__ == "__main__":
    main()
