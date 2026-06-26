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

# Files expected in ./assets/ (basenames the config points at). Optional ones are
# only needed for rollout_metrics / fast-climate scoring, not a plain run.
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

# Default HuggingFace repo holding the assets. Override with --repo_id or the
# CAMULATOR_HF_REPO environment variable.
DEFAULT_REPO = os.environ.get("CAMULATOR_HF_REPO", "NCAR/camulator")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default=DEFAULT_REPO, help=f"HuggingFace repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--repo_type", default="model", choices=["model", "dataset"], help="HF repo type")
    ap.add_argument("--revision", default=None, help="Branch/tag/commit to pull (default: main)")
    ap.add_argument("--assets_dir", default=os.path.join(os.path.dirname(__file__), "assets"),
                    help="Where to place the files (default: ./assets)")
    ap.add_argument("--include_optional", action="store_true", help="Also fetch the optional scoring files")
    ap.add_argument("--token", default=None, help="HF token for private repos (else uses cached login)")
    args = ap.parse_args()

    # Guard against the placeholder repo id (the HF repo must exist first).
    if "<" in args.repo_id or args.repo_id == "NCAR/camulator":
        sys.exit(
            f"--repo_id '{args.repo_id}' is a placeholder. Pass the real HuggingFace repo, e.g.\n"
            "    python download_assets.py --repo_id myorg/camulator\n"
            "(Maintainers: create+populate it with upload_assets.py first.)"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed.  ->  pip install huggingface_hub")

    os.makedirs(args.assets_dir, exist_ok=True)
    wanted = REQUIRED + (OPTIONAL if args.include_optional else [])
    print(f"Downloading {len(wanted)} files from {args.repo_type}:{args.repo_id} -> {args.assets_dir}")

    failed = []
    for fname in wanted:
        try:
            path = hf_hub_download(
                repo_id=args.repo_id, filename=fname, repo_type=args.repo_type,
                revision=args.revision, token=args.token,
                local_dir=args.assets_dir, local_dir_use_symlinks=False,
            )
            print(f"  ok    {fname}")
        except Exception as e:
            failed.append((fname, str(e)))
            print(f"  FAIL  {fname}: {e}")

    if failed:
        print(f"\n{len(failed)} file(s) failed. Check the repo id / filenames / access token.")
        sys.exit(1)
    print("\nAll assets present in", args.assets_dir, "- ready to run RunQuickClimate.sh")


if __name__ == "__main__":
    main()
