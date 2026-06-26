#!/usr/bin/env python3
"""
check_setup.py
--------------
Preflight check for CAMulator inference. Verifies — WITHOUT running a rollout —
that the environment and assets are ready, so you catch problems in seconds
instead of minutes into a run.

    python check_setup.py            # uses ./camulator_config.yml
    python check_setup.py -c my.yml

Checks: Python deps importable, CREDIT importable + config parses, every input
file the rollout needs exists, and whether a CUDA GPU is visible.
"""
import argparse
import os
import sys

OK = "  \033[32mok\033[0m  "
BAD = "  \033[31mFAIL\033[0m"
WARN = "  \033[33mwarn\033[0m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="./camulator_config.yml")
    ap.add_argument("--model_name", default="checkpoint.pt00065.pt", help="checkpoint filename inside save_loc")
    args = ap.parse_args()
    fails = 0

    # 1. core python deps
    for mod in ("torch", "xarray", "numpy", "yaml", "cftime", "netCDF4"):
        try:
            __import__(mod)
            print(f"{OK} import {mod}")
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} import {mod}: {e}")
            fails += 1

    # 2. CREDIT importable — exercise the FULL inference import chain that
    #    Model_State.initialize_camulator uses, not just credit.parser. This is
    #    what catches a broken torch/torch_harmonics combo (credit.models pulls
    #    the torch_harmonics C extension, which is ABI-tied to the torch version).
    try:
        import credit  # noqa: F401
        from credit.parser import credit_main_parser  # noqa: F401
        from credit.models import load_model, load_model_name  # noqa: F401
        from credit.distributed import distributed_model_wrapper  # noqa: F401
        from credit.postblock import GlobalMassFixer  # noqa: F401
        print(f"{OK} import credit (full inference chain) from {os.path.dirname(sys.modules['credit'].__file__)}")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} import credit: {type(e).__name__}: {e}")
        if "parseSchema" in str(e) or "undefined symbol" in str(e) or "torch_harmonics" in str(e):
            print("       -> torch / torch_harmonics ABI mismatch. Use the pinned combo in")
            print("          environment.yml (torch 2.4.1 + torch_harmonics 0.7.2), not torch 2.6.")
        else:
            print("       -> install THIS repo's CREDIT:  pip install -e . --no-deps  (from repo root)")
            print("          (use --no-deps; environment.yml already pins the runtime stack)")
        fails += 1

    # 3. config exists + parses with the CREDIT parser
    conf = None
    if not os.path.exists(args.config):
        print(f"{BAD} config not found: {args.config}")
        fails += 1
    else:
        try:
            import yaml
            from credit.parser import credit_main_parser
            conf = yaml.safe_load(open(args.config))
            credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
            print(f"{OK} config parses: {args.config}")
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} config parse: {e}")
            fails += 1

    # 4. required input files exist
    if conf is not None:
        req = {
            "checkpoint": os.path.join(conf.get("save_loc", "./assets/"), args.model_name),
            "mean_path": conf["data"]["mean_path"],
            "std_path": conf["data"]["std_path"],
            "save_loc_physics (statics)": conf["data"]["save_loc_physics"],
            "latitude_weights": conf["loss"]["latitude_weights"],
            "forcing_file": conf["predict"]["forcing_file"],
            "init_cond_fast_climate (IC)": conf["predict"]["init_cond_fast_climate"],
            "metadata": conf["predict"]["metadata"],
        }
        # post-block conservation fixers read their own statics (hybrid-sigma
        # coeffs); a rollout fails mid-run if this is missing, so check it too.
        try:
            pc = conf["model"]["post_conf"]
            for fixer in ("global_mass_fixer", "global_water_fixer", "global_energy_fixer_updown"):
                p = pc.get(fixer, {}).get("save_loc_physics")
                if p:
                    req[f"post_conf {fixer} statics"] = p
                    break
        except Exception:  # noqa: BLE001
            pass
        for label, path in req.items():
            if path and os.path.exists(path):
                print(f"{OK} {label}: {path}")
            else:
                print(f"{BAD} {label} MISSING: {path}")
                print("       -> stage assets:  python download_assets.py --repo_id willychap/camulator")
                fails += 1

    # 5. GPU (recommended, not strictly required)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"{OK} CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"{WARN} no CUDA GPU visible — rollout will be slow on CPU")
    except Exception:  # noqa: BLE001
        pass

    print()
    if fails:
        print(f"\033[31m{fails} check(s) failed.\033[0m Fix the above, then re-run.")
        sys.exit(1)
    print("\033[32mAll checks passed — ready to run RunQuickClimate.sh\033[0m")


if __name__ == "__main__":
    main()
