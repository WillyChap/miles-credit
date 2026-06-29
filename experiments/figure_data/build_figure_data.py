#!/usr/bin/env python3
"""Regenerate the cached data behind Figure 1 (drift) from the training logs.

Produces, in this directory:
  drift_broken.csv            <- parse_logs("camulator_ext_v2_finetune.o")  (feedback run)
  drift_fixed.csv             <- parse_logs("camulator_v2_consfix.o")        (decoupled + penalty, weight 0.1)
  truth_water_budget_1980.csv <- copy of <repo>/water_budget_1980.csv        (CAM6 ground truth)

Both drift runs are from the NEW_CLI_JOHN_CASPER_extended_v2 run; their logs live in
<repo>/rechunk_logs/. Parsing reuses plot_drift_comparison.parse_logs, which regex-extracts
each "WaterFixer step N | ... drift_pct=X%" line, averages the duplicate per-GPU-rank entries
at each step (groupby step), and derives epoch = start_epoch + (step-1)//150.

CAVEAT: the "epoch" column is approximate. It is computed from global_step/150 across chained
job files, so the fixed series can show epochs slightly beyond the last logged epoch. It is an
x-axis convenience, not exact training bookkeeping.

To cache a DIFFERENT run (e.g. a re-run E3 sweep with a new job-name prefix), change
LOGS_DIR / the prefixes below and the output names. Run from anywhere:
    python experiments/figure_data/build_figure_data.py
"""

import importlib.util
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

LOGS_DIR = os.path.join(REPO, "rechunk_logs")
PARSER_PATH = os.path.join(REPO, "plot_drift_comparison.py")
TRUTH_SRC = os.path.join(REPO, "water_budget_1980.csv")

# (output filename, log-family prefix) for the two drift trajectories
RUNS = [
    ("drift_broken.csv", "camulator_ext_v2_finetune.o"),
    ("drift_fixed.csv", "camulator_v2_consfix.o"),
]
COLS = ["epoch", "global_step", "drift_pct", "P_sink", "E_src", "residual", "dTWC_dt", "P_ratio"]


def _load_parser():
    """Import plot_drift_comparison.py (not a package) and point it at rechunk_logs/."""
    spec = importlib.util.spec_from_file_location("plot_drift_comparison", PARSER_PATH)
    pdc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdc)
    pdc.LOGS_DIR = LOGS_DIR
    return pdc


def main():
    pdc = _load_parser()
    for out_name, prefix in RUNS:
        df = pdc.parse_logs(prefix)
        if df.empty:
            raise SystemExit(f"no WaterFixer lines for prefix '{prefix}' in {LOGS_DIR}")
        out = os.path.join(HERE, out_name)
        df[COLS].to_csv(out, index=False)
        print(f"wrote {out_name:24s} rows={len(df):4d} "
              f"epochs {int(df.epoch.min())}-{int(df.epoch.max())}  (prefix {prefix})")

    truth_dst = os.path.join(HERE, "truth_water_budget_1980.csv")
    shutil.copyfile(TRUTH_SRC, truth_dst)
    print(f"copied truth_water_budget_1980.csv from {os.path.relpath(TRUTH_SRC, REPO)}")


if __name__ == "__main__":
    main()
