#!/usr/bin/env python3
"""Cache the per-fixer correction magnitudes from the feedback-run logs for Figure 3.

Parses the camulator_ext_v2_finetune.o* logs in rechunk_logs/ for the WaterFixer, MassFixer,
and EnergyFixer per-step drift diagnostics (|r-1| in percent), averages duplicate per-GPU-rank
entries at each step, derives an epoch from the step counter, and writes
fixer_control_broken.csv next to this script. Run from the repo root:
    python experiments/figure_data/build_fixer_control_data.py
"""
import glob, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGS = sorted(glob.glob(os.path.join(REPO, "rechunk_logs", "camulator_ext_v2_finetune.o*")))
STEPS_PER_EPOCH = 150
PAT_EP = re.compile(r"Beginning epoch (\d+)")
PATS = {
    "water": re.compile(r"WaterFixer step (\d+).*?drift_pct=([-\d.]+)%"),
    "mass": re.compile(r"MassFixer step (\d+).*?drift_pct=([-\d.]+)%"),
    "energy": re.compile(r"EnergyFixer step (\d+).*?\(([-\d.]+)%\)"),
}


def main():
    recs = []
    for f in LOGS:
        txt = open(f, errors="ignore").read()
        eps = [int(m.group(1)) for m in PAT_EP.finditer(txt)]
        if not eps:
            continue
        se = min(eps)
        for fixer, pat in PATS.items():
            for m in pat.finditer(txt):
                step = int(m.group(1))
                recs.append((fixer, se * STEPS_PER_EPOCH + step, se + (step - 1) // STEPS_PER_EPOCH,
                             abs(float(m.group(2)))))
    df = pd.DataFrame(recs, columns=["fixer", "global_step", "epoch", "absdrift"])
    df = df.groupby(["fixer", "global_step", "epoch"]).absdrift.mean().reset_index()
    piv = (df.pivot_table(index=["global_step", "epoch"], columns="fixer", values="absdrift")
             .reset_index().sort_values("global_step"))
    out = os.path.join(HERE, "fixer_control_broken.csv")
    piv.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(piv)}  epochs {int(piv.epoch.min())}-{int(piv.epoch.max())}")


if __name__ == "__main__":
    main()
