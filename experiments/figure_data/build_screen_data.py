#!/usr/bin/env python3
"""Cache the E3 controlled-screen drift trajectories for Figure 1b/1c.

Parses the four 2x2 ablation job logs, all fine-tuned from the same cp00092_extended
checkpoint, for the WaterFixer per-step drift and global precip sink, tracking the
training epoch from the interleaved tqdm progress. Averages the per-GPU-rank entries at
each step. The 2x2 crosses supervised-loss placement (corrected vs raw, the v1 trainer's
supervise_precorrection gate) with the imbalance penalty weight (0.0 vs 0.1):

    A  corrected supervision, no penalty   (e3_v1_w0.0,     5032824)  -> the broken cell
    B  corrected supervision, penalty 0.1  (e3_B_corr_w0.1, 5035892)
    C  raw supervision,       no penalty   (e3_C_raw_w0.0,  5035893)
    D  raw supervision,       penalty 0.1  (e3_v1_w0.1,     5032826)

Writes screen_A.csv .. screen_D.csv (and screen_broken.csv/screen_fix.csv as aliases for
A and D, kept for older figure scripts). Re-run after the jobs finish:
    python experiments/figure_data/build_screen_data.py
"""
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGDIR = os.path.join(REPO, "experiments", "logs")

# (output name, job id, label)
CELLS = [
    ("screen_A.csv", "5032824", "A: corrected loss, no penalty"),
    ("screen_B.csv", "5035892", "B: corrected loss, penalty 0.1"),
    ("screen_C.csv", "5035893", "C: raw loss, no penalty"),
    ("screen_D.csv", "5032826", "D: raw loss, penalty 0.1"),
    # aliases for older scripts
    ("screen_broken.csv", "5032824", "A (alias: broken)"),
    ("screen_fix.csv", "5032826", "D (alias: fix)"),
]
BATCHES_PER_EPOCH = 150
RE_EPOCH = re.compile(r"Epoch:\s*(\d+).*?(\d+)/%d" % BATCHES_PER_EPOCH)
RE_WF = re.compile(r"WaterFixer step (\d+) \|.*?P_sink=([-\d.eE+]+).*?drift_pct=([-\d.]+)%")


def parse_cell(jobid):
    # the PBS spool name is <jobid>.casper-pbs.OU
    path = None
    for cand in (f"{jobid}.casper-pbs.OU", f"e3_v1_w0.0.o{jobid}", f"{jobid}.OU"):
        p = os.path.join(LOGDIR, cand)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise SystemExit(f"no log found for job {jobid} in {LOGDIR}")

    cur_ep, cur_frac = 0, 0.0
    rows = []
    txt = open(path, errors="ignore").read()
    for tok in re.split(r"[\r\n]", txt):
        e = RE_EPOCH.search(tok)
        if e:
            cur_ep, cur_frac = int(e.group(1)), int(e.group(2)) / BATCHES_PER_EPOCH
        w = RE_WF.search(tok)
        if w:
            rows.append((cur_ep + cur_frac, int(w.group(1)),
                         float(w.group(2)), float(w.group(3))))
    df = pd.DataFrame(rows, columns=["epoch", "step", "P_sink", "drift_pct"])
    if df.empty:
        return df
    # average the per-rank duplicates at each WaterFixer step
    return (df.groupby("step").agg(epoch=("epoch", "mean"), P_sink=("P_sink", "mean"),
                                   drift_pct=("drift_pct", "mean")).reset_index()
              .sort_values("step"))


def main():
    for out, jobid, label in CELLS:
        df = parse_cell(jobid)
        path = os.path.join(HERE, out)
        df.to_csv(path, index=False)
        if df.empty:
            print(f"{out}: no WaterFixer lines yet (job {jobid})")
        else:
            print(f"{out}: {len(df)} pts, epoch {df.epoch.min():.2f}-{df.epoch.max():.2f}, "
                  f"drift {df.drift_pct.iloc[0]:.1f}->{df.drift_pct.iloc[-1]:.1f}%  [{label}]")


if __name__ == "__main__":
    main()
