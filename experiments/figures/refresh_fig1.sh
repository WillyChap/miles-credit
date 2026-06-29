#!/bin/bash
# "Hit go" for Figure 1: re-parse the screen logs, rebuild the 3-panel figure, install it
# into the LaTeX paper. Run after the E3 screen cells (5032824 broken, 5032826 fix) finish.
set -euo pipefail
REPO=/glade/work/wchapman/Roman_Coupling/train_johns
cd "$REPO"
python experiments/figure_data/build_screen_data.py
python experiments/figures/plot_fig1.py
cp experiments/figures/fig1.pdf papers/lessons_learned_latex/fig01_drift.pdf
echo "Fig 1 refreshed and installed. Recompile:  (cd papers/lessons_learned_latex && latexmk -pdf main)"
