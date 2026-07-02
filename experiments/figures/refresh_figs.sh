#!/bin/bash
# "Hit go" for the paper's data figures: re-parse the logs, rebuild the two data figures,
# and install them into the LaTeX paper. Run after the E3 screen cells finish.
#
# Paper figure layout:
#   Figure 1 = schematic (fig01_schematic.png, exported from fig01_schematic.pptx; not built here)
#   Figure 2 = fine-tuning run   -> plot_fig_finetune.py  -> fig02_finetune.pdf
#   Figure 3 = 2x2 ablation      -> plot_fig_ablation.py  -> fig03_ablation.pdf
set -euo pipefail
REPO=/glade/work/wchapman/Roman_Coupling/train_johns
cd "$REPO"
python experiments/figure_data/build_screen_data.py
python experiments/figure_data/build_fixer_control_data.py
python experiments/figures/plot_fig_finetune.py
python experiments/figures/plot_fig_ablation.py
cp experiments/figures/fig_finetune.pdf papers/lessons_learned_latex/fig02_finetune.pdf
cp experiments/figures/fig_ablation.pdf papers/lessons_learned_latex/fig03_ablation.pdf
echo "Figs 2 and 3 refreshed and installed. Recompile:  (cd papers/lessons_learned_latex && latexmk -pdf main)"
