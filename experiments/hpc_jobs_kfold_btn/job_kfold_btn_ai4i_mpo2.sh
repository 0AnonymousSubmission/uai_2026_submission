#!/bin/sh

echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"


cd $HOME/BMPO
source .venv/bin/activate

python -m experiments.run_kfold_btn \
    --specs-file experiments/configs/kfold_best_specs_btn.json \
    --model MPO2 \
    --datasets ai4i \
    --output-dir results/kfold_btn/ai4i_MPO2 \
    --tracker-dir runs_kfold_btn/ai4i_MPO2 \
    --n-folds 5 \
    --seeds 42 9536 47659 19540 42806 \
    --n-epochs 100 \
    --patience 100 \
    --batch-size 256 \
    --tracker file

echo "Done: $(date +%F-%R:%S)"
