#!/usr/bin/env bash
# Run SnareNet experiments with different lambd values in parallel using xargs
# Usage: 
# bash run_lambd_multi.sh > commands
# cat commands | xargs -n1 -P4 -I{} /bin/sh -c "{}"

# Change to the root directory of the repository
cd "$(dirname "$0")/.." || exit 1

# Configuration
SEEDS="456 789 123 101 112"
DATASET="noncvx"
DATE=$(date +%b%d)
WANDB_PROJECT="${DATASET}_lambd_${DATE}"

# Define the GPU to use
GPU="0"

# Define lambd values to test
LAMBD_VALUES="1e-3 1e-4 1e-1 1.0 5.0 10.0"

# Generate commands for all experiments
for SEED in $SEEDS; do
    for LAMBD in $LAMBD_VALUES; do
        # Run SnareNet with different lambd values
        echo CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
            +experiment=lambd_$DATASET \
            dataset=$DATASET \
            model=snarenet \
            seed=$SEED \
            wandb_project=$WANDB_PROJECT \
            model.lambd=$LAMBD
    done
done
