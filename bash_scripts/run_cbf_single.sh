#!/usr/bin/env bash
# Run CBF comparison experiments sequentially
# Usage:
# bash run_cbf_single.sh

# Change to the root directory of the repository
cd "$(dirname "$0")/.." || exit 1

# Configuration
DATASET="cbf"
DATE=$(date +%b%d)
WANDB_PROJECT="cbf_base_comparison_${DATE}"

# Define the GPU to use
GPU="0"

# Define (dataset_seed, seed) pairs
EXPERIMENTS="8:123 9:123"

# Run all experiments sequentially
for EXP in $EXPERIMENTS; do
    DATASET_SEED="${EXP%%:*}"
    SEED="${EXP##*:}"

    # Run DC3
    CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
        +experiment=dc3_$DATASET \
        dataset=$DATASET \
        dataset.seed=$DATASET_SEED \
        model=dc3 \
        seed=$SEED \
        wandb_project=$WANDB_PROJECT

    # Run SnareNet
    CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
        +experiment=snarenet_$DATASET \
        dataset=$DATASET \
        dataset.seed=$DATASET_SEED \
        model=snarenet \
        seed=$SEED \
        wandb_project=$WANDB_PROJECT
done
