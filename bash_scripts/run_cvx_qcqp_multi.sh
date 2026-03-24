#!/usr/bin/env bash
# Run comparison experiments in parallel using xargs
# Usage: 
# bash run_cvx_qcqp_multi.sh > commands
# cat commands | xargs -n1 -P4 -I{} /bin/sh -c "{}"

# Change to the root directory of the repository
cd "$(dirname "$0")/.." || exit 1

# Configuration
SEEDS="456 789 123 101 112"
DATASET="cvx_qcqp"
DATE=$(date +%b%d)
WANDB_PROJECT="${DATASET}_comparison_${DATE}"

# Define the GPU to use
GPU="0"

# Define rtol values to test
RTOL_VALUES="1e-4 1e-6 1e-8 1e-10 1e-12"

# Generate commands for all experiments
for SEED in $SEEDS; do    
    # Run DC3
    echo CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
        +experiment=dc3_$DATASET \
        dataset=$DATASET \
        model=dc3 \
        seed=$SEED \
        wandb_project=$WANDB_PROJECT

    # Run SnareNet with different rtol values
    for RTOL in $RTOL_VALUES; do
        echo CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
            +experiment=snarenet_$DATASET \
            dataset=$DATASET \
            model=snarenet \
            seed=$SEED \
            wandb_project=$WANDB_PROJECT \
            model.rtol=$RTOL
    done

    # Run hproj
    echo CUDA_VISIBLE_DEVICES=$GPU python3 run_hproj.py \
        dataset=$DATASET \
        seed=$SEED \
        wandb_project=$WANDB_PROJECT
done
