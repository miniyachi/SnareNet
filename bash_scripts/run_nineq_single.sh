#!/usr/bin/env bash
# Run comparison experiments sequentially for dataset with different inequalities
# Usage: 
# bash run_nineq_single.sh

# Change to the root directory of the repository
cd "$(dirname "$0")/.." || exit 1

# Configuration
SEEDS="456 112 123 789 101"
DATASET="cvx_qcqp"
NUM_INEQ_VALUES="100 50 10"
DATE=$(date +%b%d)

# Define the GPU to use
GPU="0"

# Run all experiments sequentially with different num_ineq values
for NUM_INEQ in $NUM_INEQ_VALUES; do
    WANDB_PROJECT="cvx_qcqp_ineq${NUM_INEQ}_${DATE}"
    
    # Run optimizer baseline
    CUDA_VISIBLE_DEVICES=$GPU python3 run_optimizer.py \
        +experiment=opt_$DATASET \
        dataset=$DATASET \
        dataset.num_ineq=$NUM_INEQ \
        wandb_project=$WANDB_PROJECT
    
    for SEED in $SEEDS; do    
        # Run DC3
        CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
            +experiment=dc3_ineq$NUM_INEQ \
            model=dc3 \
            dataset=$DATASET \
            dataset.num_ineq=$NUM_INEQ \
            seed=$SEED \
            wandb_project=$WANDB_PROJECT
    done

    for SEED in $SEEDS; do
        # Run SnareNet
        CUDA_VISIBLE_DEVICES=$GPU python3 run_experiment.py \
            +experiment=snarenet_ineq \
            dataset=$DATASET \
            dataset.num_ineq=$NUM_INEQ \
            model=snarenet \
            seed=$SEED \
            wandb_project=$WANDB_PROJECT
    done
done
