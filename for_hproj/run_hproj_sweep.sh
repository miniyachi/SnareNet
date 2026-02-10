#!/bin/bash
# Run H-Proj comparison experiments across multiple configurations

CUDA_VISIBLE_DEVICES="2"

# Fixed parameters
NUM_VAR=100
NUM_EQ=50
N_EXAMPLES=10000
SEED=2025

# Sweep parameters
INEQ_VALUES=(10 50 100)
TRAIN_SEEDS=(123 456 789 101 112 131 415 161 718 192 021)

# Parse command line arguments to override defaults
PROJ_TYPE="H_Bis"
PROB_TYPE="cvx_qcqp"  # Default problem type
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --prob_type)
            PROB_TYPE="$2"
            shift 2
            ;;
        --proj_type)
            PROJ_TYPE="$2"
            shift 2
            ;;
        --num_var)
            NUM_VAR="$2"
            shift 2
            ;;
        --num_eq)
            NUM_EQ="$2"
            shift 2
            ;;
        --n_examples)
            N_EXAMPLES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --skip_mapping)
            EXTRA_ARGS="$EXTRA_ARGS --skip_mapping"
            shift
            ;;
        --skip_nn)
            EXTRA_ARGS="$EXTRA_ARGS --skip_nn"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--prob_type TYPE] [--proj_type TYPE] [--num_var N] [--num_eq N] [--n_examples N] [--seed N] [--skip_mapping] [--skip_nn]"
            exit 1
            ;;
    esac
done

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#INEQ_VALUES[@]} * ${#TRAIN_SEEDS[@]}))
CURRENT_EXPERIMENT=0

echo "========================================"
echo "H-Proj Sweep Experiment"
echo "========================================"
echo "Fixed parameters:"
echo "  PROB_TYPE: $PROB_TYPE"
echo "  NUM_VAR: $NUM_VAR"
echo "  NUM_EQ: $NUM_EQ"
echo "  N_EXAMPLES: $N_EXAMPLES"
echo "  SEED: $SEED"
echo "  PROJ_TYPE: $PROJ_TYPE"
echo ""
echo "Sweep parameters:"
echo "  INEQ_VALUES: ${INEQ_VALUES[@]}"
echo "  TRAIN_SEEDS: ${TRAIN_SEEDS[@]}"
echo ""
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "========================================"
echo ""

# Create log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Record start time
START_TIME=$(date +%s)

# Loop through all combinations
for NUM_INEQ in "${INEQ_VALUES[@]}"; do
    for TRAIN_SEED in "${TRAIN_SEEDS[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        
        echo ""
        echo "========================================"
        echo "Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
        echo "========================================"
        echo "Configuration:"
        echo "  PROB_TYPE: $PROB_TYPE"
        echo "  NUM_VAR: $NUM_VAR"
        echo "  NUM_INEQ: $NUM_INEQ"
        echo "  NUM_EQ: $NUM_EQ"
        echo "  N_EXAMPLES: $N_EXAMPLES"
        echo "  SEED: $SEED"
        echo "  TRAIN_SEED: $TRAIN_SEED"
        echo "  PROJ_TYPE: $PROJ_TYPE"
        echo "========================================"
        
        # Create log file name
        LOG_FILE="$LOG_DIR/hproj_${PROB_TYPE}_${PROJ_TYPE}_ineq${NUM_INEQ}_trainseed${TRAIN_SEED}.log"
        
        # Run experiment
        bash run_hproj_comparison.sh \
            --prob_type $PROB_TYPE \
            --num_var $NUM_VAR \
            --num_ineq $NUM_INEQ \
            --num_eq $NUM_EQ \
            --n_examples $N_EXAMPLES \
            --seed $SEED \
            --train_seed $TRAIN_SEED \
            --proj_type $PROJ_TYPE \
            $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"
        
        # Check if experiment succeeded
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo ""
            echo "WARNING: Experiment failed for NUM_INEQ=$NUM_INEQ, TRAIN_SEED=$TRAIN_SEED"
            echo "Check log file: $LOG_FILE"
            echo ""
        else
            echo ""
            echo "SUCCESS: Experiment completed for NUM_INEQ=$NUM_INEQ, TRAIN_SEED=$TRAIN_SEED"
            echo ""
        fi
        
        # Calculate and display progress
        ELAPSED_TIME=$(($(date +%s) - START_TIME))
        AVG_TIME=$((ELAPSED_TIME / CURRENT_EXPERIMENT))
        REMAINING_EXPERIMENTS=$((TOTAL_EXPERIMENTS - CURRENT_EXPERIMENT))
        EST_REMAINING_TIME=$((AVG_TIME * REMAINING_EXPERIMENTS))
        
        echo "Progress: $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS experiments completed"
        echo "Elapsed time: $(printf '%02d:%02d:%02d' $((ELAPSED_TIME/3600)) $((ELAPSED_TIME%3600/60)) $((ELAPSED_TIME%60)))"
        echo "Estimated remaining time: $(printf '%02d:%02d:%02d' $((EST_REMAINING_TIME/3600)) $((EST_REMAINING_TIME%3600/60)) $((EST_REMAINING_TIME%60)))"
        echo ""
    done
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "========================================"
echo "All Experiments Complete!"
echo "========================================"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Total time: $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo "Average time per experiment: $(printf '%02d:%02d:%02d' $((TOTAL_TIME/TOTAL_EXPERIMENTS/3600)) $((TOTAL_TIME/TOTAL_EXPERIMENTS%3600/60)) $((TOTAL_TIME/TOTAL_EXPERIMENTS%60)))"
echo ""
echo "Results saved in: ../results/"
echo "Logs saved in: $LOG_DIR/"
echo "========================================"
