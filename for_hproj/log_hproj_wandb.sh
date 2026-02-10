#!/bin/bash
# Helper script to log H-Proj results to wandb
# Usage: bash log_hproj_wandb.sh
# Configure the variables below before running

# ============ CONFIGURATION ============
# Set the problem type: "cvx_qcqp" or "noncvx" and number of inequalities
PROB_TYPE="noncvx"
NUM_INEQ=50

# Set the results directory containing CSV files
RESULTS_DIR="../results/${PROB_TYPE}/var100_ineq${NUM_INEQ}_eq50_seed2025/hproj"

# Set the wandb project name
WANDB_PROJECT="HProj_${PROB_TYPE}_Bd100_var100_ineq${NUM_INEQ}_eq50"

# Set to "--dry_run" for preview, or "" to actually log
DRY_RUN=""
# ======================================

# Make sure we're in the for_hproj directory
cd "$(dirname "$0")"

# Run the script
python3 log_hproj_to_wandb.py \
    --dir "$RESULTS_DIR" \
    --project "$WANDB_PROJECT" \
    $DRY_RUN
