#!/bin/bash
# Run H-Proj comparison experiments
# 
# IMPORTANT: Make sure to activate the test_req environment before running:
#   source /home/aboukas/hard-constrain/test_req/bin/activate

# Default parameters (matching your existing experiments)
PROB_TYPE="cvx_qcqp"  # Options: cvx_qcqp, noncvx
NUM_VAR=100
NUM_INEQ=50
NUM_EQ=50
N_EXAMPLES=10000
# TEST_SIZE will auto-calculate as 8.33% (matching QCQP class in cvxqcqp_problem.py)
SEED=2023
TRAIN_SEED=""  # Empty means no seeding (non-deterministic training)
PROJ_TYPE="H_Bis"  # Options: WS, Proj, D_Proj, H_Bis
SKIP_MAPPING=false
SKIP_NN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prob_type)
            PROB_TYPE="$2"
            shift 2
            ;;
        --num_var)
            NUM_VAR="$2"
            shift 2
            ;;
        --num_ineq)
            NUM_INEQ="$2"
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
        --train_seed)
            TRAIN_SEED="$2"
            shift 2
            ;;
        --proj_type)
            PROJ_TYPE="$2"
            shift 2
            ;;
        --skip_mapping)
            SKIP_MAPPING=true
            shift
            ;;
        --skip_nn)
            SKIP_NN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "H-Proj Comparison Experiment"
echo "========================================"
echo "Problem type: ${PROB_TYPE}"
echo "Problem size: ${NUM_VAR} vars, ${NUM_INEQ} ineq, ${NUM_EQ} eq"
echo "Examples: ${N_EXAMPLES}, Seed: ${SEED}, Train seed: ${TRAIN_SEED}"
echo "Projection type: ${PROJ_TYPE}"
echo ""

# Step 1: Check if dataset exists (try multiple naming patterns)
if [ "$PROB_TYPE" = "cvx_qcqp" ]; then
    DATASET_DIR="../datasets/cvx_qcqp"
    DATASET_PATTERNS=(
        "cvx_qcqp_var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_seed${SEED}"
        "cls_cvx_qcqp_var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_seed${SEED}"
        "cvx_qcqp_dataset_var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_ex${N_EXAMPLES}_seed${SEED}"
    )
elif [ "$PROB_TYPE" = "noncvx" ]; then
    DATASET_DIR="../datasets/noncvx"
    DATASET_PATTERNS=(
        "noncvx_var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_seed${SEED}"
    )
else
    echo "Error: Unknown problem type: ${PROB_TYPE}"
    exit 1
fi

DATASET_PATH=""
for pattern in "${DATASET_PATTERNS[@]}"; do
    if [ -f "${DATASET_DIR}/${pattern}" ]; then
        DATASET_PATH="${DATASET_DIR}/${pattern}"
        echo "Found dataset: ${pattern}"
        break
    fi
done

if [ -z "$DATASET_PATH" ]; then
    echo "Dataset not found. Tried the following names:"
    for pattern in "${DATASET_PATTERNS[@]}"; do
        echo "  - ${pattern}"
    done
    echo ""
    echo "Attempting to generate dataset..."
    cd $DATASET_DIR
    python3 generate_dataset.py \
        num_var=$NUM_VAR \
        num_ineq=$NUM_INEQ \
        num_eq=$NUM_EQ \
        n_examples=$N_EXAMPLES \
        seed=$SEED
    cd ../..
    
    # Check again after generation
    for pattern in "${DATASET_PATTERNS[@]}"; do
        if [ -f "${DATASET_DIR}/${pattern}" ]; then
            DATASET_PATH="${DATASET_DIR}/${pattern}"
            break
        fi
    done
    
    if [ -z "$DATASET_PATH" ]; then
        echo "Error: Failed to generate or find dataset"
        exit 1
    fi
fi

# Step 2: Convert dataset for H-Proj
echo ""
echo "========================================"
echo "Converting dataset for H-Proj..."
echo "========================================"

if [ "$PROB_TYPE" = "cvx_qcqp" ]; then
    python3 ./convert_cvx_qcqp_for_hproj.py \
        num_var=$NUM_VAR \
        num_ineq=$NUM_INEQ \
        num_eq=$NUM_EQ \
        n_examples=$N_EXAMPLES \
        seed=$SEED
elif [ "$PROB_TYPE" = "noncvx" ]; then
    python3 ./convert_noncvx_for_hproj.py \
        num_var=$NUM_VAR \
        num_ineq=$NUM_INEQ \
        num_eq=$NUM_EQ \
        n_examples=$N_EXAMPLES \
        seed=$SEED
fi

if [ $? -ne 0 ]; then
    echo "Error: Dataset conversion failed"
    exit 1
fi

# Step 3: Run H-Proj experiment
echo ""
echo "========================================"
echo "Running H-Proj experiment..."
echo "========================================"

# Build the command with optional flags
CMD="python3 ./run_hproj_experiment.py \
    --prob_type $PROB_TYPE \
    --num_var $NUM_VAR \
    --num_ineq $NUM_INEQ \
    --num_eq $NUM_EQ \
    --n_examples $N_EXAMPLES \
    --seed $SEED \
    --proj_type $PROJ_TYPE"

# Only add train_seed if it's specified
if [ -n "$TRAIN_SEED" ]; then
    CMD="$CMD --train_seed $TRAIN_SEED"
fi

if [ "$SKIP_MAPPING" = true ]; then
    CMD="$CMD --skip_mapping"
fi

if [ "$SKIP_NN" = true ]; then
    CMD="$CMD --skip_nn"
fi

eval $CMD

if [ $? -ne 0 ]; then
    echo "Error: H-Proj experiment failed"
    exit 1
fi

# Step 4: Extract results
echo ""
echo "========================================"
echo "Extracting results..."
echo "========================================"
# Construct filename based on whether train_seed is specified
if [ -n "$TRAIN_SEED" ]; then
    OUTPUT_FILE="../results/${PROB_TYPE}/var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_seed${SEED}/hproj/hproj_${PROB_TYPE}_${PROJ_TYPE}_var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_trainseed${TRAIN_SEED}.csv"
    EXTRACT_CMD="python3 ./extract_hproj_results.py \
        --prob_type $PROB_TYPE \
        --num_var $NUM_VAR \
        --num_ineq $NUM_INEQ \
        --num_eq $NUM_EQ \
        --n_examples $N_EXAMPLES \
        --proj_type $PROJ_TYPE \
        --method_name \"H-Proj-${PROJ_TYPE}\" \
        --output $OUTPUT_FILE \
        --train_seed $TRAIN_SEED"
else
    OUTPUT_FILE="../results/${PROB_TYPE}/var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_seed${SEED}/hproj/hproj_${PROB_TYPE}_${PROJ_TYPE}_var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_unseeded.csv"
    EXTRACT_CMD="python3 ./extract_hproj_results.py \
        --prob_type $PROB_TYPE \
        --num_var $NUM_VAR \
        --num_ineq $NUM_INEQ \
        --num_eq $NUM_EQ \
        --n_examples $N_EXAMPLES \
        --proj_type $PROJ_TYPE \
        --method_name \"H-Proj-${PROJ_TYPE}\" \
        --output $OUTPUT_FILE"
fi
mkdir -p ../results/${PROB_TYPE}/var${NUM_VAR}_ineq${NUM_INEQ}_eq${NUM_EQ}_seed${SEED}/hproj

eval $EXTRACT_CMD

echo ""
echo "========================================"
echo "Experiment Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_FILE"
