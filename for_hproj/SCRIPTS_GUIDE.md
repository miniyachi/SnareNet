# H-Proj Scripts Guide

Three main scripts for running H-Proj experiments and logging results to wandb.

---

## Reproducing H-Proj Comparison Experiments

**Step 1**: Run sweep experiments for your problem type:
```bash
# For CVX-QCQP problems
bash run_hproj_sweep.sh --prob_type cvx_qcqp

# For noncvx problems
bash run_hproj_sweep.sh --prob_type noncvx
```

**Step 2**: Log results to wandb:
```bash
# Edit log_hproj_wandb.sh to configure:
#   - PROB_TYPE (must match Step 1)
#   - NUM_INEQ (to match sweep configuration)
#   - WANDB_PROJECT name
bash log_hproj_wandb.sh
```

---

## Script Details

### `run_hproj.sh`
Runs a single H-Proj experiment: generates/converts dataset → trains model → extracts results to CSV.

**Usage**:
```bash
bash run_hproj.sh --prob_type [cvx_qcqp|noncvx] --train_seed 123 [OPTIONS]
```

**Key options**: `--num_var`, `--num_ineq`, `--num_eq`, `--proj_type [WS|Proj|D_Proj|H_Bis]`

**Output**: `../results/{prob_type}/var{N}_ineq{N}_eq{N}_seed{N}/hproj/*.csv`

---

### `run_hproj_sweep.sh`
Sweeps over multiple configurations (inequality values × training seeds). Calls `run_hproj.sh` repeatedly and logs progress.

**Configuration** (edit in script):
```bash
INEQ_VALUES=(10 50 100)
TRAIN_SEEDS=(123 456 789 101 112 131 415 161 718 192 021)
```

**Usage**:
```bash
bash run_hproj_sweep.sh --prob_type [cvx_qcqp|noncvx] [OPTIONS]
```

**Output**: Results in `../results/`, logs in `logs/hproj_*.log`

---

### `log_hproj_wandb.sh`
Uploads CSV results to wandb project for visualization and comparison.

**Configuration** (edit in script before running):
```bash
PROB_TYPE="cvx_qcqp"  # or "noncvx"
NUM_INEQ=50
WANDB_PROJECT="HProj_${PROB_TYPE}_Bd100_var100_ineq${NUM_INEQ}_eq50"
DRY_RUN=""  # Set to "--dry_run" for preview
```

**Usage**:
```bash
bash log_hproj_wandb.sh
```
