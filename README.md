# SnareNet: Flexible Repair Layers for Neural Networks with Hard Constraints

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)]() <!-- TODO: Add arXiv link -->

This repository contains the official implementation of **SnareNet**, a flexible repair layer architecture for neural networks that enforces hard constraints.

## 📖 Overview

SnareNet provides a flexible framework for incorporating hard constraints into neural network predictions. This repository includes implementations of SnareNet and several baseline methods for constrained neural networks.

## 🔧 Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Weights & Biases (wandb) account for experiment tracking
- Gurobi (optional, required for traditional optimization solver experiments)

### Installation

1. Clone the repository (with submodules):
```bash
git clone --recurse-submodules https://github.com/miniyachi/SnareNet.git
cd SnareNet
```

   If you've already cloned without submodules, initialize them:
```bash
git submodule update --init --recursive
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases:
```bash
wandb login
```

4. (Optional) Configure Gurobi if you want to run traditional optimization solver experiments:
   - Install Gurobi and obtain a license from [gurobi.com](https://www.gurobi.com/)
   - Set up your Gurobi license file

## 📁 Repository Structure

```
SnareNet/
├── run_experiment.py          # Main entry-point for neural network experiments
├── run_optimizer.py           # Main entry-point for traditional optimization solvers (requires Gurobi)
├── run_test_only.py           # Load trained model from wandb and run inference on test set
├── constants.py               # Global constants and model registry
├── requirements.txt           # Python dependencies
├── for_hproj/                 # H-Proj baseline experiments (see SCRIPTS_GUIDE.md)
│   ├── SCRIPTS_GUIDE.md         # Detailed guide for H-Proj experiments
│   ├── run_hproj.sh             # Run single H-Proj experiment
│   ├── run_hproj_sweep.sh       # Run H-Proj sweep experiments
│   ├── log_hproj_wandb.sh       # Upload H-Proj results to wandb
│   ├── run_hproj_experiment.py  # Python script for H-Proj training
│   ├── log_hproj_to_wandb.py    # Python script for wandb logging
│   ├── convert_cvx_qcqp_for_hproj.py  # Convert QCQP datasets for H-Proj
│   └── convert_noncvx_for_hproj.py    # Convert non-convex datasets for H-Proj
├── hproj_repo/                # H-Proj implementation (git submodule)
├── bash_scripts/              # Experiment execution scripts
│   ├── run_cvx_qcqp_single.sh   # Run convex QCQP experiments sequentially
│   ├── run_cvx_qcqp_multi.sh    # Run convex QCQP experiments in parallel
│   ├── run_noncvx_single.sh     # Run non-convex experiments sequentially
│   ├── run_noncvx_multi.sh      # Run non-convex experiments in parallel
│   ├── run_lambd_single.sh      # Run lambda ablation experiments sequentially
│   ├── run_lambd_multi.sh       # Run lambda ablation experiments in parallel
│   ├── run_nineq_single.sh      # Run inequality constraint experiments sequentially
│   └── run_nineq_multi.sh       # Run inequality constraint experiments in parallel
├── configs/                   # Hydra configuration files
│   ├── config.yaml              # Main configuration
│   ├── config_opt.yaml          # Optimizer configuration
│   ├── dataset/                 # Dataset-specific configurations
│   ├── experiment/              # Experiment-specific configurations
│   └── model/                   # Model-specific configurations
├── models/                    # Model implementations
│   ├── snarenet.py              # SnareNet implementation
│   ├── dc3.py                   # DC3 baseline
│   └── hardnetaff.py            # HardNetAff baseline
├── datasets/                  # Dataset files
│   ├── cvx_qcqp/                # Convex QCQP datasets
│   └── noncvx/                  # Non-convex datasets
├── notebooks/                 # Jupyter notebooks for analysis and visualization
│   └── opt_learning.ipynb       # Generate plots and tables from wandb results
└── utils/                     # Utility functions
    ├── utils.py                 # Training and data loading utilities
    └── plot_table_utils.py      # Plotting and table generation utilities
```

## 🚀 Reproducing Experiments

### Step 1: Run Experiments

Use the scripts in the `bash_scripts/` directory to run experiments. Each script contains usage instructions at the top of the file.

**Script naming convention:**
- Scripts ending with `_single.sh`: Run experiments sequentially
- Scripts ending with `_multi.sh`: Run experiments in parallel (multiple processes)

**Examples:**

Run convex QCQP experiments sequentially:
```bash
bash bash_scripts/run_cvx_qcqp_single.sh
```

Run non-convex experiments in parallel:
```bash
bash bash_scripts/run_noncvx_multi.sh
```

Run lambda ablation studies:
```bash
bash bash_scripts/run_lambd_multi.sh
```

**Running H-Proj baseline experiments:**

The H-Proj baseline has its own set of scripts in the `for_hproj/` directory. See [for_hproj/SCRIPTS_GUIDE.md](for_hproj/SCRIPTS_GUIDE.md) for detailed instructions.

Quick start:
```bash
# Step 1: Run H-Proj sweep experiments
cd for_hproj
bash run_hproj_sweep.sh --prob_type cvx_qcqp  # or noncvx

# Step 2: Upload results to wandb (configure script first)
bash log_hproj_wandb.sh
```

**Note:** Results will be automatically logged to your Weights & Biases account.

### Step 2: Generate Plots and Tables

After experiments complete, use the Jupyter notebooks to download results from wandb and generate visualizations:

1. Configure your wandb credentials in the notebook
2. Open and run the notebook:
```bash
jupyter notebook notebooks/opt_learning.ipynb
```

The notebook will:
- Download experiment results from wandb
- Generate comparison plots
- Create performance tables
- Save outputs to the `results/` directory

## 💻 Usage

### Training a Model

You can train individual models using `run_experiment.py` with Hydra configuration overrides:

```bash
# Train SnareNet on non-convex dataset with seed 123
python3 run_experiment.py model=snarenet dataset=noncvx seed=123

# Train with custom epochs and soft constraint warmup
python3 run_experiment.py model=snarenet epochs=200 soft_epochs=50

# Train DC3 baseline on convex QCQP dataset
python3 run_experiment.py model=dc3 dataset=cvx_qcqp
```

### Running Traditional Optimization Solvers

Run traditional optimization solvers for comparison (requires Gurobi):

```bash
python3 run_optimizer.py dataset=cvx_qcqp
```

### Testing a Trained Model

Load a trained model from wandb and evaluate on the test set:

```bash
python3 run_test_only.py wandb_run_id=<your_run_id>
```

## 📊 Datasets

The repository includes two main dataset categories:

1. **Convex QCQP (`cvx_qcqp/`)**: Convex Quadratically Constrained Quadratic Programs
2. **Non-convex (`noncvx/`)**: Non-convex optimization problems

Dataset configurations can be found in `configs/dataset/`.

## 🧪 Models

The repository implements several constrained neural network architectures:

- **SnareNet**: Our proposed flexible repair layer architecture
- **DC3**: Deep Constraint Completion and Correction
- **HardNetAff**: Hard constraint enforcement via affine transformations
- **H-Proj**: Hierarchical projection layers (implementation in `hproj_repo/` submodule, experiments in `for_hproj/`)

Model configurations are in `configs/model/`. For H-Proj, refer to [for_hproj/SCRIPTS_GUIDE.md](for_hproj/SCRIPTS_GUIDE.md).

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{snarenet2026,
  title={SnareNet: Flexible Repair Layers for Neural Networks with Hard Constraints},
  author={}, <!-- TODO: Add authors -->
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 🙏 Acknowledgments

This work uses:
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [Hydra](https://hydra.cc/) for configuration management
- [PyTorch](https://pytorch.org/) for deep learning
- [CVXPY](https://www.cvxpy.org/) for convex optimization
- [Gurobi](https://www.gurobi.com/) for traditional optimization solvers
