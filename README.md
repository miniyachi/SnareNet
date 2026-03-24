# SnareNet: Flexible Repair Layers for Neural Networks with Hard Constraints

[![arXiv](https://img.shields.io/badge/arXiv-2602.09317-b31b1b.svg)](https://arxiv.org/abs/2602.09317) <!-- TODO: Add arXiv link -->

This repository contains the official implementation of **SnareNet**, a flexible repair layer architecture for neural networks that enforces hard constraints.

## 📖 Overview

SnareNet provides a flexible framework for incorporating hard constraints into neural network predictions. This repository includes implementations of SnareNet and several baseline methods for constrained neural networks.

## 🔧 Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Weights & Biases (wandb) account for experiment tracking
- Gurobi (optional, required for traditional optimization solver experiments)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/miniyachi/SnareNet.git
cd SnareNet
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
├── run_hproj.py               # Main entry-point for H-Proj baseline
├── run_optimizer.py           # Main entry-point for traditional optimization solvers (requires Gurobi)
├── run_test_only.py           # Load trained model from wandb and run inference on test set
├── constants.py               # Global constants and model registry
├── requirements.txt           # Python dependencies
├── bash_scripts/              # Experiment execution scripts
│   ├── run_cvx_qcqp_single.sh   # Run convex QCQP experiments sequentially
│   ├── run_cvx_qcqp_multi.sh    # Run convex QCQP experiments in parallel
│   ├── run_noncvx_single.sh     # Run non-convex experiments sequentially
│   ├── run_noncvx_multi.sh      # Run non-convex experiments in parallel
│   ├── run_lambd_single.sh      # Run lambda ablation experiments sequentially
│   ├── run_lambd_multi.sh       # Run lambda ablation experiments in parallel
│   ├── run_nineq_single.sh      # Run inequality constraint scaling experiments sequentially
│   ├── run_nineq_multi.sh       # Run inequality constraint scaling experiments in parallel
│   ├── run_cbf_single.sh        # Run CBF safe control experiments sequentially
│   └── run_cbf_multi.sh         # Run CBF safe control experiments in parallel
├── configs/                   # Hydra configuration files
│   ├── config.yaml              # Main configuration
│   ├── config_hproj.yaml        # H-Proj baseline configuration
│   ├── config_opt.yaml          # Optimizer configuration
│   ├── dataset/                 # Dataset-specific configurations
│   ├── experiment/              # Experiment-specific configurations
│   └── model/                   # Model-specific configurations
│       ├── hproj.yaml             # H-Proj model configuration
│       └── ...
├── models/                    # Model implementations
│   ├── snarenet.py              # SnareNet implementation
│   ├── dc3.py                   # DC3 baseline
│   ├── hardnetaff.py            # HardNetAff baseline
│   └── hproj.py                 # H-Proj baseline (INN + HomeoProjNet)
├── datasets/                  # Dataset files
│   ├── cvx_qcqp/                # Convex QCQP datasets
│   ├── noncvx/                  # Non-convex datasets
│   └── cbf/                     # CBF safe control datasets
│       ├── cbf_problem.py         # CBF problem definition (unicycle system, obstacles)
│       └── generate_dataset.py    # Dataset generation script
├── sbatch_scripts/            # SLURM batch job submission scripts
│   ├── run_cvx_qcqp_sbatch.py  # SLURM jobs for convex QCQP experiments
│   ├── run_noncvx_sbatch.py     # SLURM jobs for non-convex experiments
│   ├── run_lambd_sbatch.py      # SLURM jobs for lambda ablation experiments
│   ├── run_nineq_sbatch.py      # SLURM jobs for inequality scaling experiments
│   └── run_cbf_sbatch.py        # SLURM jobs for CBF experiments
├── notebooks/                 # Jupyter notebooks for analysis and visualization
│   ├── opt_learning.ipynb       # Generate plots and tables from wandb results
│   └── cont_cbf_visual.ipynb    # Visualize CBF safe control trajectories
└── utils/                     # Utility functions
    ├── utils.py                 # Training and data loading utilities
    └── plot_table_utils.py      # Plotting and table generation utilities
```

## 🚀 Reproduce Experiments

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

Run CBF safe control experiments:
```bash
bash bash_scripts/run_cbf_single.sh
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

### Running H-Proj Baseline

The H-Proj baseline uses a two-stage pipeline: first training a homeomorphic INN mapping (Stage 1), then training a BaseModel predictor (Stage 2), followed by bisection-based repair at inference.

```bash
# Run on convex QCQP dataset
python3 run_hproj.py dataset=cvx_qcqp seed=123

# Run on non-convex dataset
python3 run_hproj.py dataset=noncvx seed=123

# Skip Stage 1 and reuse existing homeoproj_module.pth
python3 run_hproj.py dataset=cvx_qcqp model.skip_mapping=true seed=123
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

The repository includes three dataset categories:

1. **Convex QCQP (`cvx_qcqp/`)**: Convex Quadratically Constrained Quadratic Programs
2. **Non-convex (`noncvx/`)**: Non-convex optimization problems
3. **CBF Safe Control (`cbf/`)**: Safe control for a unicycle system with elliptical obstacles, using Control Barrier Functions (CBF)

Dataset configurations can be found in `configs/dataset/`.

## 🧪 Models

The repository implements several constrained neural network architectures:

- **SnareNet**: Our proposed flexible repair layer architecture
- **DC3**: Deep Constraint Completion and Correction
- **HardNetAff**: Hard constraint enforcement via affine transformations
- **H-Proj**: Homeomorphic projection baseline (INN + bisection repair); implemented in [models/hproj.py](models/hproj.py), run via [run_hproj.py](run_hproj.py)

Model configurations are in `configs/model/`.

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{chu2026snarenet,
      title={SnareNet: Flexible Repair Layers for Neural Networks with Hard Constraints}, 
      author={Ya-Chi Chu and Alkiviades Boukas and Madeleine Udell},
      year={2026},
      eprint={2602.09317},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.09317}, 
}
```

## 🙏 Acknowledgments

This work uses:
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [Hydra](https://hydra.cc/) for configuration management
- [PyTorch](https://pytorch.org/) for deep learning
- [CVXPY](https://www.cvxpy.org/) for convex optimization
- [Gurobi](https://www.gurobi.com/) for traditional optimization solvers
