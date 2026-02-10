"""
Convert HardConstrainedNN datasets to Homeomorphic-Projection format.

This script loads datasets from our format and converts them to the format
expected by the hproj_repo codebase.
"""

import os
import sys
import pickle
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils.utils import load_data, get_problem_param_str


def convert_qcqp_dataset(data, dataset_cfg, seed=2023):
    """
    Convert QCQP dataset from our format to H-Proj format.
    
    Note: The dataset split should match the QCQP class in cvxqcqp_problem.py:
    - valid_frac = 0.0833 (8.33%)
    - test_frac = 0.0833 (8.33%)
    - train_frac = 0.8334 (83.34%)
    For 10000 examples: train=8334, valid=833, test=833
    H-Proj uses test_size for the split (last test_size samples)
    
    Args:
        data: Our QCQP problem instance
        dataset_cfg: Dataset configuration
        seed: Random seed used for dataset generation
    
    Returns:
        dict: Dataset in H-Proj format
    """
    print(f"Converting QCQP dataset with {data.ydim} vars, {data.nineq} ineq, {data.neq} eq, {data.num} examples")
    
    # Convert tensors to numpy
    Q = data.Q.detach().cpu().numpy()
    p = data.p.detach().cpu().numpy()
    A = data.A.detach().cpu().numpy()
    X = data.X.detach().cpu().numpy()
    G = data.G.detach().cpu().numpy()
    H = data.H.detach().cpu().numpy()
    h = data.h.detach().cpu().numpy()
    Y = data.Y.detach().cpu().numpy()
    
    # Get variable bounds (they use L, U; we might have YL, YU or need to infer)
    # Check if data has explicit bounds
    if hasattr(data, '_L') and hasattr(data, '_U'):
        L = data._L.detach().cpu().numpy() if torch.is_tensor(data._L) else data._L
        U = data._U.detach().cpu().numpy() if torch.is_tensor(data._U) else data._U
    else:
        # Use fixed variable bounds (No bound in our generation so use large fixed bounds)
        num_vars = Y.shape[1]
        L = np.full(num_vars, -100.0)
        U = np.full(num_vars, 100.0)
        print(f"Using fixed variable bounds: L = {L[0]:.2f}, U = {U[0]:.2f}")
    
    # Get input bounds if available
    if hasattr(data, 'input_L') and hasattr(data, 'input_U'):
        XL = data.input_L.detach().cpu().numpy() if torch.is_tensor(data.input_L) else data.input_L
        XU = data.input_U.detach().cpu().numpy() if torch.is_tensor(data.input_U) else data.input_U
    else:
        # Infer input bounds from X
        XL = np.min(X, axis=0) - 0.1
        XU = np.max(X, axis=0) + 0.1
    
    # Find best partial variables (for DC3 compatibility)
    # This finds the best subset of variables such that the remaining A columns are non-singular
    best_partial = data.partial_vars
    det = np.linalg.det(A[:, data.other_vars])
    print(f"Using partial vars with det(A_other) = {det:.6f}")
    
    # Get optimal values if available
    opt_vals = None
    if hasattr(data, 'opt_vals') and data.opt_vals is not None:
        opt_vals = data.opt_vals.detach().cpu().numpy() if torch.is_tensor(data.opt_vals) else data.opt_vals
        print(f"Including optimal values: {opt_vals.shape}")
    
    # Create dataset dictionary in H-Proj format
    dataset = {
        'Q': Q,
        'p': p,
        'A': A,
        'X': X,
        'G': G,
        'H': H,
        'h': h,
        'L': L,
        'U': U,
        'Y': Y,
        'YL': L,  # Add both naming conventions
        'YU': U,
        'XL': XL,
        'XU': XU,
        'best_partial': best_partial,
        'opt_vals': opt_vals
    }
    
    return dataset


@hydra.main(version_base=None, config_path="../configs/dataset", config_name="cvx_qcqp")
def main(cfg: DictConfig):
    """
    Main conversion function using Hydra configuration.
    
    Usage:
        python convert_cvx_qcqp_for_hproj.py
        python convert_cvx_qcqp_for_hproj.py dataset.num_var=100 dataset.num_ineq=50
    """
    print("=" * 80)
    print("Converting dataset for Homeomorphic-Projection (H-Proj)")
    print("=" * 80)
    print("\nDataset Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Wrap cfg in a dataset config structure
    dataset_cfg = DictConfig({
        'prob_type': 'cvx_qcqp',
        'num_var': cfg.num_var,
        'num_ineq': cfg.num_ineq,
        'num_eq': cfg.num_eq,
        'n_examples': cfg.n_examples,
        'seed': cfg.seed,
        'opt_val_saved': cfg.opt_val_saved
    })
    
    # Change to parent directory for data loading (utils.load_data uses relative paths)
    original_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(parent_dir)
    
    # Load the dataset
    print("\nLoading dataset...")
    try:
        data = load_data(dataset_cfg, torch.device('cpu'))
        print(f"Loaded dataset: {data.num} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nMake sure the dataset exists. You may need to generate it first:")
        print(f"  cd datasets/cvx_qcqp")
        print(f"  python generate_dataset.py")
        os.chdir(original_dir)
        return
    
    # Convert to H-Proj format
    print("\nConverting to H-Proj format...")
    hproj_dataset = convert_qcqp_dataset(data, dataset_cfg, seed=cfg.seed)
    
    # Create output directory (we're already in parent directory from loading)
    output_dir = os.path.join('hproj_repo', 'datasets', 'cvx_qcqp')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename matching H-Proj convention
    # Their format: "random_{seed}_{prob_type}_dataset_var{n}_ineq{m}_eq{k}_ex{N}"
    filename = f"random_{cfg.seed}_cvx_qcqp_dataset_var{cfg.num_var}_ineq{cfg.num_ineq}_eq{cfg.num_eq}_ex{cfg.n_examples}"
    filepath = os.path.join(output_dir, filename)
    
    # Save dataset
    print(f"\nSaving to: {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(hproj_dataset, f)
    
    # Restore original directory
    os.chdir(original_dir)
    
    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Modify hproj_repo/default_args.py to set your problem parameters:")
    print(f"   - probSize: [{cfg.num_var}, {cfg.num_ineq}, {cfg.num_eq}, {cfg.n_examples}]")
    print(f"   - seed: {cfg.seed}")
    print("2. Run H-Proj training:")
    print("   cd hproj_repo")
    print("   python training_all.py")
    print("\nOr use the helper script:")
    print(f"   python run_hproj_experiment.py")
    

if __name__ == '__main__':
    main()
