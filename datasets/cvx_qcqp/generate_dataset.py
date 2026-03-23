import numpy as np
import os
import pickle
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.utils import get_problem_param_str

from cvxqcqp_problem import QCQP

def generate_valid_ineq_bd(A, H, G, X, margin=0.1):
    """
    Compute h_i upper bound such that y = A^+ x is feasible for all x in X.
    
    Inequality: x^T (A^+)^T H_i A^+ x + G_i^T A^+ x <= h_i
    
    Using bounds:
      x^T M x <= ||M||_2 * ||x||_2^2
      v^T x   <= ||v||_2 * ||x||_2
    
    Args:
        A: Equality constraint matrix (num_eq, num_var)
        H: Quadratic inequality constraint matrices (num_ineq, num_var, num_var)
        G: Linear inequality constraint matrix (num_ineq, num_var)
        X: Input samples (num_examples, num_eq)
        margin: Safety margin for the bound
    
    Returns:
        h: Upper bounds for inequality constraints (num_ineq,)
    """
    Apinv = np.linalg.pinv(A)  # shape = (num_var, num_eq)
    quad_mat = np.einsum('va,ivw,wb->iab', Apinv, H, Apinv)     # Quadratic term matrices: (A^+)^T H_i A^+ for each i.  # shape = (num_ineq, num_eq, num_eq)
    lin_vec = G @ Apinv                                         # Linear term vectors: G_i^T A^+ for each i.            # shape = (num_ineq, num_eq)
    max_x_norm = np.max(np.linalg.norm(X, ord=2, axis=1))       # Max ||x||_2 over all samples.                         # scalar
    quad_norms = np.linalg.norm(quad_mat, ord=2, axis=(1, 2))   # Spectral norms of quad_mat[i] for each constraint.    # shape = (num_ineq,)  
    lin_norms = np.linalg.norm(lin_vec, ord=2, axis=1)          # L2 norms of lin_vec[i] for each constraint.           # shape = (num_ineq,)
    h = quad_norms * (max_x_norm ** 2) + lin_norms * max_x_norm + margin  # Upper bound: ||M_i||_2 * ||x||_max^2 + ||v_i||_2 * ||x||_max + margin   # shape = (num_ineq,)
    return h

@hydra.main(version_base=None, config_path="../../configs/dataset", config_name="cvx_qcqp")
def generate(dataset_cfg: DictConfig) -> None:
    """Generate CVX QCQP dataset using Hydra configuration."""
    
    print("Dataset Config:")
    print(OmegaConf.to_yaml(dataset_cfg))
    
    # Extract parameters from config
    num_var = dataset_cfg.num_var
    num_ineq = dataset_cfg.num_ineq
    num_eq = dataset_cfg.num_eq
    num_examples = dataset_cfg.n_examples
    seed = dataset_cfg.seed
    opt_val_saved = dataset_cfg.opt_val_saved

    # Set random seed
    np.random.seed(seed)

    print('Generating random coefficient matrices...')
    Q = np.diag(np.random.rand(num_var)*0.5)
    p = np.random.uniform(-1, 1, num_var)
    A = np.random.uniform(-1, 1, size=(num_eq, num_var))
    X = np.random.uniform(-0.5, 0.5, size=(num_examples, num_eq))
    G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
    H = np.random.uniform(0, 0.1,  size=(num_ineq, num_var))
    H = np.array([np.diag(H[i]) for i in range(num_ineq)])  # shape = (num_ineq, num_var, num_var)
    L = np.ones((num_var)) * -10
    U = np.ones((num_var)) * 10
    h = generate_valid_ineq_bd(A, H, G, X, margin=0.1)

    print('creating QCQP problem instance...')
    problem = QCQP(Q, p, A, X, G, H, h, L, U)
    if opt_val_saved:
        print(f'Solving {num_examples} examples of convex QCQP with {num_var} variables, {num_eq} equality constraints, and {num_ineq} inequality constraints...')
        problem.opt_solve(solver_type='gurobipy')
        print(f'Finished solving QCQP problem instances.')
        Y = problem.Y.detach().cpu().numpy()
        opt_vals = problem.opt_vals.cpu().numpy()
    else:
        print('Skipping optimal value computation (opt_val_saved=False)')
        Y = None
        opt_vals = None

    print('Saving QCQP dataset...')
    save_dir = os.path.dirname(os.path.abspath(__file__))
    prob_params = get_problem_param_str(dataset_cfg)
    filename = f"cvx_qcqp_{prob_params}"
    
    # Save as dictionary of raw data instead of problem class instance
    dataset_dict = {
        'prob_type': 'cvx_qcqp',
        'class_input': {
            'Q': Q,
            'p': p,
            'A': A,
            'X': X,
            'G': G,
            'H': H,
            'h': h,
            'L': L,
            'U': U,
        },
        'opt_sols': Y,
        'opt_vals': opt_vals,
    }
    
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump(dataset_dict, f)
    print(f'Saved QCQP dataset to {save_dir}/{filename}')

if __name__ == "__main__":
    generate()

# python3 datasets/cvx_qcqp/generate_dataset.py num_var=100 num_ineq=10 num_eq=50 n_examples=10000