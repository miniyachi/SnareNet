import numpy as np
import cvxpy as cp
import os
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import get_problem_param_str

from noncvx_problem import NonCvxProblem

@hydra.main(version_base=None, config_path="../../configs/dataset", config_name="noncvx")
def generate(dataset_cfg: DictConfig) -> None:
    """Generate Non-convex dataset using Hydra configuration."""
    
    print("Configuration:")
    print(OmegaConf.to_yaml(dataset_cfg))
    
    # Extract parameters from config
    num_var = dataset_cfg.num_var
    num_ineq = dataset_cfg.num_ineq
    num_eq = dataset_cfg.num_eq
    seed = dataset_cfg.seed
    num_examples = dataset_cfg.n_examples
    opt_val_saved = dataset_cfg.opt_val_saved

    # Set random seed
    np.random.seed(seed)

    print('Generating random coefficient matrices...')
    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    C = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    b = np.sum(np.abs(A@np.linalg.pinv(C)), axis=1)
    X = np.random.uniform(-1, 1, size=(num_examples, num_eq))

    # Create Non-convex problem instance
    problem = NonCvxProblem(Q, p, A, b, C, X)
    if opt_val_saved:
        print(f'Solving {num_examples} examples of non-convex problem with {num_var} variables, {num_eq} equality constraints, and {num_ineq} inequality constraints...')
        total_time, avg_time = problem.opt_solve(solver_type='scipy')
        print(f'Finished solving non-convex problem instances.')
        print(f'Total time: {total_time:.2f}s, Average time: {avg_time:.4f}s per problem')
        Y = problem.Y.detach().cpu().numpy()
        opt_vals = problem.opt_vals.cpu().numpy()
    else:
        print('Skipping optimal value computation (opt_val_saved=False)')
        Y = None
        opt_vals = None

    print('Saving Non-convex dataset...')
    save_dir = os.path.dirname(os.path.abspath(__file__))
    prob_params = get_problem_param_str(dataset_cfg)
    filename = f"noncvx_{prob_params}"
    
    # Save as dictionary of raw data instead of problem class instance
    dataset_dict = {
        'prob_type': 'noncvx',
        'class_input': {
            'Q': Q,
            'p': p,
            'A': A,
            'b': b,
            'C': C,
            'X': X,
        },
        'opt_sols': Y,
        'opt_vals': opt_vals,
    }
    
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump(dataset_dict, f)
    print(f'Saved dataset to {save_dir}/{filename}')


if __name__ == "__main__":
    generate()

# python3 datasets/noncvx/generate_dataset.py