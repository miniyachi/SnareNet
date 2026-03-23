import numpy as np
import pickle
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.utils import get_problem_param_str
from cbf_problem import Obstacle, Unicycle_Acc

os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

@hydra.main(version_base=None, config_path="../../configs/dataset", config_name="cbf")
def generate_cbf_dataset(dataset_cfg: DictConfig) -> None:
    """Generate CBF dataset using Hydra configuration."""
    
    print("Dataset Config:")
    print(OmegaConf.to_yaml(dataset_cfg))
    
    # Extract parameters from config
    num_examples = dataset_cfg.n_examples
    seed = dataset_cfg.seed
    
    # CBF-specific parameters
    loss_max = dataset_cfg.get('loss_max', 20000)
    alpha = dataset_cfg.get('alpha', 20)
    kappa = dataset_cfg.get('kappa', 10)
    T = dataset_cfg.get('T', 1.0)
    dt = dataset_cfg.get('dt', 0.02)
    valid_frac = dataset_cfg.get('valid_frac', 0.0833)
    test_frac = dataset_cfg.get('test_frac', 0.0833)
    x_min = dataset_cfg.get('x_min', -5.5)
    y_min = dataset_cfg.get('y_min', -5)
    x_max = dataset_cfg.get('x_max', 7.5)
    y_max = dataset_cfg.get('y_max', 8)

    np.random.seed(seed)

    Q = np.diag([100, 100, 0.0, 0.1, 0.1])
    R = np.diag([0.1, 0.1])
    init_box = [np.array([x_min, y_min, -np.pi/4, 0.0, 0.0]),
                np.array([x_max, y_max, -np.pi/8, 0.0, 0.0])]
    sys = Unicycle_Acc(init_box, kappa=kappa)

    obs_param_list = [(-5, 5, 2, 1.41), (-1, 3, 0.25, 1.41), (-9, 2, 1, 1)]
    obs_list = []
    num_obs = len(obs_param_list)
    for i in range(num_obs):
        obs_list.append(Obstacle(*obs_param_list[i]))

    X = sys.generate_states(num_examples)
    failed_indices = np.array(range(num_examples))
    while len(failed_indices) > 0:
        failed_samples = X[failed_indices, :]
        failure = np.repeat(False, len(failed_indices))
        for obs in obs_list:
            cbf = sys.get_cbf_h(torch.tensor(failed_samples), obs)
            failure = failure | (cbf.numpy() < 0)
        failed_indices = failed_indices[failure]
        X[failed_indices, :] = sys.generate_states(len(failed_indices))
        print(f'regenerated {len(failed_indices)} samples')

    print(f"problem length:{num_examples} examples with {int(T/dt)} steps")

    print('Saving CBF dataset...')
    assert(len(obs_list) == dataset_cfg.get('num_ineq'))
    save_dir = os.path.dirname(os.path.abspath(__file__))
    prob_params = get_problem_param_str(dataset_cfg)
    filename = f"cbf_{prob_params}"
    
    # Save as dictionary of raw data instead of problem class instance
    dataset_dict = {
        'prob_type': 'cbf',
        'class_input': {
            'Q': Q,
            'R': R,
            'X': X,
            'sys': sys,
            'obs_list': obs_list,
            'loss_max': loss_max,
            'alpha': alpha,
            'T': T,
            'dt': dt,
            'valid_frac': valid_frac,
            'test_frac': test_frac,
        },
        'opt_sols': None,  # CBF doesn't pre-compute solutions
        'opt_vals': None,
    }
    
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump(dataset_dict, f)
    print(f'Saved CBF dataset to {save_dir}/{filename}')

if __name__ == "__main__":
    generate_cbf_dataset()