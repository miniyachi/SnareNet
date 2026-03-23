"""
Utility functions for training and evaluating hard-constrained neural networks
"""

import torch
import torch.optim as optim
torch.set_default_dtype(torch.float64)
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import time
import random
import os
import sys
import pickle
import wandb
from scipy.stats import gmean
from omegaconf import DictConfig, OmegaConf

from constants import PROBTYPE_TO_CLASS

# available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# print(f"Available GPUs: {available_gpus}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AdaptiveRelaxation:
    """Adaptive relaxation for constraint satisfaction during training.
    
    Gradually tightens constraint violations from initial measured values to zero
    over the course of training.
    """
    def __init__(self, start_epoch, decay_epochs, device, decay_fn='linear'):
        self.start_epoch = start_epoch
        self.decay_epochs = decay_epochs
        self.device = device
        self.eps_initial = None
        self.initialized = False
        
        # Set decay function
        if callable(decay_fn):
            self.decay_fn = decay_fn
        elif decay_fn == 'linear':
            self.decay_fn = self._linear_decay
        elif decay_fn == 'harmonic':
            self.decay_fn = self._harmonic_decay
        elif decay_fn == 'linear_harmonic':
            self.decay_fn = self._linear_harmonic_decay
        else:
            raise ValueError(f"Unknown decay function: {decay_fn}")

    def get_init_eps(self, data, net, train_loader):
        """Measure initial violations across training set and set decay rate"""
        if self.initialized:
            return
        
        self.initialized = True
        
        # Initialize with -inf so torch.maximum works directly
        n_constraints = data.nineq + data.neq
        self.eps_initial = torch.full((n_constraints,), float('-inf'), device=self.device)  # shape = (nineq + neq,)
        
        # Find max violations for each constraint over training set
        with torch.no_grad():
            # Temporarily disable projection to get raw NN output
            if not hasattr(net, 'set_projection'):
                raise AttributeError("Network must have 'set_projection' method to use adaptive relaxation.")
            
            original_projection_state = net._if_project
            net.set_projection(False)
            
            # One pass through training data
            for _, Xtrain, _ in train_loader:
                Xtrain = Xtrain.to(self.device)
                Y_tr = net(Xtrain)
                violations = torch.max(data.get_resid(Xtrain, Y_tr), dim=0).values  # shape = (nineq + neq,)
                self.eps_initial = torch.maximum(violations, self.eps_initial)
            
            # Restore original projection state
            net.set_projection(original_projection_state)

    
    def _linear_decay(self, step, total_steps, initial_value):
        """Linear decay from initial_value to 0"""
        return initial_value * (1 - step / total_steps)
    
    def _harmonic_decay(self, step, total_steps, initial_value):
        """Harmonic decay: eps_init/1, eps_init/2, eps_init/3, ..."""
        return initial_value / (step + 1)
    
    def _linear_harmonic_decay(self, step, total_steps, initial_value):
        """Linear decay for first half, harmonic decay for second half"""
        mid_step = total_steps / 2
        if step < mid_step:
            # First half: linear decay from initial_value to initial_value/2
            return initial_value * (1 - step / (2 * total_steps))
        else:
            # Second half: harmonic decay starting from initial_value/2
            # Adjust step to start from 1 at mid_step
            adjusted_step = step - mid_step + 1
            return (initial_value / 2) / adjusted_step
    
    def get_eps(self, epoch):
        """Get current epsilon value based on epoch"""
        if not self.initialized or epoch < self.start_epoch:
            return torch.tensor(0.0, device=self.device)
        
        decay_step = epoch - self.start_epoch
        if decay_step >= self.decay_epochs:
            return torch.tensor(0.0, device=self.device)
        
        eps = self.decay_fn(decay_step, self.decay_epochs, self.eps_initial)
        return torch.clamp(eps, min=0)

def get_value_or_die(d, key):
    output = d.get(key, None)
    if output is None:
        raise KeyError(f"Key '{key}' not found. Valid keys are: {list(d.keys())}")
    return output

def nangmean(arr, floor=1e-16):
    """Compute geometric mean ignoring NaN values, with a floor to avoid log(0)."""
    # print('Printing array in nangmean:', arr)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan
    return gmean(np.maximum(valid, floor))


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_problem_param_str(dataset_cfg) -> str:
    """
    Generate a string encoding the dataset configuration parameters.
    
    Args:
        dataset_cfg: Dataset configuration dict or DictConfig
        
    Returns:
        String like "var50_eq20_ineq20_seed2025"
    """
    prob_type = dataset_cfg['prob_type'] if isinstance(dataset_cfg, dict) else dataset_cfg.prob_type
    
    # Only generate problem parameter string for problems that have configurable dimensions
    if prob_type in ['cvxqp', 'cvx_qcqp', 'noncvx', 'cbf', 'discrete_cbf']:
        num_var = dataset_cfg['num_var'] if isinstance(dataset_cfg, dict) else dataset_cfg.num_var
        num_eq = dataset_cfg['num_eq'] if isinstance(dataset_cfg, dict) else dataset_cfg.num_eq
        num_ineq = dataset_cfg['num_ineq'] if isinstance(dataset_cfg, dict) else dataset_cfg.num_ineq
        seed = dataset_cfg['seed'] if isinstance(dataset_cfg, dict) else dataset_cfg.seed
        
        return f"var{num_var}_ineq{num_ineq}_eq{num_eq}_seed{seed}"
    else:
        raise NotImplementedError(f"Problem parameter string generation not implemented for problem type: {prob_type}")


def get_hyperparam_str(model_cfg) -> str:
    """
    Generate hyperparameter string encoding model-specific configuration.
    
    Args:
        model_cfg: Model configuration dict or DictConfig
        
    Returns:
        String like "pinv_lambda0.0001" or "cg100_lambda0.0001_adaRel500"
    """
    model_name = model_cfg['name'] if isinstance(model_cfg, dict) else model_cfg.name
    
    if model_name == 'snarenet':
        is_cg = model_cfg['is_cg'] if isinstance(model_cfg, dict) else model_cfg.is_cg
        cg_maxiter = model_cfg['cg_maxiter'] if isinstance(model_cfg, dict) else model_cfg.cg_maxiter
        lambd = model_cfg.get('lambd', 0) if isinstance(model_cfg, dict) else model_cfg.get('lambd', 0)
        adaptive_relaxation = model_cfg['adaptive_relaxation'] if isinstance(model_cfg, dict) else model_cfg.adaptive_relaxation
        decay_epochs = model_cfg['decay_epochs'] if isinstance(model_cfg, dict) else model_cfg.decay_epochs
        rtol = model_cfg.get('rtol', None)
        
        inv_method = f'cg{cg_maxiter}' if is_cg else 'pinv'
        lambd_str = f'lambda{lambd}'
        trust_region = f'tR' if model_cfg.get('trust_region', False) else ''
        ada_relax = f'adaRel{decay_epochs}' if adaptive_relaxation else ''
        rtol_str = f'rtol{rtol}' if rtol is not None else ''
        
        parts = [inv_method, lambd_str, ada_relax, trust_region, rtol_str]
        return "_".join(p for p in parts if p)
    elif model_name == 'hardnetaff':
        adaptive_relaxation = model_cfg.get('adaptive_relaxation', False)
        decay_epochs = model_cfg.get('decay_epochs', None)
        
        ada_relax = f'adaRel{decay_epochs}' if adaptive_relaxation and decay_epochs else ''
        
        return ada_relax if ada_relax else "no_hyperparams"
    elif model_name == 'dc3':
        useTrainCorr = model_cfg.get('useTrainCorr')
        useTestCorr = model_cfg.get('useTestCorr')
        corrTrainSteps = model_cfg.get('corrTrainSteps')
        corrTestMaxSteps = model_cfg.get('corrTestMaxSteps')
        corrLr = model_cfg.get('corrLr')
        corrEps = model_cfg.get('corrEps')
        
        parts = []
        if useTrainCorr and corrTrainSteps is not None:
            parts.append(f'trainCorr{corrTrainSteps}')
        if useTestCorr and corrTestMaxSteps is not None:
            parts.append(f'testCorr{corrTestMaxSteps}')
        if corrLr is not None:
            parts.append(f'corrLr{corrLr}')
        if corrEps is not None:
            parts.append(f'corrEps{corrEps}')
        
        return "_".join(parts) if parts else "no_hyperparams"
    elif model_name in ['cvxpy', 'scipy']:  # Optimizer baseline solvers
        tolerance = model_cfg.get('tolerance', 1e-4)
        return f"tol{tolerance}"
    elif model_name == 'hproj':
        proj_type = model_cfg.get('proj_type', "")
        return proj_type
    elif model_name == 'optnet':
        solver_args = model_cfg.get('solver_args', None)
        if solver_args and solver_args.get('solve_method'):
            return f"solver{solver_args['solve_method']}"
        else:
            return "cvxpy_default"
    else:
        raise NotImplementedError(f"Hyperparameter string generation not implemented for model name: {model_name}")

def get_method_name(cfg: DictConfig) -> str:
    """
    Generate method name based on model, hyperparameters, and soft epochs.
    
        method_name: "{modelName}_{hyperparams}_{softEpochs}"
    """
    model_name = cfg.model.name
    hyperparams = get_hyperparam_str(cfg.model)
    soft_epochs = f"_soft{cfg.soft_epochs}" if cfg.get('soft_epochs', 0) > 0 else ""
    method_name = f"{model_name}_{hyperparams}{soft_epochs}"
    return method_name


def set_wandb_run_info(cfg: DictConfig) -> tuple[str | None, str, str]:
    """
    Generate wandb entity, project, and run name based on configuration.

        wandb_entity: cfg.wandb_workspace if provided, else None (wandb default)
        wandb_project: "{probType}_{problemParams}"
        wandb_name: "seed{seed}_{modelName}_{hyperparams}_soft{softEpochs}"
    """
    wandb_entity = cfg.get('wandb_workspace', None)

    # Set project name by problem type and parameters
    prob_type = cfg.dataset.prob_type
    prob_params = get_problem_param_str(cfg.dataset)
    wandb_project = f"{prob_type}_{prob_params}" if not cfg.get('wandb_project', None) else cfg.wandb_project

    # Set run name by seeds, models, and soft epochs
    seed = cfg.seed
    method_name = get_method_name(cfg)
    wandb_name = f"seed{seed}_{method_name}" if not cfg.get('wandb_name', None) else cfg.wandb_name

    return wandb_entity, wandb_project, wandb_name


def setup_save_directory(cfg: DictConfig) -> str:
    """
    Setup and create the save directory.
    
    Args:
        cfg: Configuration object
    
    Returns:
        save_dir: Path with structure "results/{probType}/{problemParams}/{model.name}/{hyperparams}/seed{N}"
    """
    save_dir = os.path.join(
        'results',
        cfg.dataset.prob_type,
        get_problem_param_str(cfg.dataset),
        cfg.model.name,
        get_hyperparam_str(cfg.model),
        f"seed{cfg.seed}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    return save_dir
    return method_name

def load_data(dataset_cfg, device):
    """
    Load data and put on GPU if needed.
    
    Dataset format: Dictionary with raw coefficient matrices
    - Keys: prob_type, class_input, opt_sols, opt_vals
    
    Args:
        dataset_cfg: Dataset configuration (OmegaConf DictConfig or dict) containing:
            - prob_type: problem type string
            - n_examples: number of examples
            - For cvx_qcqp: num_var, num_ineq, num_eq
        device: torch device to load data onto
    
    Returns:
        data: Problem class instance (CvxQPProblem, QCQP, etc.)
    """
    prob_type = dataset_cfg.prob_type
    
    # Build filename based on problem type
    if prob_type in ['cvx_qcqp', 'cvxqp', 'cbf', 'discrete_cbf', 'noncvx']:
        prob_params = get_problem_param_str(dataset_cfg)
        filename = f"{prob_type}_{prob_params}"
    else:
        raise NotImplementedError(f"Problem type '{prob_type}' not implemented for data loading.")
    
    # Get the project root directory (utils.py is now in utils/ subdirectory, so go up one level)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load from datasets directory (use absolute path from project root)
    filepath = os.path.join(project_root, 'datasets', prob_type, filename)
    dataset_dir = os.path.join(project_root, 'datasets', prob_type)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find dataset for {prob_type} with param {prob_params}.\n"
            f"Expected path: {filepath}"
        )
    
    print(f"Loading dataset from: {filepath}")
    
    # Load the dataset
    with open(filepath, 'rb') as f:
        # Add dataset directory to path for any problem-specific imports
        if dataset_dir not in sys.path:
            sys.path.append(dataset_dir)
        loaded_data = pickle.load(f)
    
    # Construct the appropriate problem class
    if prob_type not in PROBTYPE_TO_CLASS:
        raise NotImplementedError(f"Problem type '{prob_type}' not implemented for new format loading.")
    
    problem_class = PROBTYPE_TO_CLASS[prob_type]
    data = problem_class(**loaded_data['class_input'])
    
    # Set optimal solutions if they exist
    if loaded_data.get('opt_sols') is not None:
        data._Y = torch.tensor(loaded_data['opt_sols'])
    if loaded_data.get('opt_vals') is not None:
        data.opt_vals = torch.tensor(loaded_data['opt_vals'])  # Keep on CPU
    
    # Move all tensors to the specified device using the problem's .to() method
    data.to(device)
    
    return data


def agg_dict(stats, key, value, op='concat'):
    """Aggregate statistics into a dictionary (modifies stats in place)"""
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError(f"Operation '{op}' not implemented")
    else:
        stats[key] = value


def record_stats(stats, runtime, eval_metric, ineq_err, eq_err, opt_gap, prefix, has_opt_vals=False):
    """Record evaluation statistics with a given prefix"""
    make_prefix = lambda x: f"{prefix}_{x}"
    agg_dict(stats, make_prefix('time'), runtime, op='sum')                                     # scalar
    agg_dict(stats, make_prefix('eval'), eval_metric)                                           # (batch_size,)
    agg_dict(stats, make_prefix('ineq_err_max'), np.max(ineq_err, axis=1))                      # (batch_size,) - max across constraints
    agg_dict(stats, make_prefix('ineq_err_gmean'), gmean(np.maximum(ineq_err, 1e-16), axis=1))  # (batch_size,) - gmean across constraints
    agg_dict(stats, make_prefix('ineq_err_nviol'), np.sum(ineq_err > 1e-4, axis=1))             # (batch_size,) - count violations across constraints
    agg_dict(stats, make_prefix('eq_err_max'), np.max(eq_err, axis=1))                          # (batch_size,) - max across constraints
    agg_dict(stats, make_prefix('eq_err_gmean'), gmean(np.maximum(eq_err, 1e-16), axis=1))      # (batch_size,) - gmean across constraints
    agg_dict(stats, make_prefix('eq_err_nviol'), np.sum(eq_err > 1e-4, axis=1))                 # (batch_size,) - count violations across constraints
    
    # Check if instance is feasible: all constraints satisfied (for multiple thresholds)
    for tol in [1e-1, 1e-2, 1e-4]:
        ineq_satisfied = np.all(ineq_err < tol, axis=1)                                          # (batch_size,)
        eq_satisfied = np.all(eq_err < tol, axis=1)                                              # (batch_size,)
        is_feasible = (ineq_satisfied & eq_satisfied).astype(float)                              # (batch_size,)
        tol_str = f"{tol:.0e}".replace('-0', '-')  # Format: 1e-01, 1e-02, 1e-04
        agg_dict(stats, make_prefix(f'is_feasible_{tol_str}'), is_feasible)
    
    if has_opt_vals:
        agg_dict(stats, make_prefix('opt_gap'), opt_gap)                                            # (batch_size,) - optimality gap (NaN if opt_vals not computed)
        agg_dict(stats, make_prefix('nopt'), (opt_gap < 1e-4).astype(float))                        # (batch_size,) - 1 if optimal (gap < 1e-4), 0 otherwise
        # Check if instance is fully solved: optimal + all constraints satisfied (using 1e-4 threshold)
        ineq_satisfied_1e4 = np.all(ineq_err < 1e-4, axis=1)                                         # (batch_size,)
        eq_satisfied_1e4 = np.all(eq_err < 1e-4, axis=1)                                             # (batch_size,)
        opt_satisfied = (opt_gap < 1e-4)                                                             # (batch_size,)
        n_solved = (ineq_satisfied_1e4 & eq_satisfied_1e4 & opt_satisfied).astype(float)             # (batch_size,)
        agg_dict(stats, make_prefix('n_solved'), n_solved)
    return stats


def eval_net(data, X, Ytarget, opt_vals, net, prefix, stats, has_opt_vals=False):
    """Evaluate network on given data and record statistics
    
    Args:
        data: Dataset object
        X: Input tensor
        Ytarget: Target output tensor
        opt_vals: Optimal values (numpy array or tensor on CPU) for computing optimality gap (can be None if not computed)
        net: Neural network to evaluate
        prefix: Prefix for stats keys (e.g., 'valid', 'test')
        stats: Dictionary to record statistics into
        has_opt_vals: Flag indicating if optimal values should be recorded
    """
    with torch.no_grad():
        # Warm up to avoid autotuner pollution
        _ = net(X)

        # Time the forward pass
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            Y = net(X)
            ender.record()
            torch.cuda.synchronize()
            runtime = starter.elapsed_time(ender) / 1000.0  # Convert to seconds
        else:
            start_time = time.time()
            Y = net(X)
            runtime = time.time() - start_time

    # Compute evaluation metrics
    eval_metric = data.get_eval_metric(net, X, Y, Ytarget).detach().cpu().numpy()   # (batch_size,)
    ineq_err = data.get_err_metric1(net, X, Y, Ytarget).detach().cpu().numpy()      # (batch_size, num_ineq)
    eq_err = data.get_err_metric2(net, X, Y, Ytarget).detach().cpu().numpy()        # (batch_size, num_eq)
    
    # Compute optimality gap on CPU: f(Y) - f*
    # Note: opt_vals comes from TensorDataset which converts numpy to tensor, so we convert back
    if has_opt_vals:
        opt_vals_np = opt_vals.numpy()
        opt_gap = eval_metric - opt_vals_np                                              # (batch_size,) - NaN if opt_vals not computed
    else:
        opt_gap = None
    
    return record_stats(stats, runtime, eval_metric, ineq_err, eq_err, opt_gap, prefix, has_opt_vals=has_opt_vals)


def train_net(data, cfg, net_cls, save_dir, net_modifier_fn=None, wandb_run=None, prof = None):
    """
    Train a neural network with the given configuration.
    
    Args:
        data: Dataset object with trainX, trainY, validX, validY, testX, testY
        cfg: Hydra configuration object with all training parameters
        net_cls: Network class to instantiate
        save_dir: Directory to save checkpoints and statistics
        net_modifier_fn: Optional function to modify network at each epoch (e.g., for projection)
    
    Returns:
        net: Trained network
        stats: Training and evaluation statistics
    """
    nepochs = cfg.epochs
    soft_epochs = cfg.soft_epochs
    batch_size = cfg.batch_size
    has_opt_vals = cfg.dataset.get('opt_val_saved', False)

    # Set random seeds for reproducibility
    set_seed(cfg.seed)

    # Create datasets and dataloaders with indices for accessing opt_vals
    train_indices = torch.arange(len(data.trainX))
    valid_indices = torch.arange(len(data.validX))
    test_indices = torch.arange(len(data.testX))
    
    train_dataset = TensorDataset(train_indices, data.trainX, data.trainY)
    valid_dataset = TensorDataset(valid_indices, data.validX, data.validY)
    test_dataset = TensorDataset(test_indices, data.testX, data.testY)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    # Initialize network and optimizer
    net = net_cls(data, cfg)
    net.to(DEVICE)
    solver_opt = optim.Adam(net.parameters(), lr=cfg.learning_rate)

    # Initialize adaptive relaxation handler (only if enabled)
    ar_handler = None
    if cfg.model.get('adaptive_relaxation', False):
        decay_epochs = cfg.model.get('decay_epochs', None)
        decay_schedule = cfg.model.get('decay_schedule', 'linear')  # Default to linear
        if decay_epochs is None:
            decay_epochs = (nepochs - soft_epochs) // 2     # Default to half of remaining epochs
        ar_handler = AdaptiveRelaxation(
            start_epoch=soft_epochs,
            decay_epochs=decay_epochs,
            device=DEVICE,
            decay_fn=decay_schedule
        )

    # Training loop
    stats = {}
    for epoch in range(nepochs):
        epoch_stats = {}

        # Modify network if needed (e.g., enable/disable projection)
        if net_modifier_fn is not None:
            net = net_modifier_fn(net, epoch)

        # Training phase
        net.train()

        # Adaptive relaxation: initialize once at start_epoch, then update eps each epoch
        if ar_handler is not None and epoch >= soft_epochs:
            # Initialize epsilon by one pass over training data with model after soft_epochs
            if not ar_handler.initialized:
                ar_handler.get_init_eps(data, net, train_loader)
            # Get the epsilon for current epoch
            net.set_eps(ar_handler.get_eps(epoch).detach())

            # print(f"Epoch {epoch}: Adaptive relaxation epsilon set to {net.get_eps().cpu().numpy()}")
        
        for idx, Xtrain, Ytarget_train in train_loader:
            Xtrain = Xtrain.to(DEVICE)
            Ytarget_train = Ytarget_train.to(DEVICE)
            
            start_time = time.time()
            solver_opt.zero_grad()
            
            # Compute loss (network uses its own eps attribute)
            train_loss = data.get_train_loss(net, Xtrain, Ytarget_train, cfg)
                
            train_loss.sum().backward()
            if prof is not None:
                prof.step()
            solver_opt.step()
            train_time = time.time() - start_time   # scalar
            
            agg_dict(epoch_stats, 'train_loss', train_loss.detach().cpu().numpy())
            agg_dict(epoch_stats, 'train_time', train_time, op='sum')

        # Print training progress
        print(f"Epoch {epoch}: "
              f"train loss {np.mean(epoch_stats['train_loss']):.2f}, "
              f"train time {epoch_stats['train_time']:.3f}s")

        # Evaluation phase
        if epoch % cfg.eval_freq == 0:
            net.eval()
            for idx, Xvalid, Ytarget_valid in valid_loader:
                Xvalid = Xvalid.to(DEVICE)
                Ytarget_valid = Ytarget_valid.to(DEVICE)
                opt_vals_valid = data.validOptvals[idx] if has_opt_vals else None
                eval_net(data, Xvalid, Ytarget_valid, opt_vals_valid, net, 'valid', epoch_stats, has_opt_vals=has_opt_vals)

            # Print the stats aggregated across the validation instances
            print(f"  eval {np.mean(epoch_stats['valid_eval']):.2f}, "
                  f"ineq_err max {np.max(epoch_stats['valid_ineq_err_max']):.2e}, "
                  f"ineq_err gmean {gmean(np.maximum(epoch_stats['valid_ineq_err_gmean'], 1e-16)):.2e}, "
                  f"ineq_err nviol {np.mean(epoch_stats['valid_ineq_err_nviol']):.2f}, "
                  f"eq_err max {np.max(epoch_stats['valid_eq_err_max']):.2e}, "
                  f"eq_err gmean {gmean(np.maximum(epoch_stats['valid_eq_err_gmean'], 1e-16)):.2e}, "
                  f"eq_err nviol {np.mean(epoch_stats['valid_eq_err_nviol']):.2f}, "
                  f"n_feas_1e-1 {np.sum(epoch_stats['valid_is_feasible_1e-1']):.0f}, "
                  f"n_feas_1e-2 {np.sum(epoch_stats['valid_is_feasible_1e-2']):.0f}, "
                  f"n_feas_1e-4 {np.sum(epoch_stats['valid_is_feasible_1e-4']):.0f}, "
                  f"time {np.sum(epoch_stats['valid_time']):.3f}s")
            if wandb_run is not None:
                wandb_log = {
                    "epoch": epoch,
                    "train/loss": np.mean(epoch_stats['train_loss']),
                    "train/time": epoch_stats['train_time'],
                    "valid/eval": np.mean(epoch_stats['valid_eval']),
                    "valid/ineq_err_max": np.max(epoch_stats['valid_ineq_err_max']),
                    "valid/ineq_err_gmean": gmean(np.maximum(epoch_stats['valid_ineq_err_gmean'], 1e-16)),
                    "valid/ineq_err_nviol": np.mean(epoch_stats['valid_ineq_err_nviol']),
                    "valid/eq_err_max": np.max(epoch_stats['valid_eq_err_max']),
                    "valid/eq_err_gmean": gmean(np.maximum(epoch_stats['valid_eq_err_gmean'], 1e-16)),
                    "valid/eq_err_nviol": np.mean(epoch_stats['valid_eq_err_nviol']),
                    "valid/n_feasible_1e-1": np.sum(epoch_stats['valid_is_feasible_1e-1']),
                    "valid/n_feasible_1e-2": np.sum(epoch_stats['valid_is_feasible_1e-2']),
                    "valid/n_feasible_1e-4": np.sum(epoch_stats['valid_is_feasible_1e-4']),
                    "valid/last_iter_taken": net.get_iter_taken() if hasattr(net, 'get_iter_taken') else None,
                }
                if ar_handler is not None:
                    wandb_log.update({
                        "adaptive_relaxation_eps_max": torch.max(net.get_eps()).item()
                    })                    
                if 'valid_opt_gap' in epoch_stats:
                    wandb_log.update({
                        "valid/opt_gap_gmean": gmean(np.maximum(epoch_stats['valid_opt_gap'], 1e-16)),
                        "valid/opt_gap_max": np.nanmax(epoch_stats['valid_opt_gap']),
                        "valid/nopt": np.nansum(epoch_stats['valid_nopt']),
                        "valid/n_solved": np.nansum(epoch_stats['valid_n_solved']),
                    })
                wandb_run.log(wandb_log, step=epoch)
                
        # Aggregate statistics
        if cfg.save_all_stats:
            if epoch == 0:
                for key in epoch_stats.keys():
                    stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)
            else:
                for key in epoch_stats.keys():
                    stats[key] = np.concatenate(
                        (stats[key], np.expand_dims(np.array(epoch_stats[key]), axis=0))
                    )
        else:
            # Only keep latest epoch stats
            for key in epoch_stats.keys():
                stats[key] = np.expand_dims(np.array(epoch_stats[key]), axis=0)

        # Save checkpoints periodically
        if epoch % cfg.results_save_freq == 0:
            with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
                pickle.dump(stats, f)
            with open(os.path.join(save_dir, f'net_epoch{epoch}.dict'), 'wb') as f:
                torch.save(net.state_dict(), f)
            with open(os.path.join(save_dir, f'net_current.dict'), 'wb') as f:
                torch.save(net.state_dict(), f)

            if wandb_run is not None:
                artifact = wandb.Artifact(
                    name=wandb_run.name,
                    type='model',
                    metadata={'run_id': wandb_run.id}
                )
                artifact.add_file(os.path.join(save_dir, 'net_current.dict'), name='net_current.dict')
                artifact.add_file(os.path.join(save_dir, f'net_epoch{epoch}.dict'), name=f'net_epoch{epoch}.dict')


    # Save final results
    with open(os.path.join(save_dir, 'stats.dict'), 'wb') as f:
        pickle.dump(stats, f)
    with open(os.path.join(save_dir, 'net.dict'), 'wb') as f:
        torch.save(net.state_dict(), f)

    # Save model as wandb artifact
    if wandb_run is not None:
        artifact = wandb.Artifact(
            name=wandb_run.name,
            type='model',
            metadata={'run_id': wandb_run.id}
        )
        artifact.add_file(os.path.join(save_dir, 'net.dict'), name='net.dict')
        wandb_run.log_artifact(artifact)
        print(f"Model uploaded to wandb as artifact: {wandb_run.name}")
    
    print(f"\nTraining complete! Results saved to: {save_dir}")
    return net, stats


def test_net(data, cfg, net, save_dir, wandb_run=None):
    """
    Test a trained neural network on the test set.
    
    Args:
        data: Dataset object with testX, testY
        cfg: Configuration object
        net: Trained network to evaluate
        save_dir: Directory to save test statistics
        wandb_run: Optional wandb run object for logging
    
    Returns:
        test_stats: Test evaluation statistics
    """
    print("\n" + "=" * 80)
    print("Testing model on test set...")
    print("=" * 80)

    has_opt_vals  = cfg.dataset.get('opt_val_saved', False)
    
    # Create test dataloader with indices
    test_indices = torch.arange(len(data.testX))
    test_dataset = TensorDataset(test_indices, data.testX, data.testY)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    
    # Evaluate on test set
    test_stats = {}
    net.eval()
    for idx, Xtest, Ytarget_test in test_loader:
        Xtest = Xtest.to(DEVICE)
        Ytarget_test = Ytarget_test.to(DEVICE)
        opt_vals_test = data.testOptvals[idx] if has_opt_vals else None
        eval_net(data, Xtest, Ytarget_test, opt_vals_test, net, 'test', test_stats, has_opt_vals=has_opt_vals)
    
    # Print test results
    print(f"\nTest Results:")
    result_str = (f"  eval {np.mean(test_stats['test_eval']):.2f}, "
                  f"ineq_err max {gmean(np.maximum(test_stats['test_ineq_err_max'], 1e-16)):.2e}, "
                  f"ineq_err gmean {gmean(np.maximum(test_stats['test_ineq_err_gmean'], 1e-16)):.2e}, "
                  f"ineq_err nviol {np.mean(test_stats['test_ineq_err_nviol']):.2f}, "
                  f"eq_err max {gmean(np.maximum(test_stats['test_eq_err_max'], 1e-16)):.2e}, "
                  f"eq_err gmean {gmean(np.maximum(test_stats['test_eq_err_gmean'], 1e-16)):.2e}, "
                  f"eq_err nviol {np.mean(test_stats['test_eq_err_nviol']):.2f}, "
                  f"n_feas_1e-1 {np.sum(test_stats['test_is_feasible_1e-1']):.0f}, "
                  f"n_feas_1e-2 {np.sum(test_stats['test_is_feasible_1e-2']):.0f}, "
                  f"n_feas_1e-4 {np.sum(test_stats['test_is_feasible_1e-4']):.0f}, ")
    if 'test_opt_gap' in test_stats:
        result_str += (f"opt_gap gmean {gmean(np.maximum(test_stats['test_opt_gap'], 1e-16)):.2e}, "
                       f"opt_gap max {np.nanmax(test_stats['test_opt_gap']):.2e}, ")
    result_str += f"time {np.sum(test_stats['test_time']):.3f}s"
    print(result_str)
    
    # Log test results to wandb
    if wandb_run is not None:
        wandb_log = {
            "test/eval": np.mean(test_stats['test_eval']),
            "test/ineq_err_max": np.max(test_stats['test_ineq_err_max']),
            "test/ineq_err_gmean": gmean(np.maximum(test_stats['test_ineq_err_gmean'], 1e-16)),
            "test/ineq_err_nviol": np.mean(test_stats['test_ineq_err_nviol']),
            "test/eq_err_max": np.max(test_stats['test_eq_err_max']),
            "test/eq_err_gmean": gmean(np.maximum(test_stats['test_eq_err_gmean'], 1e-16)),
            "test/eq_err_nviol": np.mean(test_stats['test_eq_err_nviol']),
            "test/n_feasible_1e-1": np.sum(test_stats['test_is_feasible_1e-1']),
            "test/n_feasible_1e-2": np.sum(test_stats['test_is_feasible_1e-2']),
            "test/n_feasible_1e-4": np.sum(test_stats['test_is_feasible_1e-4']),
            "test/time": np.sum(test_stats['test_time']),
        }
        if 'test_opt_gap' in test_stats:
            wandb_log.update({
                "test/opt_gap_gmean": gmean(np.maximum(test_stats['test_opt_gap'], 1e-16)),
                "test/opt_gap_max": np.nanmax(test_stats['test_opt_gap']),
                "test/nopt": np.nansum(test_stats['test_nopt']),
                "test/n_solved": np.nansum(test_stats['test_n_solved']),
            })
        wandb_run.log(wandb_log)
    
    # Match dimensions of stats to that of training stats (axis 0 is for epoch)
    for key in test_stats.keys():
        test_stats[key] = np.expand_dims(np.array(test_stats[key]), axis=0)
    
    # Save test statistics
    with open(os.path.join(save_dir, 'test_stats.dict'), 'wb') as f:
        pickle.dump(test_stats, f)
    
    print(f"Test results saved to: {os.path.join(save_dir, 'test_stats.dict')}")
    return test_stats
