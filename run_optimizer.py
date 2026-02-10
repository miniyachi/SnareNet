"""
Run traditional optimization solver baselines on test data for constrained optimization problems.
Results are logged to Wandb and saved locally.

Usage:
    # Run specific solver on a dataset
    python3 run_optimizer.py dataset=noncvx
"""

import os
import pickle
import torch
torch.set_default_dtype(torch.float64)

import wandb
import numpy as np
from scipy.stats import gmean
from setproctitle import setproctitle
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Import our utils
from utils.utils import load_data, record_stats, setup_save_directory, set_wandb_project_run_name

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@hydra.main(version_base=None, config_path="configs", config_name="config_opt")
def run_optimizer(cfg: DictConfig):
    """Run optimization solver baseline with Hydra configuration."""
    
    print("=" * 80)
    print("Running optimizer baseline with configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set process title
    setproctitle(f'{cfg.model.name}-{cfg.dataset.prob_type}')
    
    # Load data
    data = load_data(cfg.dataset, DEVICE)
    
    # Get Hydra's output directory (for logs)
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    print(f"Hydra logs to: {hydra_output_dir}")
    
    # Setup save directory and save config
    save_dir = setup_save_directory(cfg)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saving results to: {save_dir}")
    
    # Set wandb project and run names
    wandb_project, wandb_name = set_wandb_project_run_name(cfg)
    
    with wandb.init(project=wandb_project, name=wandb_name) as run:
        # Log config to wandb
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        
        # Run optimizer on test set
        print("\n" + "="*80)
        print(f"Running {cfg.model.name} solver on test set...")
        print("="*80)
        
        test_time_total, test_time_parallel = data.opt_solve(
            indices=data.test_indices,
            solver_type=cfg.model.name, 
            tol=cfg.model.tolerance
        )
        
        # Extract test solutions from data.Y
        Ytest_opt = data.Y[data.test_indices].detach().cpu().numpy()
        
        # Save solutions if requested
        if cfg.get('save_solutions', False):
            sol_path = os.path.join(save_dir, f"{cfg.dataset.prob_type}_dataset_ex{cfg.dataset.n_examples}_sol")
            with open(sol_path, 'wb') as f:
                pickle.dump(Ytest_opt, f)
            print(f"✓ Solutions saved to: {sol_path}")
        
        # Compute metrics
        print("\nComputing metrics...")
        opt_results = get_opt_results(
            {}, 
            data, 
            data.testX, 
            torch.tensor(Ytest_opt).to(DEVICE),
            data.testY, 
            test_time_parallel,
            'test'
        )
        opt_results.update(dict([('test_time_total', test_time_total)]))
        
        # Save results
        with open(os.path.join(save_dir, 'results.dict'), 'wb') as f:
            pickle.dump(opt_results, f)
        
        # Print results summary
        print("\n" + "="*80)
        print("Results Summary:")
        print("="*80)
        print(f"Test objective (mean): {opt_results['test_eval'].mean():.6f}")
        print(f"Test ineq violation (max): {opt_results['test_ineq_err_max'].mean():.2e}")
        print(f"Test eq violation (max): {opt_results['test_eq_err_max'].mean():.2e}")
        print(f"Test time (total): {test_time_total:.4f} seconds")
        print(f"Test time (avg per sample): {test_time_parallel:.4f} seconds")
        print("="*80)
        
        # Log to wandb
        wandb_log = {
            "test/eval": np.mean(opt_results['test_eval']),
            "test/ineq_err_max": np.max(opt_results['test_ineq_err_max']),
            "test/ineq_err_gmean": gmean(np.maximum(opt_results['test_ineq_err_gmean'], 1e-16)),
            "test/ineq_err_nviol": np.mean(opt_results['test_ineq_err_nviol']),
            "test/eq_err_max": np.max(opt_results['test_eq_err_max']),
            "test/eq_err_gmean": gmean(np.maximum(opt_results['test_eq_err_gmean'], 1e-16)),
            "test/eq_err_nviol": np.mean(opt_results['test_eq_err_nviol']),
            "test/n_feasible_1e-1": np.sum(opt_results['test_is_feasible_1e-1']),
            "test/n_feasible_1e-2": np.sum(opt_results['test_is_feasible_1e-2']),
            "test/n_feasible_1e-4": np.sum(opt_results['test_is_feasible_1e-4']),
            "test/time": test_time_total,
            "test/time_per_sample": test_time_parallel,
        }
        if 'test_opt_gap' in opt_results:
            wandb_log.update({
                "test/opt_gap_gmean": gmean(np.maximum(opt_results['test_opt_gap'], 1e-16)),
                "test/opt_gap_max": np.nanmax(opt_results['test_opt_gap']),
                "test/nopt": np.nansum(opt_results['test_nopt']),
                "test/n_solved": np.nansum(opt_results['test_n_solved']),
            })
        run.log(wandb_log)
        
        print(f"\n✓ Results saved to: {save_dir}/results.dict")
        print(f"✓ Logged to wandb project: {wandb_project}, run: {wandb_name}")
        
        # Finish wandb run
        run.finish()


def get_opt_results(results, data, X, Y, Ytarget, time, prefix):
    """Compute optimization results metrics."""
    eval_metric = data.get_eval_metric(None, X, Y, Ytarget).detach().cpu().numpy()
    err1 = data.get_err_metric1(None, X, Y, Ytarget).detach().cpu().numpy()
    err2 = data.get_err_metric2(None, X, Y, Ytarget).detach().cpu().numpy()
    opt_gap = eval_metric - data.testOptvals.cpu().numpy()  # dataset to call run_optimizer must have opt vals
    
    return record_stats(results, time, eval_metric, err1, err2, opt_gap, prefix, has_opt_vals=True)


if __name__ == '__main__':
    run_optimizer()
