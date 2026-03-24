"""  
Train and evaluate neural network models for various constrained neural networks.
Results are logged to Wandb and saved locally.

Usage:
    # Train with specific configuration overrides
    python3 run_experiment.py model=snarenet dataset=noncvx seed=123
    
    # Train with custom epochs and soft constraint warmup
    python3 run_experiment.py epochs=200 soft_epochs=50
"""

import os
import wandb

# Get list of visible GPUs
cuda_visible = os.getenv('CUDA_VISIBLE_DEVICES')
if cuda_visible:
    visible_gpu_list = [int(gpu.strip()) for gpu in cuda_visible.split(',')]
    print(f"Visible GPUs: {visible_gpu_list}")
else:
    print("CUDA_VISIBLE_DEVICES not set, using default GPU behavior")

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function, schedule
torch.set_default_dtype(torch.float64)

import os
from setproctitle import setproctitle
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Import our utils and model
from utils.utils import load_data, train_net, test_net, set_wandb_run_info, setup_save_directory, set_seed
from constants import MODELNAME_TO_CLASS


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def save_profiling_results(profiler, sort_by_keyword: str, output_file: str = 'profiling_results_solve.txt'):
    """
    Save profiling results to a text file with multiple sorting options.
    
    Args:
        profiler: PyTorch profiler object
        sort_by_keyword: Primary sorting keyword (e.g., 'cuda_time_total')
        output_file: Output file path for profiling results
    """
    with open(output_file, 'w') as file:
        file.write(profiler.key_averages().table(sort_by=sort_by_keyword, row_limit=100))
        file.write("\n\n")  # Add separator between tables
        file.write(profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=100))
        file.write("\n\n")  # Add separator between tables
        file.write(profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=100))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_exp(cfg: DictConfig):
    """Run experiment with Hydra configuration."""
    
    print("=" * 80)
    print("Running experiment with configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    model_name = cfg.model.name
    
    # Set Python process title
    setproctitle(f'{model_name}-{cfg.dataset.prob_type}')
    
    # Seed before load_data so np.random.choice in problem constructors is reproducible
    set_seed(cfg.seed)

    # Load data class
    data = load_data(cfg.dataset, DEVICE)
    
    # Get Hydra's output directory (for logs)
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    print(f"Hydra logs to: {hydra_output_dir}")
    
    # Set up save directory and save config
    save_dir = setup_save_directory(cfg=cfg)
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saving results to: {save_dir}")
    
    # Define projection modifier
    soft_epochs = cfg.soft_epochs
    def modify_net(net, epoch):
        if epoch < soft_epochs:
            net.set_repair(False)
        else:
            net.set_repair(True)
        return net
    
    # Set wandb project and run names
    wandb_entity, wandb_project, wandb_name = set_wandb_run_info(cfg)

    # Set up profiling if enabled
    profiling = cfg.get('profiling', False)
    if profiling:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sort_by_keyword = "cuda_time_total"
        my_schedule = schedule(wait=5, warmup=1, active=4)

    with wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_name) as run:
        # Log config to wandb
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        
        # Train model
        model_class = MODELNAME_TO_CLASS.get(model_name, None)
        if model_class is None:
            raise ValueError(f"Unknown model name: {model_name}")
        
        if profiling:
            print(f"Profiling enabled for {model_name} training.")
            with profile(activities=activities, schedule=my_schedule , profile_memory=True, record_shapes=True) as profi:
                net, stats = train_net(data, cfg, model_class, save_dir, net_modifier_fn=modify_net, wandb_run=run, prof=profi)
            save_profiling_results(profi, sort_by_keyword)
        else:
            net, stats = train_net(data, cfg, model_class, save_dir, net_modifier_fn=modify_net, wandb_run=run)
        
        # Test model on test set
        test_stats = test_net(data, cfg, net, save_dir, wandb_run=run)
        
        # Finish wandb run
        run.finish()


if __name__ == '__main__':
    run_exp()
