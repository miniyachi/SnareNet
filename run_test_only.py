"""
Test trained model(s) by loading from Wandb artifacts.

Usage:
    # Test a model using default config
    python3 run_test_only.py
    
    # Test with specific configuration overrides
    python3 run_test_only.py model=snarenet dataset=portfolio seed=123
    
    # Test with custom wandb project/run name
    python3 run_test_only.py wandb.project=my_project wandb.run_name=my_run
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
torch.set_default_dtype(torch.float64)

import os
from setproctitle import setproctitle
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Import our utils and model
from utils.utils import load_data, test_net, set_wandb_project_run_name, setup_save_directory, set_seed
from constants import MODELNAME_TO_CLASS


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_from_wandb(cfg: DictConfig, data, wandb_project: str, wandb_run_name: str):
    """
    Load a trained model from wandb artifacts.
    
    Args:
        cfg: Configuration object with model specifications
        data: Dataset object (to get problem dimensions)
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name (used as artifact name)
    
    Returns:
        net: Loaded neural network model
    """
    model_name = cfg.model.name
    model_class = MODELNAME_TO_CLASS.get(model_name, None)
    if model_class is None:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Initialize model architecture
    net = model_class(data, cfg)
    net.to(DEVICE)
    
    # Download model artifact from wandb
    api = wandb.Api()
    
    # Construct artifact path: project/artifact_name:version
    # By default, use the latest version (v0 is typically the first/only version)
    artifact_path = f"{wandb_project}/{wandb_run_name}:latest"
    
    print(f"Downloading model from wandb artifact: {artifact_path}")
    try:
        artifact = api.artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        
        # Load model weights
        model_path = os.path.join(artifact_dir, 'net.dict')
        net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded successfully from: {model_path}")
        
    except Exception as e:
        print(f"Error loading model from wandb: {e}")
        print(f"Attempted to load from: {artifact_path}")
        raise
    
    return net


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_test_exp(cfg: DictConfig):
    """Run testing experiment with Hydra configuration."""
    
    print("=" * 80)
    print("Running TEST experiment with configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    model_name = cfg.model.name
    
    # Set Python process title
    setproctitle(f'{model_name}-{cfg.dataset.prob_type}-TEST')
    
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Load data class
    data = load_data(cfg.dataset, DEVICE)
    
    # Get Hydra's output directory (for logs)
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    print(f"Hydra logs to: {hydra_output_dir}")
    
    # Set up save directory for test results
    save_dir = setup_save_directory(cfg=cfg)
    test_save_dir = os.path.join(save_dir, 'test_only')
    os.makedirs(test_save_dir, exist_ok=True)
    
    with open(os.path.join(test_save_dir, 'test_config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saving test results to: {test_save_dir}")
    
    # Set wandb project and run names
    wandb_project, wandb_name = set_wandb_project_run_name(cfg)
    
    # Load model from wandb
    print(f"\nLoading model from wandb project: {wandb_project}, run: {wandb_name}")
    net = load_model_from_wandb(cfg, data, wandb_project, wandb_name)
    
    # Enable projection for testing (hard constraints)
    if hasattr(net, 'set_projection'):
        net.set_projection(True)
        print("Projection enabled for testing.")
    
    # Ensure model is in eval mode (should already be set, but double-check)
    net.eval()
    print(f"Model training mode: {net.training} (should be False)")
    
    # Create a new wandb run for testing
    test_wandb_project = f"{wandb_project}_test"
    test_wandb_name = f"{wandb_name}_test"
    with wandb.init(project=test_wandb_project, name=test_wandb_name, job_type="test") as run:
        # Log config to wandb
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        
        # Test model on test set
        test_stats = test_net(data, cfg, net, test_save_dir, wandb_run=run)
        
        # Finish wandb run
        run.finish()
    
    print(f"\nTesting complete! Results saved to: {test_save_dir}")


if __name__ == '__main__':
    run_test_exp()
