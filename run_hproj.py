"""
Run H-Proj baseline on constrained optimization problems.

Two-stage pipeline:
  Stage 1: Train homeomorphic INN mapping (unsupervised) — saves homeoproj_module.pth
  Stage 2: Train BaseModel predictor (unsupervised soft-penalty loss) — saves homeoprojnet.pth

Then evaluate HomeoProjNet (BaseModel + INN bisection repair) on the test set.
Results are saved locally and logged to Wandb.

Usage:
    python3 run_hproj.py dataset=cvx_qcqp seed=123
    python3 run_hproj.py dataset=noncvx seed=123
    python3 run_hproj.py dataset=cvx_qcqp model.skip_mapping=true seed=123  # reuse mapping

Adapted from:
    hproj_repo/training_all.py (train_mdh_mapping, train_nn_solver)
    hproj_repo/training_utils.py (training loop)
    Source: https://arxiv.org/abs/2306.09292
"""

import os
import pickle
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import wandb
from scipy.stats import gmean
from setproctitle import setproctitle
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.utils import (
    load_data, setup_save_directory,
    set_wandb_run_info, set_seed, eval_net
)
from models.hproj import (
    HomeoProjNet, INN,
    ActNorm, LUInvertibleMM, MADE, FlowSigmoid
)
from models.base_model import BaseModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


###############################################################################
# Helper
###############################################################################

def _scale_to_bounds(y_partial, data):
    """Map y_partial ∈ [0,1]^{|partial_vars|} to [L_partial, U_partial]."""
    L = data.L[data.partial_vars]
    U = data.U[data.partial_vars]
    return y_partial * (U - L) + L


###############################################################################
# Stage 1: train homeomorphic INN mapping
# Adapted from hproj_repo/training_all.py::train_mdh_mapping
###############################################################################

def train_hproj_mapping(data, cfg, model_save_dir):
    """Train INN homeomorphic mapping (Stage 1).

    Trains the INN to map [0,1]^{|partial_vars|} onto the feasible set of
    partial variables, conditioned on input parameters X.  The loss maximises
    volume (log-det), minimises constraint penalty, distortion, and transport.

    Saves the trained INN to <model_save_dir>/homeoproj_module.pth.
    """
    paras = cfg.model.mapping_para
    n_dim = len(data.partial_vars)
    t_dim = data.encoded_xdim
    num_layer = paras.num_layer

    # Build INN (same architecture as HomeoProjRepairModule)
    flow_modules = []
    for _ in range(num_layer):
        flow_modules += [
            ActNorm(num_inputs=n_dim),
            LUInvertibleMM(num_inputs=n_dim),
            ActNorm(num_inputs=n_dim),
            MADE(num_inputs=n_dim, num_hidden=n_dim // 2, num_cond_inputs=t_dim),
        ]
    flow_modules += [ActNorm(num_inputs=n_dim), FlowSigmoid()]
    model = INN(flow_modules).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=paras.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=paras.lr_decay_step, gamma=paras.lr_decay
    )

    # Sample initial x_train in [0,1]^n_dim (body of the unit cube)
    x_train_tensor = torch.rand(paras.n_samples, n_dim, device=DEVICE)
    # Sample t_train uniformly in [input_L, input_U]
    t_train_tensor = torch.rand(paras.t_samples, t_dim, device=DEVICE)
    t_train_tensor = t_train_tensor * (data.input_U - data.input_L) + data.input_L

    bias_val = 0.5  # midpoint of [0, 1] bound
    bias_tensor = torch.full((paras.batch_size, n_dim), bias_val, device=DEVICE)

    model.train()
    for n in range(paras.total_iteration):
        optimizer.zero_grad()

        xi = np.random.choice(paras.n_samples, paras.batch_size, replace=True)
        ti = np.random.choice(paras.t_samples, paras.batch_size, replace=True)
        x_input = x_train_tensor[xi]
        t_input = t_train_tensor[ti]

        xt, logdet, logdis = model(x_input, t_input)
        trans = torch.mean((x_input - xt) ** 2, dim=1, keepdim=True)

        xt_scale = _scale_to_bounds(xt, data)
        xt_full = data.complete_partial(t_input, xt_scale)

        # Constraint penalty: sum of non-negative residuals
        violation = data.get_resid(t_input, xt_full)
        penalty = torch.sum(violation, dim=-1, keepdim=True)

        loss = (
            -torch.mean(logdet) / n_dim
            + paras.penalty_coefficient * torch.mean(penalty)
            + paras.distortion_coefficient * torch.mean(logdis)
            + paras.transport_coefficient * torch.mean(trans)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        if n % 1000 == 0 and n > 0:
            model.eval()
            with torch.no_grad():
                x0, _, _ = model(bias_tensor, t_train_tensor[: paras.batch_size])
                x0_scale = _scale_to_bounds(x0, data)
                x0_full = data.complete_partial(t_train_tensor[: paras.batch_size], x0_scale)
                pen0 = torch.sum(data.get_resid(t_train_tensor[: paras.batch_size], x0_full), dim=-1)
            print(
                f"[Stage 1] Iter {n}/{paras.total_iteration} | "
                f"vol={-logdet.mean().item()/n_dim:.4f} | "
                f"penalty={penalty.mean().item():.4f} | "
                f"test_pen={pen0.mean().item():.8f}"
            )
            model.train()

    save_path = os.path.join(model_save_dir, "homeoproj_module.pth")
    torch.save(model, save_path)
    print(f"[Stage 1] INN mapping saved to: {save_path}")
    return model


###############################################################################
# Stage 2: train BaseModel predictor with soft-penalty unsupervised loss
# Adapted from hproj_repo/training_all.py::train_nn_solver (unsupervise branch)
###############################################################################

def train_hproj_solver(data, cfg, model_save_dir):
    """Train BaseModel predictor (Stage 2).

    The predictor outputs partial variables in [0,1] (sigmoid applied to BaseModel).
    Trained with an unsupervised objective: minimise weighted sum of
    objective value, inequality violation, and equality violation on randomly
    sampled X from [input_L, input_U].

    Saves the trained BaseModel to <model_save_dir>/homeoprojnet.pth.
    """
    paras = cfg.model.nn_para

    base_net = BaseModel(data, cfg, output_dim=len(data.partial_vars)).to(DEVICE)

    optimizer = torch.optim.Adam(base_net.parameters(), lr=paras.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=paras.lr_decay_step, gamma=paras.lr_decay
    )

    base_net.train()
    for i in range(paras.total_iteration + 1):
        optimizer.zero_grad()

        # Sample random X from [input_L, input_U] (unsupervised training)
        Xbatch = torch.rand(paras.batch_size, data.encoded_xdim, device=DEVICE)
        Xbatch = Xbatch * (data.input_U - data.input_L) + data.input_L

        Y_partial = torch.sigmoid(base_net(Xbatch))        # (batch, |partial_vars|) in (0,1)
        Y_scaled = _scale_to_bounds(Y_partial, data)        # (batch, |partial_vars|) in [L, U]
        Y_full = data.complete_partial(Xbatch, Y_scaled)    # (batch, ydim)

        training_obj = data.get_eval_metric(None, Xbatch, Y_full, None)         # (batch,)
        ineq_penalty = torch.sum(data.get_err_metric1(None, Xbatch, Y_full, None), dim=1)  # (batch,)
        eq_penalty = torch.sum(data.get_err_metric2(None, Xbatch, Y_full, None), dim=1)    # (batch,)

        train_loss = (
            paras.objWeight * training_obj
            + paras.softWeightInEqFrac * ineq_penalty
            + paras.softWeightEqFrac * eq_penalty
        )
        train_loss.mean().backward()
        optimizer.step()
        scheduler.step()

        if i % 1000 == 0 and i > 0:
            base_net.eval()
            with torch.no_grad():
                Yp = torch.sigmoid(base_net(data.testX))
                Ys = _scale_to_bounds(Yp, data)
                Yf = data.complete_partial(data.testX, Ys)
                test_obj = data.get_eval_metric(None, data.testX, Yf, None).mean().item()
                test_ineq = data.get_err_metric1(None, data.testX, Yf, None).max().item()
                test_eq = data.get_err_metric2(None, data.testX, Yf, None).max().item()
            print(
                f"[Stage 2] Iter {i}/{paras.total_iteration} | "
                f"obj={test_obj:.4f} | ineq_max={test_ineq:.4e} | eq_max={test_eq:.4e}"
            )
            base_net.train()

    save_path = os.path.join(model_save_dir, "homeoprojnet.pth")
    torch.save(base_net, save_path)
    print(f"[Stage 2] NN solver saved to: {save_path}")
    return base_net


###############################################################################
# Main driver
###############################################################################

@hydra.main(version_base=None, config_path="configs", config_name="config_hproj")
def run_hproj(cfg: DictConfig):
    """Run H-Proj baseline with Hydra configuration."""
    print("=" * 80)
    print("Running H-Proj baseline with configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    setproctitle(f"hproj-{cfg.dataset.prob_type}")
    set_seed(cfg.seed)

    # Load data
    data = load_data(cfg.dataset, DEVICE)

    # Setup directories
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    print(f"Hydra logs to: {hydra_output_dir}")

    save_dir = setup_save_directory(cfg)
    model_save_dir = os.path.join(save_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saving results to: {save_dir}")

    wandb_entity, wandb_project, wandb_name = set_wandb_run_info(cfg)

    with wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_name) as run:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

        # ------------------------------------------------------------------
        # Stage 1: train homeomorphic mapping
        # ------------------------------------------------------------------
        mapping_path = os.path.join(model_save_dir, "homeoproj_module.pth")
        if not cfg.model.skip_mapping:
            print("\n" + "=" * 80)
            print("Stage 1: Training homeomorphic INN mapping...")
            print("=" * 80)
            train_hproj_mapping(data, cfg, model_save_dir)
        else:
            print(f"Skipping Stage 1 (skip_mapping=true). Loading from: {mapping_path}")

        # ------------------------------------------------------------------
        # Stage 2: train predictor
        # ------------------------------------------------------------------
        solver_path = os.path.join(model_save_dir, "homeoprojnet.pth")
        if not cfg.model.skip_nn:
            print("\n" + "=" * 80)
            print("Stage 2: Training BaseModel predictor...")
            print("=" * 80)
            train_hproj_solver(data, cfg, model_save_dir)
        else:
            print(f"Skipping Stage 2 (skip_nn=true). Loading from: {solver_path}")

        # ------------------------------------------------------------------
        # Build HomeoProjNet and load trained weights
        # ------------------------------------------------------------------
        print("\nBuilding HomeoProjNet and loading weights...")

        # Build net without loading mapping (mapping_weights_path is null in yaml),
        # then load both weights explicitly.
        net = HomeoProjNet(data, cfg).to(DEVICE)
        net._repair.load_mapping(mapping_path)

        # Load BaseModel predictor weights
        trained_base = torch.load(solver_path, map_location=DEVICE, weights_only=False)
        net._base.load_state_dict(trained_base.state_dict())

        net.set_repair(True)
        net.eval()

        # ------------------------------------------------------------------
        # Evaluate on test set
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("Evaluating on test set...")
        print("=" * 80)

        has_opt_vals = cfg.dataset.get("opt_val_saved", False)
        opt_vals_test = data.testOptvals if has_opt_vals else None

        results = {}
        results = eval_net(
            data,
            data.testX,
            data.testY,
            opt_vals_test,
            net,
            "test",
            results,
            has_opt_vals=has_opt_vals,
        )

        # Save results
        with open(os.path.join(save_dir, "results.dict"), "wb") as f:
            pickle.dump(results, f)

        # ------------------------------------------------------------------
        # Print summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("Results Summary:")
        print("=" * 80)
        print(f"Test objective (mean):      {results['test_eval'].mean():.6f}")
        print(f"Test ineq violation (max):  {results['test_ineq_err_max'].mean():.2e}")
        print(f"Test eq violation (max):    {results['test_eq_err_max'].mean():.2e}")
        print(f"Test time (total):          {np.sum(results['test_time']):.4f} seconds")
        print("=" * 80)

        # ------------------------------------------------------------------
        # Log to wandb
        # ------------------------------------------------------------------
        wandb_log = {
            "test/eval": np.mean(results["test_eval"]),
            "test/ineq_err_max": np.max(results["test_ineq_err_max"]),
            "test/ineq_err_gmean": gmean(np.maximum(results["test_ineq_err_gmean"], 1e-16)),
            "test/ineq_err_nviol": np.mean(results["test_ineq_err_nviol"]),
            "test/eq_err_max": np.max(results["test_eq_err_max"]),
            "test/eq_err_gmean": gmean(np.maximum(results["test_eq_err_gmean"], 1e-16)),
            "test/eq_err_nviol": np.mean(results["test_eq_err_nviol"]),
            "test/n_feasible_1e-1": np.sum(results["test_is_feasible_1e-1"]),
            "test/n_feasible_1e-2": np.sum(results["test_is_feasible_1e-2"]),
            "test/n_feasible_1e-4": np.sum(results["test_is_feasible_1e-4"]),
            "test/time": np.sum(results["test_time"]),
        }
        if has_opt_vals:
            wandb_log.update({
                "test/opt_gap_gmean": gmean(np.maximum(results["test_opt_gap"], 1e-16)),
                "test/opt_gap_max": np.nanmax(results["test_opt_gap"]),
                "test/nopt": np.nansum(results["test_nopt"]),
                "test/n_solved": np.nansum(results["test_n_solved"]),
            })
        run.log(wandb_log)

        # Upload model weights as wandb artifact
        artifact = wandb.Artifact(
            name=run.name,
            type='model',
            metadata={'run_id': run.id}
        )
        artifact.add_file(mapping_path, name='homeoproj_module.pth')
        artifact.add_file(solver_path, name='homeoprojnet.pth')
        run.log_artifact(artifact)
        print(f"✓ Model weights uploaded to wandb as artifact: {run.name}")

        print(f"\n✓ Results saved to: {save_dir}/results.dict")
        print(f"✓ Logged to wandb project: {wandb_project}, run: {wandb_name}")

        run.finish()


if __name__ == "__main__":
    run_hproj()
