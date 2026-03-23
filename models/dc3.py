import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from models.base_model import BaseModel


class DC3RepairLayer(nn.Module):
    """
    Gradient-based correction layer for constraint enforcement.

    Applies completion (for equality constraints) followed by iterative
    gradient-descent correction. Uses fewer steps during training and runs
    until convergence at test time.
    """

    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._cfg = cfg

    def forward(self, out, x):
        cfg = self._cfg.model
        if cfg.useCompl:
            out = self._data.complete_partial(x, out)
        out = self._grad_steps(x, out)
        if not self.training:
            out = self._extra_grad_steps(x, out)
        return out

    def _grad_steps(self, X, Y):
        cfg = self._cfg.model
        if not cfg.useTrainCorr:
            return Y
        lr = cfg.corrLr
        num_steps = cfg.corrTrainSteps
        momentum = cfg.corrMomentum
        partial_corr = cfg.corrMode == 'partial'
        if partial_corr and not cfg.useCompl:
            assert False, "Partial correction not available without completion."
        Y_new = Y
        old_Y_step = 0
        for _ in range(num_steps):
            if partial_corr:
                Y_step = self._data.get_ineq_partial_grad(X, Y_new)
            else:
                Y_step = self._data.get_resid_grad(X, Y_new)
            new_Y_step = lr * Y_step + momentum * old_Y_step
            Y_new = Y_new - new_Y_step
            old_Y_step = new_Y_step
        return Y_new

    def _extra_grad_steps(self, X, Y):
        """Test-time correction: run until convergence (no grad needed)."""
        cfg = self._cfg.model
        if not cfg.useTestCorr:
            return Y
        lr = cfg.corrLr
        eps_converge = cfg.corrEps
        max_steps = cfg.corrTestMaxSteps
        momentum = cfg.corrMomentum
        partial_corr = cfg.corrMode == 'partial'
        if partial_corr and not cfg.useCompl:
            assert False, "Partial correction not available without completion."
        Y_new = Y
        i = 0
        old_Y_step = 0
        with torch.no_grad():
            while (i == 0 or torch.max(self._data.get_resid(X, Y_new)) > eps_converge) and i < max_steps:
                if partial_corr:
                    Y_step = self._data.get_ineq_partial_grad(X, Y_new)
                else:
                    Y_step = self._data.get_resid_grad(X, Y_new)
                new_Y_step = lr * Y_step + momentum * old_Y_step
                Y_new = Y_new - new_Y_step
                old_Y_step = new_Y_step
                i += 1
        return Y_new


class DC3(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._cfg = cfg
        self._if_project = False

        output_dim = data.ydim - data.neq if cfg.model.useCompl else data.ydim
        self._base = BaseModel(data, cfg, output_dim=output_dim)
        self._repair = DC3RepairLayer(data, cfg)

    def set_projection(self, val=True):
        self._if_project = val

    def forward(self, x):
        encoded_x = self._data.encode_input(x)
        out = self._base(encoded_x)

        if self._if_project:
            out = self._repair(out, x)

        return out
