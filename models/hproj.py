"""
H-Proj baseline: HomeoProjNet (BaseModel + homeomorphic bisection repair module).

INN architecture components (get_mask, MaskedLinear, MADE, ActNorm, LUInvertibleMM,
Sigmoid, INN) are copied from:
    hproj_repo/flows_utils.py — https://arxiv.org/abs/2306.09292

homeo_bisection is adapted from:
    hproj_repo/training_utils.py — same source.
"""
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

from models.base_model import BaseModel

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


###############################################################################
# INN components — copied from hproj_repo/flows_utils.py
###############################################################################

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0))


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, cond_in_features=None, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Sequential(nn.Linear(cond_in_features, 2 * out_features))
        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            w, b = self.cond_linear(cond_inputs).chunk(2, 1)
            output = output * w + b
        return output


nn.MaskedLinear = MaskedLinear


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation (https://arxiv.org/abs/1502.03509)."""
    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None, act='relu'):
        super().__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')
        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.trunk = nn.Sequential(
            act_func(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            nn.MaskedLinear(num_hidden, num_inputs * 2, output_mask),
        )

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a
        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a


class LUInvertibleMM(nn.Module):
    """Invertible 1×1 conv via LU decomposition (https://arxiv.org/abs/1807.03039)."""
    def __init__(self, num_inputs):
        super().__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = scipy.linalg.lu(self.W.numpy())
        self.P = torch.tensor(P)
        self.L_param = nn.Parameter(torch.tensor(L))
        self.U_param = nn.Parameter(torch.tensor(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.tensor(sign_S)
        self.log_S = nn.Parameter(torch.tensor(log_S))
        self.I = torch.eye(self.L_param.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L_param.device):
            self.L_mask = self.L_mask.to(self.L_param.device)
            self.U_mask = self.U_mask.to(self.L_param.device)
            self.I = self.I.to(self.L_param.device)
            self.P = self.P.to(self.L_param.device)
            self.sign_S = self.sign_S.to(self.L_param.device)

        L = self.L_param * self.L_mask + self.I
        U = self.U_param * self.U_mask + torch.diag(self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(W), -self.log_S.unsqueeze(0).repeat(inputs.size(0), 1)


class ActNorm(nn.Module):
    """Activation normalization from Glow (https://arxiv.org/abs/1807.03039)."""
    def __init__(self, num_inputs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if not self.initialized:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (inputs - self.bias) * torch.exp(self.weight), \
                   self.weight.unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(-self.weight) + self.bias, \
                   -self.weight.unsqueeze(0).repeat(inputs.size(0), 1)


class FlowSigmoid(nn.Module):
    """Flow-compatible sigmoid; maps to (-0.1, 1.1)."""
    def __init__(self):
        super().__init__()
        self.lower = -0.1
        self.upper = 1.1

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            y = torch.sigmoid(inputs)
            scale_y = y * (self.upper - self.lower) + self.lower
            return scale_y, torch.log(y * (1 - y) * (self.upper - self.lower))
        else:
            x = (inputs - self.lower) / (self.upper - self.lower)
            return torch.log(x / (1 - x)), -torch.log((inputs - inputs ** 2) / (self.upper - self.lower))


class INN(nn.Module):
    """Sequence of invertible flow layers."""
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, t):
        m, _ = x.shape
        log_det = torch.zeros(m, device=x.device)
        log_dis = torch.zeros(m, device=x.device)
        for flow in self.flows:
            x, ls = flow.forward(x, t)
            ld = ls.sum(-1)
            dis = torch.max(ls, -1)[0] - torch.min(ls, -1)[0]
            log_det += ld.view(-1)
            log_dis += dis.view(-1)
        return x, log_det, log_dis

    def backward(self, z, t):
        for flow in self.flows[::-1]:
            z, _ = flow.forward(z, t, mode='inverse')
        return z, None, None


###############################################################################
# Homeomorphic bisection — adapted from hproj_repo/training_utils.py
###############################################################################

def _homeo_bisection(inn, data, cfg, f, x):
    """
    Binary search in INN latent space to find a feasible point.

    Args:
        inn:  trained INN module
        data: dataset object (provides complete_partial and get_resid)
        cfg:  Hydra config (reads proj_para from cfg.model.proj_para)
        f:    infeasible prediction in [0,1]^{|partial_vars|} (batch, |partial_vars|)
        x:    raw input parameters (batch, xdim), used to condition INN

    Returns:
        xt_full: feasible full solution (batch, ydim)
    """
    steps = cfg.model.proj_para.corrTestMaxSteps
    eps = cfg.model.proj_para.corrEps
    bis_step = cfg.model.proj_para.corrBis
    bound_mean = 0.5  # midpoint of [0, 1] cube (mapping_para.bound = [0, 1])

    L_partial = data.L[data.partial_vars]
    U_partial = data.U[data.partial_vars]

    with torch.no_grad():
        bias = torch.tensor(bound_mean, device=f.device).view(1, 1)
        x_latent, _, _ = inn.backward(f, x)
        alpha_upper = torch.ones([f.shape[0], 1], device=f.device)
        alpha_lower = torch.zeros([f.shape[0], 1], device=f.device)

        for k in range(steps):
            alpha = (1 - bis_step) * alpha_lower + bis_step * alpha_upper
            xt, _, _ = inn(alpha * (x_latent - bias) + bias, x)
            # Scale from [0,1] to [L, U] for partial variables
            xt_scale = xt * (U_partial - L_partial) + L_partial
            xt_full = data.complete_partial(x, xt_scale)
            # Feasibility check via combined constraint residual
            violation = data.get_resid(x, xt_full)
            penalty = torch.max(violation, dim=1)[0]
            sub_feasible = penalty < eps
            sub_infeasible = ~sub_feasible
            alpha_lower[sub_feasible] = alpha[sub_feasible]
            alpha_upper[sub_infeasible] = alpha[sub_infeasible]
            if (alpha_upper - alpha_lower).max() < 1e-2:
                break

        xt, _, _ = inn(alpha_lower * (x_latent - bias) + bias, x)
        xt_scale = xt * (U_partial - L_partial) + L_partial
        xt_full = data.complete_partial(x, xt_scale)

    return xt_full


###############################################################################
# Model classes
###############################################################################

class HomeoProjRepairModule(nn.Module):
    """
    Homeomorphic bisection projection using a pre-trained INN.

    Not end-to-end trainable — weights are loaded from a checkpoint produced
    by Stage 1 (train_hproj_mapping in run_hproj.py).

    Source: adapted from hproj_repo/training_utils.py::homeo_bisection
            and hproj_repo/flows_utils.py (INN architecture)
    """
    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._cfg = cfg

        n_dim = len(data.partial_vars)
        t_dim = data.encoded_xdim
        num_layer = cfg.model.mapping_para.num_layer

        flow_modules = []
        for _ in range(num_layer):
            flow_modules += [
                ActNorm(num_inputs=n_dim),
                LUInvertibleMM(num_inputs=n_dim),
                ActNorm(num_inputs=n_dim),
                MADE(num_inputs=n_dim, num_hidden=n_dim // 2, num_cond_inputs=t_dim),
            ]
        flow_modules += [ActNorm(num_inputs=n_dim), FlowSigmoid()]
        self.inn = INN(flow_modules)

    def load_mapping(self, path):
        # Stage 1 saves the INN directly: torch.save(inn_model, path)
        self.inn = torch.load(path, map_location=DEVICE, weights_only=False)
        self.inn.eval()

    def forward(self, f, x):
        return _homeo_bisection(self.inn, self._data, self._cfg, f, x)


class HomeoProjNet(nn.Module):
    """
    H-Proj baseline: BaseModel predictor + homeomorphic bisection repair module.

    Pattern mirrors HardNetAff (base + repair). Stage 1 trains the INN mapping;
    Stage 2 trains the BaseModel predictor. Both weights are loaded before inference.

    Source: hproj_repo — https://arxiv.org/abs/2306.09292
    """
    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._if_repair = True

        # Reuse BaseModel with the same hidden_size as other models.
        # output_dim = |partial_vars| since equality completion is in the repair module.
        self._base = BaseModel(data, cfg, output_dim=len(data.partial_vars))
        self._repair = HomeoProjRepairModule(data, cfg)

        mapping_path = cfg.model.get('mapping_weights_path', None)
        if mapping_path:
            self._repair.load_mapping(mapping_path)

    def set_repair(self, val=True):
        self._if_repair = val

    def forward(self, x, isTest=False):
        encoded_x = self._data.encode_input(x)
        # Apply sigmoid so base output lives in (0,1) — required for INN bisection
        out = torch.sigmoid(self._base(encoded_x))

        if self._if_repair:
            out = self._repair(out, x)

        return out
