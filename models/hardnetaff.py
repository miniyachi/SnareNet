import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from models.base_model import BaseModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HardNetAffRepairLayer(nn.Module):
    """
    Affine projection layer for constraint enforcement: bl <= Ay <= bu.

    Projects network outputs onto the affine constraint set using a least-squares
    (or pinv-based) correction step, with optional adaptive relaxation.
    """

    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._cfg = cfg
        self.register_buffer('_eps', torch.zeros(data.nineq + data.neq))

    def set_eps(self, eps):
        self._eps = eps

    def get_eps(self):
        return self._eps

    def forward(self, f, x):
        """Project f to satisfy bl <= Af <= bu (with relaxation eps)."""
        A, bl, bu = self._data.get_coefficients(x)

        if self._cfg.dataset.prob_type in ['cvxqp', 'noncvx']:
            # Efficient path for input-independent A (e.g. cvxqp, noncvx)
            Af = f @ A.T                                          # (batch, m)
            r = nn.ReLU()(bl - self._eps - Af) - nn.ReLU()(Af - bu - self._eps)
            return f + r @ torch.linalg.pinv(A).T                # (batch, n)

        # General path: lstsq is more numerically stable than pinv
        # A: (batch, m, n), f: (batch, n)
        Af = (A @ f.unsqueeze(-1)).squeeze(-1)                   # (batch, m)
        r = nn.ReLU()(bl - self._eps - Af) - nn.ReLU()(Af - bu - self._eps)
        return f + torch.linalg.lstsq(A, r.unsqueeze(-1)).solution.squeeze(-1)


class HardNetAff(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._if_project = False

        self._base = BaseModel(data, cfg)
        self._repair = HardNetAffRepairLayer(data, cfg)

    def set_projection(self, val=True):
        self._if_project = val

    def set_eps(self, eps):
        self._repair.set_eps(eps)

    def get_eps(self):
        return self._repair.get_eps()

    def forward(self, x, isTest=False):
        encoded_x = self._data.encode_input(x)
        out = self._base(encoded_x)

        if self._if_project:
            out = self._repair(out, x)

        return out
