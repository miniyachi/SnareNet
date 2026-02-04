import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HardNetAff(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._cfg = cfg
        self._if_project = False
        self._eps = torch.zeros(data.nineq + data.neq, device=DEVICE)  # Epsilon for adaptive relaxation (num_constraints,)
        layer_sizes = [data.encoded_xdim, self._cfg.model.hidden_size, self._cfg.model.hidden_size]
        output_dim = data.ydim

        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        # layers = reduce(operator.add,
        #     [[nn.Linear(a,b), nn.ReLU()]
        #         for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        
        layers += [nn.Linear(layer_sizes[-1], output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self._net = nn.Sequential(*layers)
    
    def set_projection(self, val=True):
        """set whether to do projection or not"""
        self._if_project = val
    
    def set_eps(self, eps):
        """Set epsilon for adaptive relaxation (used during training)"""
        self._eps = eps
    
    def get_eps(self):
        """Get current epsilon value"""
        return self._eps

    def apply_projection(self, f, x):
        """project f to satisfy bl<=Af<=bu"""
        A, bl, bu = self._data.get_coefficients(x)
        
        if self._cfg.dataset.prob_type in ['cvxqp', 'noncvx']: # efficient computation for input-independent A (for fair comparison with DC3)
            # A = A[0,:,:] A returned as a 2D tensor for opt
            Af = A @ f[:,:,None]
            eps_expanded = self._eps.unsqueeze(0).unsqueeze(-1)  # Shape: [1, num_constraints, 1]
            return f + (torch.linalg.pinv(A) @ (nn.ReLU()(bl[:,:,None] - eps_expanded - Af) - nn.ReLU()(Af - bu[:,:,None] - eps_expanded)))[:,:,0]
        
        # listsq is more stable than pinv
        Af = A @ f[:,:,None]
        eps_expanded = self._eps.unsqueeze(0).unsqueeze(-1)  # Shape: [1, num_constraints, 1]
        return f + torch.linalg.lstsq(A.unsqueeze(0).expand(f.shape[0], -1, -1), nn.ReLU()(bl[:,:,None] - eps_expanded - Af) - nn.ReLU()(Af - bu[:,:,None] - eps_expanded)).solution[:,:,0]


    def forward(self, x, isTest=False):
        encoded_x = self._data.encode_input(x)
        out = self._net(encoded_x)

        if self._if_project:
            out = self.apply_projection(out, x)
        
        return out
