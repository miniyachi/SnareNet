import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce

class DC3(nn.Module):
    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._cfg = cfg
        self._if_project = False
        layer_sizes = [data.encoded_xdim, self._cfg.model.hidden_size, self._cfg.model.hidden_size]
        output_dim = data.ydim - data.neq if self._cfg.model.useCompl else data.ydim
        
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
        """Set whether to do completion and correction or not"""
        self._if_project = val

    def grad_steps(self, X, Y):
        take_grad_steps = self._cfg.model.useTrainCorr
        if take_grad_steps:
            lr = self._cfg.model.corrLr
            num_steps = self._cfg.model.corrTrainSteps
            momentum = self._cfg.model.corrMomentum
            partial_var = self._cfg.model.useCompl
            partial_corr = True if self._cfg.model.corrMode == 'partial' else False
            if partial_corr and not partial_var:
                assert False, "Partial correction not available without completion."
            Y_new = Y
            old_Y_step = 0
            for i in range(num_steps):
                if partial_corr:
                    Y_step = self._data.get_ineq_partial_grad(X, Y_new)
                else:
                    Y_step = self._data.get_resid_grad(X, Y_new)
                new_Y_step = lr * Y_step + momentum * old_Y_step
                Y_new = Y_new - new_Y_step
                old_Y_step = new_Y_step
            return Y_new
        else:
            return Y
    
    def extra_grad_steps(self, X, Y):
        """Used only at test time, so let PyTorch avoid building the computational graph"""
        take_grad_steps = self._cfg.model.useTestCorr
        if take_grad_steps:
            lr = self._cfg.model.corrLr
            eps_converge = self._cfg.model.corrEps
            max_steps = self._cfg.model.corrTestMaxSteps
            momentum = self._cfg.model.corrMomentum
            partial_var = self._cfg.model.useCompl
            partial_corr = True if self._cfg.model.corrMode == 'partial' else False
            if partial_corr and not partial_var:
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
        else:
            return Y

    def forward(self, x):
        encoded_x = self._data.encode_input(x)
        out = self._net(encoded_x)
        
        if self._if_project:
            # completion
            if self._cfg.model.useCompl:
                out = self._data.complete_partial(x, out)
            # correction
            out = self.grad_steps(x, out)
            if not self.training:
                out = self.extra_grad_steps(x, out)
        
        return out