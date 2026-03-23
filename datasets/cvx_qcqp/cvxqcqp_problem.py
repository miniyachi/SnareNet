import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
import time
from functools import partial
from tqdm import tqdm


class QCQP:
    """
        Learning a single solver (input:x, output: solution)
        for the following quadratically constrained quadratic problems (QCQPs) for all x:
        minimize_y 1/2 * y^T Q y + p^T y
        s.t.       y^T H_i y + G_i^T y <= h_i, for i = 1, ... , m
                   Ay =  x
    """
    def __init__(self, Q, p, A, X, G, H, h, L, U, valid_frac=0.0833, test_frac=0.0833):
        # n = variable dimension; m = number of inequality constraints; b = batch size
        self._Q = torch.tensor(Q)   # size = [n, n]
        self._p = torch.tensor(p)   # size = [n]
        self._A = torch.tensor(A)   # size = [num_eq, n]
        self._G = torch.tensor(G)   # size = [m, n]
        self._H = torch.tensor(H)   # size = [m, n, n]
        self._h = torch.tensor(h)   # size = [m]
        self._X = torch.tensor(X)   # size = [b, num_eq]
        self._encoded_xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = A.shape[0]
        self._nineq = h.shape[0]
        self._valid_frac = valid_frac
        self._test_frac = test_frac

        # Optimal solutions and optimal values from opt_solver [Optional: prereset as NaN]
        self._Y = torch.full((X.shape[0], Q.shape[0]), float('nan'))    # size = [b, n]
        self.opt_vals = torch.full((X.shape[0],), float('nan'))  # size = [b] - kept on CPU for optimality gap computation

        ### For Pytorch - track device of tensors
        self._device = self._Q.device

        ### Determine independent/dependent variable split (for DC3 only)
        det = 0
        i = 0
        while abs(det) < 0.0001 and i < 100:
            self._partial_vars = np.random.choice(self._ydim, self._ydim - self._neq, replace=False)
            self._other_vars = np.setdiff1d(np.arange(self._ydim), self._partial_vars)
            det = torch.det(self._A[:, self._other_vars])
            i += 1
        if i == 100:
            raise Exception(f"Could not find independent variable set with non-singular A submatrix after 100 tries.")
        else:
            self._A_partial = self._A[:, self._partial_vars]
            self._A_other_inv = torch.inverse(self._A[:, self._other_vars])

    def __str__(self):
        return f'QCQP-{self.num}'

    def to(self, device):
        """Move all tensors to specified device and update derived tensors"""
        self._device = device
        # Move all primary tensors
        self._Q = self._Q.to(device)
        self._p = self._p.to(device)
        self._A = self._A.to(device)
        self._G = self._G.to(device)
        self._H = self._H.to(device)
        self._h = self._h.to(device)
        self._X = self._X.to(device)
        self._Y = self._Y.to(device)
        # Recreate derived tensors on the new device
        self._A_partial = self._A[:, self._partial_vars]
        self._A_other_inv = torch.inverse(self._A[:, self._other_vars])
        return self

    @property
    def Q(self):
        return self._Q

    @property
    def p(self):
        return self._p

    @property
    def A(self):
        return self._A

    @property
    def X(self):
        return self._X
    
    @property
    def G(self):
        return self._G
    
    @property
    def H(self):
        return self._H

    @property
    def h(self):
        return self._h

    @property
    def Y(self):
        return self._Y

    @property
    def Q_np(self):
        return self.Q.detach().cpu().numpy()

    @property
    def p_np(self):
        return self.p.detach().cpu().numpy()

    @property
    def A_np(self):
        return self.A.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

    @property
    def G_np(self):
        return self.G.detach().cpu().numpy()

    @property
    def H_np(self):
        return self.H.detach().cpu().numpy()

    @property
    def h_np(self):
        return self.h.detach().cpu().numpy()

    @property
    def Y_np(self):
        return self.Y.detach().cpu().numpy()

    @property
    def encoded_xdim(self):
        return self._encoded_xdim

    @property
    def ydim(self):
        return self._ydim

    @property
    def num(self):
        return self._num

    @property
    def nineq(self):
        return self._nineq

    @property
    def neq(self):
        return self._neq

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    @property
    def train_indices(self):
        return np.arange(int(self.num * self.train_frac))

    @property
    def valid_indices(self):
        start = int(self.num * self.train_frac)
        end = int(self.num * (self.train_frac + self.valid_frac))
        return np.arange(start, end)

    @property
    def test_indices(self):
        start = int(self.num * (self.train_frac + self.valid_frac))
        return np.arange(start, self.num)

    @property
    def trainX(self):
        return self.X[self.train_indices]

    @property
    def validX(self):
        return self.X[self.valid_indices]

    @property
    def testX(self):
        return self.X[self.test_indices]

    @property
    def trainY(self):
        return self.Y[self.train_indices]

    @property
    def validY(self):
        return self.Y[self.valid_indices]

    @property
    def testY(self):
        return self.Y[self.test_indices]

    @property
    def trainOptvals(self):
        return self.opt_vals[self.train_indices]

    @property
    def validOptvals(self):
        return self.opt_vals[self.valid_indices]

    @property
    def testOptvals(self):
        return self.opt_vals[self.test_indices]

    @property
    def device(self):
        return self._device

    ########## For DC3 only ##########
    @property
    def partial_vars(self):
        return self._partial_vars

    @property
    def other_vars(self):
        return self._other_vars

    @property
    def partial_unknown_vars(self):
        return self._partial_vars
    ####################################
    
    def encode_input(self, X):
        return X

    def calc_Y(self, X):
        return NotImplementedError

    def evaluate(self, X, Y):
        return (0.5*(Y@self.Q)*Y + self.p*Y).sum(dim=1)

    def get_lower_bound(self, X):
        """lower bound for inequality constraints bl(x)<=g(x,y)<=bu(x)"""
        bl_ineq = torch.full((X.shape[0], self._nineq), -float('inf'), device=self.device)   # size = [b, m]
        return torch.cat((bl_ineq, X), dim=1)   # size = [b, m + num_eq]
    
    def get_upper_bound(self, X):
        """upper bound for inequality constraints bl(x)<=g(x,y)<=bu(x)"""
        bu_ineq = self.h.expand(X.shape[0], -1) # size = [b, m]
        return torch.cat((bu_ineq, X), dim=1)   # size = [b, m + num_eq]
    
    def get_ineq_res(self, X, Y):
        # X size = [b, num_eq], Y size = [b, n], H size = [m, n, n]
        quadratic = torch.einsum('bj,ijk,bk->bi', Y, self.H, Y)  # size = [b, m]
        linear = torch.einsum('bj,ij->bi', Y, self.G)   # size = [b, m]
        return torch.clamp(quadratic + linear - self.h, min = 0)  # size = [b, m]
    
    def get_eq_res(self, X, Y):
        # X size = [b, num_eq], Y size = [b, n], H size = [m, n, n]
        return torch.abs(Y@self.A.T - X)  # size = [b, num_eq]
    
    def get_resid(self, X, Y):
        ineq_resid = self.get_ineq_res(X, Y)  # size = [b, m]
        eq_resid = self.get_eq_res(X, Y)    # size = [b, num_eq]
        return torch.cat((ineq_resid, eq_resid), dim=1) # size = [b, m + num_eq]

    def get_train_loss(self, net, X, Ytarget, cfg):
        Y = net(X)
        main_loss = self.evaluate(X, Y) # size = [b]
        regularization = torch.norm(self.get_resid(X, Y), dim=1)**2  # size = [b]
        return main_loss + cfg.soft_weight * regularization   # size = [b]
    
    def get_eval_metric(self, net, X, Y, Ytarget):
        return self.evaluate(X, Y)
    
    def get_err_metric1(self, net, X, Y, Ytarget):
        """compute ineq error"""
        return self.get_ineq_res(X, Y)
    
    def get_err_metric2(self, net, X, Y, Ytarget):
        """compute eq error"""
        return self.get_eq_res(X, Y)

    def get_lower_upper_bounds(self, X):
        return self.get_lower_bound(X), self.get_upper_bound(X)

    def jacobian(self, Y):
        """Jacobian of the constraints w.r.t. y"""
        J_ineq = 2 * torch.einsum('bk,ijk->bij', Y, self.H) + self.G.repeat(Y.shape[0], 1, 1)    # size = [b, m, n]
        J_eq = self.A.repeat(Y.shape[0], 1, 1)  # size = [b, num_eq, n]
        return torch.cat((J_ineq, J_eq), dim=1)  # size = [b, m + num_eq, n]

    def g(self, Y):
        """constraint function g(x,y)"""
        ineq_g = torch.einsum('bj,ijk,bk->bi', Y, self.H, Y) + torch.einsum('bj,ij->bi', Y, self.G)  # size = [b, m]
        eq_g = torch.einsum('bj,ij->bi', Y, self.A)  # size = [b, num_eq]
        return torch.cat((ineq_g, eq_g), dim=1)  # size = [b, m + num_eq]

    def get_jacobian(self, X):
        """Getter function for the Jacobian of the constraints w.r.t. y"""
        return self.jacobian

    def get_g(self, X):
        """Getter function for the constraint function g(x,y)"""
        return self.g

    def get_coefficients(self, X):
        """coefficients for inequality constraints bl(x)<=A_eff(x)y<=bu(x)"""
        bl = self.get_lower_bound(X)
        bu = self.get_upper_bound(X)
        return bl, bu, self.g, self.jacobian
        

    ########## For DC3 only ##########
    def get_resid_grad(self, X, Y):
        """gradient of ||resid||^2 - analytical gradient for quadratic constraints"""
        # Inequality residuals and gradients
        ineq_res = self.get_ineq_res(X, Y)  # size [b, m]
        # Gradient of each inequality constraint: 2*H_i*y + G_i
        quad_grad = 2 * torch.einsum('ijk,bk->bij', self.H, Y)  # size [b, m, n]
        linear_grad = self.G.unsqueeze(0).expand(Y.shape[0], -1, -1)  # size [b, m, n]
        ineq_constraint_grad = quad_grad + linear_grad  # size [b, m, n]
        # Gradient of inequality penalty: sum_i 2*res_i*(2*H_i*y + G_i)
        grad_ineq = 2 * (ineq_res.unsqueeze(-1) * ineq_constraint_grad).sum(dim=1)  # size [b, n]
        
        # Equality residuals and gradients
        eq_res_signed = Y @ self.A.T - X  # size [b, num_eq] (with sign)
        # Gradient of each equality constraint: A_j
        # Gradient of equality penalty: sum_j 2*eq_res_j * A_j
        grad_eq = 2 * eq_res_signed @ self.A  # size [b, n]
        
        # Total gradient
        return grad_ineq + grad_eq
    
    def get_ineq_partial_grad(self, X, Y):
        """gradient for DC3 with equality completion - analytical gradient for quadratic constraints"""
        # Compute the full y from partial variables (for gradient computation)
        # y_other = (x - y_partial @ A_partial.T) @ A_other_inv.T
        
        # Compute inequality residuals: clamp(y^T H_i y + G_i^T y - h_i, 0)
        ineq_res = self.get_ineq_res(X, Y)  # size [b, m]
        
        # Compute gradient of each constraint w.r.t. full y: 2*H_i*y + G_i
        # For quadratic term: gradient is 2*H_i*y
        # H has shape [m, n, n], Y has shape [b, n]
        quad_grad = 2 * torch.einsum('ijk,bk->bij', self.H, Y)  # size [b, m, n]
        # For linear term: gradient is G_i
        linear_grad = self.G.unsqueeze(0).expand(Y.shape[0], -1, -1)  # size [b, m, n]
        # Full gradient of constraint i w.r.t. y
        constraint_grad = quad_grad + linear_grad  # size [b, m, n]
        
        # Gradient of penalty w.r.t. y: sum_i 2*res_i*(2*H_i*y + G_i)
        # size [b, m, 1] * [b, m, n] -> [b, m, n] -> sum over m -> [b, n]
        grad_full = 2 * (ineq_res.unsqueeze(-1) * constraint_grad).sum(dim=1)  # size [b, n]
        
        # Now apply chain rule for partial variables
        # dy/dy_partial = [I; -A_other_inv @ A_partial]
        # So: grad_partial = grad_full[:, partial] - grad_full[:, other] @ A_other_inv @ A_partial
        grad_partial = grad_full[:, self.partial_vars] - grad_full[:, self.other_vars] @ (self._A_other_inv @ self._A_partial)
        
        # Build full gradient vector
        grad = torch.zeros(X.shape[0], self.ydim, device=X.device)
        grad[:, self.partial_vars] = grad_partial
        grad[:, self.other_vars] = - (grad_partial @ self._A_partial.T) @ self._A_other_inv.T
        return grad

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        Y = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._A_partial.T) @ self._A_other_inv.T
        return Y
    
    def get_cvxpy_projection_layer(self):
        """
        Create a differentiable cvxpy layer for projecting onto the constraint set.
        
        Solves: minimize ||y - y_hat||^2
                subject to y^T H_i y + G_i^T y <= h_i for i=1,...,m
                           Ay = x
        
        Returns a CvxpyLayer that can be used in PyTorch models.
        """
        # Define cvxpy variables
        y = cp.Variable(self.ydim)
        y_hat = cp.Parameter(self.ydim)  # Neural network output
        x = cp.Parameter(self.neq)  # Input parameter
        
        # Convert to numpy for cvxpy
        A_np = self.A_np
        H_np = self.H_np
        G_np = self.G_np
        h_np = self.h_np
        
        # Define constraints
        constraints = [A_np @ y == x]  # Equality constraints
        
        # Quadratic inequality constraints: y^T H_i y + G_i^T y <= h_i
        for i in range(self.nineq):
            constraints.append(cp.quad_form(y, H_np[i]) + G_np[i] @ y <= h_np[i])
        
        # Define objective: minimize ||y - y_hat||^2
        objective = cp.Minimize(cp.sum_squares(y - y_hat))
        
        # Create problem
        problem = cp.Problem(objective, constraints)
        
        # Create differentiable layer
        cvxpy_layer = CvxpyLayer(problem, parameters=[y_hat, x], variables=[y])
        
        return cvxpy_layer
    ####################################
    
    ###### For optimization solver

    def opt_solve(self, indices=None, solver_type='gurobipy', tol=1e-4, verbose=False):
        Q, p, A, H, G, h = self.Q_np, self.p_np, self.A_np, self.H_np, self.G_np, self.h_np

        # If indices not provided, solve for all instances
        if indices is None:
            indices = np.arange(self.num)

        X_np = self.X_np[indices]
        total_time = 0

        for idx, Xi in enumerate(tqdm(X_np, desc=f"Solving {solver_type}")):
            if solver_type == 'gurobipy':
                model = gp.Model("qcqp")
                model.setParam('OutputFlag', 1 if verbose else 0)
                model.setParam('FeasibilityTol', tol)
                model.setParam('OptimalityTol', tol)
                model.setParam('NumericFocus', 3)   # Max numerical precision (1-3)

                y = model.addMVar(self._ydim, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

                # Equality constraints: Ay = x
                model.addMConstr(A, y, '=', Xi)

                # Quadratic inequality constraints: y^T H_i y + G_i^T y <= h_i
                # H[i] are PSD (diagonal with non-negative entries), so no NonConvex flag needed
                for i in range(self._nineq):
                    model.addConstr(y @ H[i] @ y + G[i] @ y <= h[i])

                # Objective: minimize 0.5 * y^T Q y + p^T y
                model.setObjective(0.5 * y @ Q @ y + p @ y, GRB.MINIMIZE)

                start_time = time.time()
                model.optimize()
                end_time = time.time()

                if model.SolCount > 0:
                    self._Y[indices[idx]] = torch.tensor(y.X, device=self.device)
                    self.opt_vals[indices[idx]] = model.ObjVal
                else:
                    print(f"Warning: No solution found for index {indices[idx]}. Status: {model.status}")

                total_time += (end_time - start_time)

            elif solver_type == 'cvxpy':
                y = cp.Variable(self._ydim)
            
                # Equality constraints: Ay = x
                constraints = [A @ y == Xi]
            
                # Inequality constraints: y^T H_i y + G_i^T y <= h_i
                for i in range(self._nineq):
                    Ht = H[i]
                    Gt = G[i]
                    ht = h[i]
                    constraints.append(cp.quad_form(y, Ht) + Gt.T @ y <= ht)
            
                # Objective: minimize 0.5 * y^T Q y + p^T y
                prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                                    constraints)
            
                start_time = time.time()
                prob.solve(solver=cp.GUROBI)
                end_time = time.time()
            
                # Update in-place
                self._Y[indices[idx]] = torch.tensor(y.value, device=self.device)
                self.opt_vals[indices[idx]] = prob.value
            
                total_time += (end_time - start_time)

            else:
                raise NotImplementedError

        return total_time, total_time/len(X_np)
