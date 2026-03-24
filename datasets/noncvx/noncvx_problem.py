import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import time
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from tqdm import tqdm
from scipy.optimize import minimize

###################################################################
# LEARNING NON-CONVEX OPTIMIZATION SOLVER
###################################################################

class NonCvxProblem:
    """
        Learning a single solver (input:x, output: solution)
        for the following non-convex quadratic problems for all x:
        minimize_y 1/2 * y^T Q y + p^T sin(y)
        s.t.       Ay <= b, Cy = x
    """
    def __init__(self, Q, p, A, b, C, X, valid_frac=0.0833, test_frac=0.0833):
        self._Q = torch.tensor(Q)
        self._p = torch.tensor(p)
        self._A = torch.tensor(A)
        self._b = torch.tensor(b)
        self._C = torch.tensor(C)
        self._X = torch.tensor(X)
        self._encoded_xdim = X.shape[1]
        self._ydim = Q.shape[0]
        self._num = X.shape[0]
        self._neq = C.shape[0]
        self._nineq = A.shape[0]
        self._valid_frac = valid_frac
        self._test_frac = test_frac

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
            det = torch.det(self._C[:, self._other_vars])
            i += 1
        if i == 100:
            raise Exception(f"Could not find independent variable set with non-singular C submatrix after 100 tries.")
        else:
            self._C_partial = self._C[:, self._partial_vars]
            self._C_other_inv = torch.inverse(self._C[:, self._other_vars])

        # Variable and input bounds used by the H-Proj baseline only.
        # NonCvxProblem has no explicit variable bounds; using large placeholders.
        self.L = torch.full((self._ydim,), -100.0)                # (ydim,) variable lower bounds
        self.U = torch.full((self._ydim,), 100.0)                 # (ydim,) variable upper bounds
        self.input_L = self.trainX.min(dim=0).values - 0.1        # (xdim,) input parameter lower bounds
        self.input_U = self.trainX.max(dim=0).values + 0.1        # (xdim,) input parameter upper bounds

    def __str__(self):
        return f'NonCvxProblem-{self.num}'

    def to(self, device):
        """Move all tensors to specified device and update derived tensors"""
        self._device = device
        # Move all primary tensors
        self._Q = self._Q.to(device)
        self._p = self._p.to(device)
        self._A = self._A.to(device)
        self._b = self._b.to(device)
        self._C = self._C.to(device)
        self._X = self._X.to(device)
        self._Y = self._Y.to(device)
        # Recreate derived tensors on the new device
        self._C_partial = self._C[:, self._partial_vars]
        self._C_other_inv = torch.inverse(self._C[:, self._other_vars])
        self.L = self.L.to(device)
        self.U = self.U.to(device)
        self.input_L = self.input_L.to(device)
        self.input_U = self.input_U.to(device)
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
    def b(self):
        return self._b

    @property
    def C(self):
        return self._C

    @property
    def X(self):
        return self._X

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
    def b_np(self):
        return self.b.detach().cpu().numpy()

    @property
    def C_np(self):
        return self.C.detach().cpu().numpy()

    @property
    def X_np(self):
        return self.X.detach().cpu().numpy()

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
    def trainX(self):
        return self.X[:int(self.num*self.train_frac)]

    @property
    def validX(self):
        return self.X[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

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
    def trainOptvals(self):
        return self.opt_vals[self.train_indices]

    @property
    def validOptvals(self):
        return self.opt_vals[self.valid_indices]

    @property
    def testOptvals(self):
        return self.opt_vals[self.test_indices]

    @property
    def testX(self):
        return self.X[int(self.num*(self.train_frac + self.valid_frac)):]

    @property
    def trainY(self):
        return self.Y[:int(self.num*self.train_frac)]

    @property
    def validY(self):
        return self.Y[int(self.num*self.train_frac):int(self.num*(self.train_frac + self.valid_frac))]

    @property
    def testY(self):
        return self.Y[int(self.num*(self.train_frac + self.valid_frac)):]

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
        return (0.5*(Y@self.Q)*Y + self.p*torch.sin(Y)).sum(dim=1)

    def get_lower_bound(self, X):
        """lower bound for inequality constraints bl(x)<=A_eff(x)y<=bu(x)"""
        bl_ineq = torch.full((X.shape[0], self.nineq), float('-inf'), device=self.device)
        return torch.cat((bl_ineq, X), dim=1)
    
    def get_upper_bound(self, X):
        """upper bound for inequality constraints bl(x)<=A_eff(x)y<=bu(x)"""
        bu_ineq = self.b.repeat(X.shape[0],1)
        return torch.cat((bu_ineq, X), dim=1)
    
    def get_resid(self, X, Y):
        ineq_reisd = torch.clamp(Y@self.A.T - self.b, 0)
        eq_resid = torch.abs(Y@self.C.T - X)
        return torch.cat((ineq_reisd, eq_resid), dim=1)
    
    def get_train_loss(self, net, X, Ytarget, cfg):
        Y = net(X)
        main_loss = self.evaluate(X, Y)
        regularization = torch.norm(self.get_resid(X, Y), dim=1)**2
        return main_loss + cfg.soft_weight * regularization
    
    def get_eval_metric(self, net, X, Y, Ytarget):
        return self.evaluate(X, Y)
    
    def get_err_metric1(self, net, X, Y, Ytarget):
        """compute ineq error"""
        return torch.clamp(Y@self.A.T - self.b, 0)
    
    def get_err_metric2(self, net, X, Y, Ytarget):
        """compute eq error"""
        return torch.abs(Y@self.C.T - X)

    def get_lower_upper_bounds(self, X):
        return self.get_lower_bound(X), self.get_upper_bound(X)

    def get_g(self, X):
        return lambda x: (torch.cat((self.A, self.C), dim=0) @ x[:, :, None]).squeeze(-1)

    def get_jacobian(self, X):
        return lambda x: torch.cat((self.A, self.C), dim=0).unsqueeze(0).expand(x.shape[0], -1, -1)

    def get_coefficients(self, X):
        """coefficients for inequality constraints bl(x)<=A_eff(x)y<=bu(x)"""
        A_eff = torch.cat((self.A, self.C), dim=0)
        # A_eff = A_eff.repeat(X.shape[0],1,1) commented as we deal with input-independent A_eff separately
        bl = self.get_lower_bound(X)
        bu = self.get_upper_bound(X)
        return A_eff, bl, bu


    ########## For DC3 only ##########

    def get_resid_grad(self, X, Y):
        """gradient of ||resid||^2"""
        ineq_grad = 2*torch.clamp(Y@self.A.T - self.b, 0)@self.A
        eq_grad = 2*(Y@self.C.T - X)@self.C
        return ineq_grad + eq_grad
        # grad_list = []
        # for n in range(Y.shape[0]):
        #     x = X[n].view(1, -1)
        #     y = Y[n].view(1, -1)
        #     y = torch.autograd.Variable(y, requires_grad=True)
        #     resid_penalty = self.get_resid(x, y) ** 2
        #     resid_penalty = torch.sum(resid_penalty, dim=-1, keepdim=True)
        #     grad = torch.autograd.grad(resid_penalty, y)[0]
        #     grad_list.append(grad.view(1, -1))
        # grad = torch.cat(grad_list, dim=0)
        # return grad
    
    def get_ineq_partial_grad(self, X, Y):
        """gradient for DC3 with equality completion"""
        # coefficients for reduced inequality constraint A_red(x)y_partial<=b_red(x)
        A_red = self.A[:, self.partial_vars] - self.A[:, self.other_vars] @ (self._C_other_inv @ self._C_partial)
        bu_red = self.b - (X @ self._C_other_inv.T) @ self.A[:, self.other_vars].T
        grad_partial = 2 * torch.clamp(Y[:, self.partial_vars] @ A_red.T - bu_red, 0) @ A_red
        grad = torch.zeros(X.shape[0], self.ydim, device=X.device)
        grad[:, self.partial_vars] = grad_partial
        grad[:, self.other_vars] = - (grad_partial @ self._C_partial.T) @ self._C_other_inv.T
        return grad
        # grad_list = []
        # for n in range(Y.shape[0]):
        #     Y_pred = Y[n, self.partial_vars].view(1, -1)
        #     x = X[n].view(1, -1)
        #     Y_pred = torch.autograd.Variable(Y_pred, requires_grad=True)
        #     y = self.complete_partial(x, Y_pred)
        #     ineq_resid = torch.clamp(y@self.A.T - self.b, 0)
        #     ineq_penalty = ineq_resid ** 2
        #     ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
        #     grad_pred = torch.autograd.grad(ineq_penalty, Y_pred)[0]
        #     grad = torch.zeros(1, self.ydim, device=X.device)
        #     grad[0, self.partial_vars] = grad_pred
        #     grad[0, self.other_vars] = - (grad_pred @ self._C_partial.T) @ self._C_other_inv.T
        #     grad_list.append(grad)
        # return torch.cat(grad_list, dim=0)

    def complete_partial(self, X, Z):
        """solve for other variables given partial variables Z"""
        Y = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y[:, self.partial_vars] = Z
        Y[:, self.other_vars] = (X - Z @ self._C_partial.T) @ self._C_other_inv.T
        return Y
    
    def get_cvxpy_projection_layer(self):
        """
        Create a differentiable cvxpy layer for projecting onto the constraint set.
        
        Solves: minimize ||y - y_hat||^2
                subject to Ay <= b, Cy = x
        
        Returns a CvxpyLayer that can be used in PyTorch models.
        """
        # Define cvxpy variables
        y = cp.Variable(self.ydim)
        y_hat = cp.Parameter(self.ydim)  # Neural network output
        x = cp.Parameter(self.neq)  # Input parameter
        
        # Convert to numpy for cvxpy
        A_np = self.A_np
        b_np = self.b_np
        C_np = self.C_np
        
        # Define constraints
        constraints = [
            A_np @ y <= b_np,  # Inequality constraints
            C_np @ y == x      # Equality constraints
        ]
        
        # Define objective: minimize ||y - y_hat||^2
        objective = cp.Minimize(cp.sum_squares(y - y_hat))
        
        # Create problem
        problem = cp.Problem(objective, constraints)
        
        # Create differentiable layer
        cvxpy_layer = CvxpyLayer(problem, parameters=[y_hat, x], variables=[y])
        
        return cvxpy_layer
    ####################################

    
    ###### For optimization solver

    def opt_solve(self, indices=None, solver_type='scipy', tol=1e-4, verbose=False):
        Q, p, A, b, C = self.Q_np, self.p_np, self.A_np, self.b_np, self.C_np

        if indices is None:
            indices = np.arange(self.num)
        
        X_np = self.X_np[indices]
        total_time = 0
        
        for idx, Xi in enumerate(tqdm(X_np, desc=f"Solving {solver_type}")):
            if solver_type == 'scipy':
                # Use scipy's SLSQP for local optimization (much faster but only finds local optimum)
                # Define objective function
                def objective(y_vals):
                    quad_term = 0.5 * np.dot(y_vals * Q.diagonal(), y_vals)
                    sin_term = np.dot(p, np.sin(y_vals))
                    return quad_term + sin_term
                
                # Define gradient
                def gradient(y_vals):
                    grad_quad = Q.diagonal() * y_vals
                    grad_sin = p * np.cos(y_vals)
                    return grad_quad + grad_sin
                
                # Define constraints
                constraints = []
                # Equality constraints: Cy = x
                for i in range(C.shape[0]):
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda y_vals, i=i: np.dot(C[i], y_vals) - Xi[i],
                        'jac': lambda y_vals, i=i: C[i]
                    })
                
                # Inequality constraints: Ay <= b
                for i in range(A.shape[0]):
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda y_vals, i=i: b[i] - np.dot(A[i], y_vals),
                        'jac': lambda y_vals, i=i: -A[i]
                    })
                
                # Initial guess: try to satisfy equality constraints
                y0 = np.linalg.lstsq(C, Xi, rcond=None)[0]
                
                # Bounds for variables
                bounds = [(-100, 100)] * self._ydim
                
                start_time = time.time()
                result = minimize(
                    objective,
                    y0,
                    method='SLSQP',
                    jac=gradient,
                    constraints=constraints,
                    bounds=bounds,
                    options={'ftol': tol, 'disp': verbose, 'maxiter': 1000}
                )
                end_time = time.time()
                
                if result.success or result.status == 9:  # 9 means iteration limit reached with feasible solution
                    self._Y[indices[idx]] = torch.tensor(result.x, device=self.device)
                    self.opt_vals[indices[idx]] = result.fun
                else:
                    print(f"Warning: Scipy optimization failed for index {indices[idx]}. Status: {result.status}, Message: {result.message}")
                    # Keep NaN values
                
                total_time += (end_time - start_time)
            
            else:
                raise NotImplementedError(f"Solver type {solver_type} not implemented.")

        return total_time, total_time/len(X_np)