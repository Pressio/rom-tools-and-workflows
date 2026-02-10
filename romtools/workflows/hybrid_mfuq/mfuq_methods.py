from math import floor
import numpy as np
import torch
from scipy.optimize import minimize, NonlinearConstraint


class MFMC:
    """
    Multifidelity Monte Carlo with approximate control variates.
    Optimization is performed using correlations (not covariances).
    
    PyTorch-compatible version that uses autograd for gradients.
    """

    def __init__(self, budget, kind, hybrid=True, n_active=1, use_torch=True):
        self.budget = budget
        self.fac = int(hybrid)
        self.n_active = n_active
        self.use_torch = use_torch

        if kind in ("MF", "ACV-MF"):
            self.type = "ACV-MF"
        elif kind in ("IS", "ACV-IS"):
            self.type = "ACV-IS"
        else:
            raise ValueError(f"Unknown MFMC type '{kind}'")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_corrs_and_costs(self, hf_corr_list, lf_corr_list, cost_list):
        self.hf_corr_list = hf_corr_list
        self.lf_corr_list = lf_corr_list
        self.cost_list = cost_list

        if self.n_active is None:
            self.n_active = len(hf_corr_list)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def expand_s(self, s_active):
        """Pad active s entries with leading zeros."""
        n = len(self.hf_corr_list)
        if self.n_active == n:
            return s_active

        if self.use_torch and torch.is_tensor(s_active):
            s = torch.zeros(n, dtype=s_active.dtype, device=s_active.device)
            s[-self.n_active:] = s_active
        else:
            s = np.zeros(n)
            s[-self.n_active:] = s_active
        return s

    # ------------------------------------------------------------------
    # Matrices
    # ------------------------------------------------------------------

    def build_F(self, r):
        if self.use_torch and torch.is_tensor(r):
            if self.type == "ACV-MF":
                xr, yr = torch.meshgrid(r, r, indexing='ij')
                m = torch.minimum(xr, yr)
                return (m - 1) / m

            # ACV-IS
            v = (r - 1) / r
            F_outer = torch.outer(v, v)
            # Replace diagonal without inplace operation
            F = F_outer + torch.diag(v - F_outer.diagonal())
            return F
        else:
            if self.type == "ACV-MF":
                xr, yr = np.meshgrid(r, r)
                m = np.minimum(xr, yr)
                return (m - 1) / m

            # ACV-IS
            v = (r - 1) / r
            F = np.outer(v, v)
            np.fill_diagonal(F, v)
            return F

    def build_C(self, s_active):
        s = self.expand_s(s_active)

        if self.use_torch and torch.is_tensor(s):
            entries = torch.stack([f(s) for f in self.lf_corr_list])
            n = floor((1 + np.sqrt(1 + 8 * len(entries))) / 2)

            # Get lower triangle indices
            idx_i, idx_j = np.tril_indices(n, -1)
            idx_i_t = torch.tensor(idx_i, dtype=torch.long, device=entries.device)
            idx_j_t = torch.tensor(idx_j, dtype=torch.long, device=entries.device)
            
            # Start with identity
            C = torch.eye(n, dtype=entries.dtype, device=entries.device)
            
            # Use index_put (non-inplace version) to set lower triangle
            C = C.index_put((idx_i_t, idx_j_t), entries, accumulate=False)
            # Set upper triangle (transpose)
            C = C.index_put((idx_j_t, idx_i_t), entries, accumulate=False)
            
            return C
        else:
            entries = np.array([f(s) for f in self.lf_corr_list])
            n = floor((1 + np.sqrt(1 + 8 * len(entries))) / 2)

            C = np.eye(n)
            idx = np.tril_indices(n, -1)
            C[idx] = entries
            C[(idx[1], idx[0])] = entries
            return C

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def set_objective_and_constraint(self, log=True, bounds=None):
        self.log = log

        n_lofi = len(self.hf_corr_list)
        dim = 1 + n_lofi + self.n_active

        self.bounds = bounds if bounds is not None else [(None, None)] * dim

        def objective(x):
            if self.use_torch:
                x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                
                N = x_torch[0]
                r = x_torch[1:n_lofi + 1]
                s_active = x_torch[n_lofi + 1:]

                F = self.build_F(r)
                C = self.build_C(s_active)

                s = self.expand_s(s_active)
                c = torch.stack([f(s) for f in self.hf_corr_list])

                v = torch.diag(F) * c
                R2 = torch.linalg.solve(F * C, v).dot(v)

                val = (1 - R2) / N
                result = torch.log(val) if self.log else val
                
                return float(result.detach().numpy())
            else:
                N = x[0]
                r = x[1:n_lofi + 1]
                s_active = x[n_lofi + 1:]

                F = self.build_F(r)
                C = self.build_C(s_active)

                s = self.expand_s(s_active)
                c = np.array([f(s) for f in self.hf_corr_list])

                v = np.diag(F) * c
                R2 = np.linalg.solve(F * C, v).dot(v)

                val = (1 - R2) / N
                return np.log(val) if self.log else val

        def objective_grad(x):
            if self.use_torch:
                x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                
                N = x_torch[0]
                r = x_torch[1:n_lofi + 1]
                s_active = x_torch[n_lofi + 1:]

                F = self.build_F(r)
                C = self.build_C(s_active)

                s = self.expand_s(s_active)
                c = torch.stack([f(s) for f in self.hf_corr_list])

                v = torch.diag(F) * c
                R2 = torch.linalg.solve(F * C, v).dot(v)

                val = (1 - R2) / N
                result = torch.log(val) if self.log else val
                
                result.backward()
                return x_torch.grad.detach().numpy().astype(np.float64)
            else:
                # Use numerical gradients if not using torch
                return None

        def constraint(x):
            if self.use_torch:
                x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                
                N = x_torch[0]
                r = x_torch[1:n_lofi + 1]
                s_active = x_torch[n_lofi + 1:]

                s = self.expand_s(s_active)
                w = torch.stack([f(s) for f in self.cost_list])

                result = N * (1 + torch.dot(r, w) + self.fac * torch.sum(s_active))
                return float(result.detach().numpy())
            else:
                N = x[0]
                r = x[1:n_lofi + 1]
                s_active = x[n_lofi + 1:]

                s = self.expand_s(s_active)
                w = np.array([f(s) for f in self.cost_list])

                return N * (1 + np.dot(r, w) + self.fac * np.sum(s_active))

        def constraint_grad(x):
            if self.use_torch:
                x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
                
                N = x_torch[0]
                r = x_torch[1:n_lofi + 1]
                s_active = x_torch[n_lofi + 1:]

                s = self.expand_s(s_active)
                w = torch.stack([f(s) for f in self.cost_list])

                result = N * (1 + torch.dot(r, w) + self.fac * torch.sum(s_active))
                
                result.backward()
                return x_torch.grad.detach().numpy().astype(np.float64)
            else:
                # Use numerical gradients if not using torch
                return None

        self.objective = objective
        self.objective_grad = objective_grad
        self.constraint = constraint
        self.constraint_grad = constraint_grad

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self):
        n_lofi = len(self.cost_list)
        dim = 1 + n_lofi + self.n_active

        x0 = np.random.uniform(1, self.bounds[-1][1], dim)
        
        if self.use_torch:
            # Use analytical gradients from autograd
            nlc = NonlinearConstraint(
                self.constraint, 
                -np.inf, 
                self.budget,
                jac=self.constraint_grad
            )
            
            self.result = minimize(
                self.objective,
                x0,
                method="slsqp",
                jac=self.objective_grad,
                bounds=self.bounds,
                constraints=nlc,
                options={"maxiter": 500},
            )
        else:
            # Use finite difference gradients
            nlc = NonlinearConstraint(self.constraint, -np.inf, self.budget)
            
            self.result = minimize(
                self.objective,
                x0,
                method="slsqp",
                jac="2-point",
                bounds=self.bounds,
                constraints=nlc,
                options={"maxiter": 500},
            )

    def check_gradients(self, x=None, eps=1e-6, tol=1e-5):
        """
        Compare PyTorch autograd gradients with finite-difference numerical gradients.

        Parameters:
            x : initial point (optional). Random if None.
            eps : small step for finite differences
            tol : tolerance for gradient comparison
        """
        if not self.use_torch:
            raise RuntimeError("Gradient check only available with use_torch=True")

        if x is None:
            n_lofi = len(self.hf_corr_list)
            dim = 1 + n_lofi + self.n_active
            x = np.random.uniform(1, 2, dim)  # safe random starting point

        # Compute torch gradients
        grad_torch = self.objective_grad(x)

        # Compute numerical gradients (finite differences)
        grad_num = np.zeros_like(x)
        fx = self.objective(x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            fx_eps = self.objective(x_eps)
            grad_num[i] = (fx_eps - fx) / eps

        # Compare
        diff = np.linalg.norm(grad_torch - grad_num)
        rel_diff = diff / (np.linalg.norm(grad_torch) + np.linalg.norm(grad_num) + 1e-12)
        
        print("Gradient check:")
        print("Torch grad:", grad_torch)
        print("Numerical grad:", grad_num)
        print("Absolute difference:", diff)
        print("Relative difference:", rel_diff)

        if rel_diff < tol:
            print("✅ Gradients match within tolerance.")
        else:
            print("⚠️ Gradients differ! Check your implementation.")

        return grad_torch, grad_num, rel_diff
