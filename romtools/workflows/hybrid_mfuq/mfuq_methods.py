from math import floor
import numpy as np
import torch
from scipy.optimize import minimize, NonlinearConstraint


class MFMC:
    """
    Multifidelity Monte Carlo with approximate control variates.

    Supports two correlation APIs:

    1. Scalar/componentwise API:
       - hf_corr_list: functions returning Corr[Q0, Qi]
       - lf_corr_list: functions returning lower-triangular entries of Corr[Qi, Qj]

    2. Matrix API:
       - corr_matrix_fn(s_active) returns the full correlation matrix P of size
         (m+1, m+1), ordered as [FOM, LF_1, ..., LF_m].

    The matrix API is intended for globally structured correlation surrogates,
    e.g. Archakov--Hansen.
    """

    def __init__(self, budget, kind, hybrid=True, n_active=1, use_torch=True):
        self.budget = budget
        self.fac = int(hybrid)
        self.n_active = n_active
        self.use_torch = use_torch
        self.corr_matrix_fn = None

        if kind in ("MF", "ACV-MF"):
            self.type = "ACV-MF"
        elif kind in ("IS", "ACV-IS"):
            self.type = "ACV-IS"
        else:
            raise ValueError(f"Unknown MFMC type '{kind}'")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_corrs_and_costs(
        self,
        hf_corr_list,
        lf_corr_list,
        cost_list,
        corr_matrix_fn=None,
    ):
        self.hf_corr_list = hf_corr_list
        self.lf_corr_list = lf_corr_list
        self.cost_list = cost_list
        self.corr_matrix_fn = corr_matrix_fn

        if self.n_active is None:
            self.n_active = 1

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def expand_s(self, s_active):
        """Pad active s entries with leading zeros."""
        n = len(self.cost_list)

        if self.n_active == n:
            return s_active

        if self.use_torch and torch.is_tensor(s_active):
            s = torch.zeros(n, dtype=s_active.dtype, device=s_active.device)
            s[-self.n_active:] = s_active
        else:
            s = np.zeros(n)
            s[-self.n_active:] = s_active

        return s

    def _initial_guess(self):
        """Create a feasible-ish random initial point inside simple bounds."""
        n_lofi = len(self.cost_list)
        dim = 1 + n_lofi + self.n_active

        x0 = np.zeros(dim, dtype=float)

        for i, (lo, hi) in enumerate(self.bounds):
            if lo is None:
                lo = 1.0
            if hi is None:
                # Reasonable finite upper guess for unbounded variables.
                hi = max(2.0, self.budget)

            if hi == lo:
                x0[i] = lo
            else:
                x0[i] = np.random.uniform(lo, hi)

        return x0

    @staticmethod
    def _to_numpy_scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().numpy())
        return float(value)

    # ------------------------------------------------------------------
    # Matrices
    # ------------------------------------------------------------------

    def build_F(self, r):
        if self.use_torch and torch.is_tensor(r):
            if self.type == "ACV-MF":
                xr, yr = torch.meshgrid(r, r, indexing="ij")
                m = torch.minimum(xr, yr)
                return (m - 1.0) / m

            # ACV-IS
            v = (r - 1.0) / r
            F_outer = torch.outer(v, v)
            F = F_outer + torch.diag(v - F_outer.diagonal())
            return F

        if self.type == "ACV-MF":
            xr, yr = np.meshgrid(r, r)
            m = np.minimum(xr, yr)
            return (m - 1.0) / m

        # ACV-IS
        v = (r - 1.0) / r
        F = np.outer(v, v)
        np.fill_diagonal(F, v)
        return F

    def build_C(self, s_active):
        """Build LF-LF correlation matrix from scalar lower-triangular entries."""
        s = self.expand_s(s_active)

        if self.use_torch and torch.is_tensor(s):
            entries = torch.stack([f(s) for f in self.lf_corr_list])
            n = floor((1 + np.sqrt(1 + 8 * len(entries))) / 2)

            idx_i, idx_j = np.tril_indices(n, -1)
            idx_i_t = torch.tensor(idx_i, dtype=torch.long, device=entries.device)
            idx_j_t = torch.tensor(idx_j, dtype=torch.long, device=entries.device)

            C = torch.eye(n, dtype=entries.dtype, device=entries.device)
            C = C.index_put((idx_i_t, idx_j_t), entries, accumulate=False)
            C = C.index_put((idx_j_t, idx_i_t), entries, accumulate=False)

            return C

        entries = np.array([f(s) for f in self.lf_corr_list])
        n = floor((1 + np.sqrt(1 + 8 * len(entries))) / 2)

        C = np.eye(n)
        idx = np.tril_indices(n, -1)
        C[idx] = entries
        C[(idx[1], idx[0])] = entries

        return C

    def _torch_correlations(self, s_active):
        """
        Return c and C in torch mode.

        c[i] = Corr[Q0, Qi]
        C[i,j] = Corr[Qi, Qj]
        """
        if self.corr_matrix_fn is not None:
            P = self.corr_matrix_fn(s_active)
            if not torch.is_tensor(P):
                P = torch.as_tensor(P, dtype=torch.float64)

            P = P.to(dtype=s_active.dtype, device=s_active.device)
            c = P[1:, 0]
            C = P[1:, 1:]
            return c, C

        s = self.expand_s(s_active)
        c = torch.stack([f(s) for f in self.hf_corr_list])
        C = self.build_C(s_active)

        return c, C

    def _numpy_correlations(self, s_active):
        if self.corr_matrix_fn is not None:
            P = self.corr_matrix_fn(s_active)
            if torch.is_tensor(P):
                P = P.detach().cpu().numpy()
            P = np.asarray(P, dtype=float)
            c = P[1:, 0]
            C = P[1:, 1:]
            return c, C

        s = self.expand_s(s_active)
        c = np.array([f(s) for f in self.hf_corr_list], dtype=float)
        C = self.build_C(s_active)

        return c, C

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def set_objective_and_constraint(self, log=True, bounds=None):
        self.log = log

        n_lofi = len(self.cost_list)
        dim = 1 + n_lofi + self.n_active

        self.bounds = bounds if bounds is not None else [(None, None)] * dim

        def objective(x):
            if self.use_torch:
                x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)

                N = x_torch[0]
                r = x_torch[1:n_lofi + 1]
                s_active = x_torch[n_lofi + 1:]

                F = self.build_F(r)
                c, C = self._torch_correlations(s_active)

                v = torch.diag(F) * c
                R2 = torch.linalg.solve(F * C, v).dot(v)

                val = (1.0 - R2) / N
                result = torch.log(val) if self.log else val

                return float(result.detach().cpu().numpy())

            N = x[0]
            r = x[1:n_lofi + 1]
            s_active = x[n_lofi + 1:]

            F = self.build_F(r)
            c, C = self._numpy_correlations(s_active)

            v = np.diag(F) * c
            R2 = np.linalg.solve(F * C, v).dot(v)

            val = (1.0 - R2) / N
            return np.log(val) if self.log else val

        def objective_grad(x):
            if not self.use_torch:
                return None

            x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)

            N = x_torch[0]
            r = x_torch[1:n_lofi + 1]
            s_active = x_torch[n_lofi + 1:]

            F = self.build_F(r)
            c, C = self._torch_correlations(s_active)

            v = torch.diag(F) * c
            R2 = torch.linalg.solve(F * C, v).dot(v)

            val = (1.0 - R2) / N
            result = torch.log(val) if self.log else val

            result.backward()

            return x_torch.grad.detach().cpu().numpy().astype(np.float64)

        def constraint(x):
            if self.use_torch:
                x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)

                N = x_torch[0]
                r = x_torch[1:n_lofi + 1]
                s_active = x_torch[n_lofi + 1:]

                s = self.expand_s(s_active)
                w = torch.stack([f(s) for f in self.cost_list])

                result = N * (
                    1.0 + torch.dot(r, w) + self.fac * torch.sum(s_active)
                )

                return float(result.detach().cpu().numpy())

            N = x[0]
            r = x[1:n_lofi + 1]
            s_active = x[n_lofi + 1:]

            s = self.expand_s(s_active)
            w = np.array([f(s) for f in self.cost_list], dtype=float)

            return N * (1.0 + np.dot(r, w) + self.fac * np.sum(s_active))

        def constraint_grad(x):
            if not self.use_torch:
                return None

            x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)

            N = x_torch[0]
            r = x_torch[1:n_lofi + 1]
            s_active = x_torch[n_lofi + 1:]

            s = self.expand_s(s_active)
            w = torch.stack([f(s) for f in self.cost_list])

            result = N * (
                1.0 + torch.dot(r, w) + self.fac * torch.sum(s_active)
            )

            result.backward()

            return x_torch.grad.detach().cpu().numpy().astype(np.float64)

        self.objective = objective
        self.objective_grad = objective_grad
        self.constraint = constraint
        self.constraint_grad = constraint_grad

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self):
        x0 = self._initial_guess()

        if self.use_torch:
            nlc = NonlinearConstraint(
                self.constraint,
                -np.inf,
                self.budget,
                jac=self.constraint_grad,
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
        Compare PyTorch autograd gradients with finite-difference gradients.
        """
        if not self.use_torch:
            raise RuntimeError("Gradient check only available with use_torch=True")

        if x is None:
            n_lofi = len(self.cost_list)
            dim = 1 + n_lofi + self.n_active
            x = self._initial_guess()

        grad_torch = self.objective_grad(x)

        grad_num = np.zeros_like(x)
        fx = self.objective(x)

        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            fx_eps = self.objective(x_eps)
            grad_num[i] = (fx_eps - fx) / eps

        diff = np.linalg.norm(grad_torch - grad_num)
        rel_diff = diff / (
            np.linalg.norm(grad_torch) + np.linalg.norm(grad_num) + 1e-12
        )

        print("Gradient check:")
        print("Torch grad:", grad_torch)
        print("Numerical grad:", grad_num)
        print("Absolute difference:", diff)
        print("Relative difference:", rel_diff)

        if rel_diff < tol:
            print("✅ Gradients match within tolerance.")
        else:
            print("⚠️ Gradients differ. For AH this may reflect the detached fixed-point solve.")

        return grad_torch, grad_num, rel_diff