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

    def __init__(self, budget, kind, hybrid=True, n_active=1, use_torch=True,
                 reg_eps=1e-8, init_r_upper=20.0):
        self.budget = budget
        self.fac = int(hybrid)
        self.n_active = n_active
        self.use_torch = use_torch
        self.corr_matrix_fn = None

        # Tikhonov regularization added to F*C before solving for R^2. Guards
        # against the near-singular system that arises as any r_i -> 1 (see
        # _solve_R2 for why).
        self.reg_eps = reg_eps

        # Cap used for open-ended oversampling-ratio bounds when drawing
        # random initial guesses (see _initial_guess). Optimal r's are
        # rarely anywhere near the budget scale, so falling back to
        # self.budget there wastes most random restarts on implausible
        # points.
        self.init_r_upper = init_r_upper

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
        """
        Create a feasibility-aware random initial point.

        Two changes relative to drawing every entry independently and
        uniformly within self.bounds:

        1. Open-ended oversampling ratios r are capped at self.init_r_upper
           rather than at self.budget. Optimal r's are almost never near the
           budget scale, so the old fallback (e.g. r in [1.001, 240] for a
           budget-240 restart) wasted most random starts on implausible
           points far from any reasonable optimum.
        2. N is back-solved from the budget constraint given the drawn r
           and s, instead of drawn independently. Each restart now starts
           just inside the feasible region rather than at an arbitrary
           point SLSQP has to fight its way back from.
        """
        n_lofi = len(self.cost_list)
        dim = 1 + n_lofi + self.n_active

        x0 = np.zeros(dim, dtype=float)

        # Draw r (indices 1..n_lofi) and s_active (remaining indices) first.
        for i in range(1, dim):
            lo, hi = self.bounds[i]
            if lo is None:
                lo = 1.001
            if hi is None:
                hi = self.init_r_upper

            x0[i] = lo if hi <= lo else np.random.uniform(lo, hi)

        r = x0[1:n_lofi + 1]
        s_active = x0[n_lofi + 1:]

        # Back out N from the budget constraint N * unit_cost <= budget.
        # Cost surrogates may hold torch parameters internally and can
        # return a grad-tracking tensor even when handed plain numpy s
        # (e.g. a fitted polynomial with torch coefficients), so extract
        # each value through the class's existing tensor/float-safe helper
        # rather than assuming a plain float.
        s = self.expand_s(s_active)
        w = np.array([self._to_numpy_scalar(f(s)) for f in self.cost_list], dtype=float)
        unit_cost = 1.0 + np.dot(r, w) + self.fac * np.sum(s_active)

        N_lo, N_hi = self.bounds[0]
        N_lo = 1.0 if N_lo is None else N_lo

        N_feasible = self.budget / max(unit_cost, 1e-8)
        if N_hi is not None:
            N_feasible = min(N_feasible, N_hi)

        # Land somewhere inside the feasible strip rather than right on its
        # boundary, so SLSQP has room to move in either direction.
        frac = np.random.uniform(0.5, 1.0)
        x0[0] = max(N_lo, N_feasible * frac)

        return x0

    @staticmethod
    def _to_numpy_scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().numpy())
        return float(value)

    @staticmethod
    def _tril_indices(n):
        """Strict lower-triangular index pairs, shared by the numpy and
        torch branches of build_C (backend-independent index math)."""
        return np.tril_indices(n, -1)

    @staticmethod
    def _split_corr_matrix(P):
        """
        Split a full correlation matrix P, ordered [FOM, LF_1, ..., LF_m],
        into the HF-LF correlation vector c and the LF-LF correlation
        matrix C. Shared by the corr_matrix_fn branch of
        _torch_correlations and _numpy_correlations; identical shape logic
        in both, only how P itself is obtained differs by backend.
        """
        c = P[1:, 0]
        C = P[1:, 1:]
        return c, C

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

            idx_i, idx_j = self._tril_indices(n)
            idx_i_t = torch.tensor(idx_i, dtype=torch.long, device=entries.device)
            idx_j_t = torch.tensor(idx_j, dtype=torch.long, device=entries.device)

            C = torch.eye(n, dtype=entries.dtype, device=entries.device)
            C = C.index_put((idx_i_t, idx_j_t), entries, accumulate=False)
            C = C.index_put((idx_j_t, idx_i_t), entries, accumulate=False)

            return C

        entries = np.array([f(s) for f in self.lf_corr_list])
        n = floor((1 + np.sqrt(1 + 8 * len(entries))) / 2)

        C = np.eye(n)
        idx = self._tril_indices(n)
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
            return self._split_corr_matrix(P)

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
            return self._split_corr_matrix(P)

        s = self.expand_s(s_active)
        c = np.array([f(s) for f in self.hf_corr_list], dtype=float)
        C = self.build_C(s_active)

        return c, C

    def _solve_R2(self, F, C, c):
        """
        Solve v^T (F*C)^{-1} v for v = diag(F) * c, with a small Tikhonov
        term added to F*C.

        As any oversampling ratio r_i approaches its lower bound of 1,
        diag(F)_i -> 0, which collapses row/column i of F*C toward zero.
        R^2 itself has a well-defined limit there, but the matrix being
        inverted becomes severely ill-conditioned right at that boundary,
        and autograd's backward pass through the solve reuses (and can
        amplify) that same ill-conditioning. The regularization keeps both
        the forward solve and its gradient numerically stable without
        materially perturbing R^2 away from the boundary.
        """
        n = F.shape[0]

        if self.use_torch and torch.is_tensor(F):
            M = F * C + self.reg_eps * torch.eye(n, dtype=F.dtype, device=F.device)
            v = torch.diag(F) * c
            return torch.linalg.solve(M, v).dot(v)

        M = F * C + self.reg_eps * np.eye(n)
        v = np.diag(F) * c
        return np.linalg.solve(M, v).dot(v)

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

                R2 = self._solve_R2(F, C, c)

                val = (1.0 - R2) / N
                result = torch.log(val) if self.log else val

                return float(result.detach().cpu().numpy())

            N = x[0]
            r = x[1:n_lofi + 1]
            s_active = x[n_lofi + 1:]

            F = self.build_F(r)
            c, C = self._numpy_correlations(s_active)

            R2 = self._solve_R2(F, C, c)

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

            R2 = self._solve_R2(F, C, c)

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