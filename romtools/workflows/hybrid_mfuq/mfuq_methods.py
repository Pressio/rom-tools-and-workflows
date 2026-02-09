from math import floor
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


class MFMC:
    """
    Multifidelity Monte Carlo with approximate control variates.
    Optimization is performed using correlations (not covariances).
    """

    def __init__(self, budget, kind, hybrid=True, n_active=1):
        self.budget = budget
        self.fac = int(hybrid)
        self.n_active = n_active

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

        s = np.zeros(n)
        s[-self.n_active:] = s_active
        return s

    # ------------------------------------------------------------------
    # Matrices
    # ------------------------------------------------------------------

    def build_F(self, r):
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

        def constraint(x):
            N = x[0]
            r = x[1:n_lofi + 1]
            s_active = x[n_lofi + 1:]

            s = self.expand_s(s_active)
            w = np.array([f(s) for f in self.cost_list])

            return N * (1 + np.dot(r, w) + self.fac * np.sum(s_active))

        self.objective = objective
        self.constraint = constraint

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self):
        n_lofi = len(self.cost_list)
        dim = 1 + n_lofi + self.n_active

        x0 = np.random.uniform(1, self.bounds[-1][1], dim)
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
