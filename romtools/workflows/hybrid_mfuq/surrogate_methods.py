"""
Surrogate Methods for Multi-Fidelity UQ

Builds surrogate models for correlation and cost functions using:
- Neural network approach (VeclNet) for valid correlation matrices
- Sigmoid fitting for smooth monotonic functions
- Polynomial fitting for simple trends

All surrogates are PyTorch-differentiable for backpropagation through s.
"""

import os
from functools import lru_cache
from typing import List, Tuple, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.optimize import least_squares

import torch
import torch.nn as nn
from torch.optim import LBFGS


def fit_polynomial(ins, outs, order=5):
    '''
    Fit a tensor product of 1-D polynomials to data.
    ins: [dim, n_data] or [dim] — input variables
    outs: [n_data] or scalar — target values
    '''
    ins = np.atleast_2d(ins)
    if ins.shape[0] > ins.shape[1]:
        ins = ins.T
    dim, n_data = ins.shape

    def evaluate_tensor_product(coeffs, x):
        coeffs = coeffs.reshape(dim, order + 1)
        x = np.atleast_2d(x)
        if x.shape[0] != dim:
            x = x.T
        single_input = x.shape[1] == 1 and x.ndim == 2 and x.shape[0] == dim

        result = np.ones(x.shape[1])
        for d in range(dim):
            powers = np.vander(x[d], N=order+1, increasing=True)
            poly_vals = powers @ coeffs[d]
            result *= poly_vals

        return result[0] if single_input else result

    def residuals(coeffs):
        return evaluate_tensor_product(coeffs, ins) - outs

    x0 = np.full((dim * (order + 1),), 0.5)
    result = least_squares(residuals, x0, loss='linear')

    return lambda x: evaluate_tensor_product(result.x, x)


def fit_sigmoid(ins, outs, character=[]):
    '''
    Fit a tensor product of 3- or 5-parameter sigmoids to data.
    
    ins: shape [dim, n_data] or [dim]
    outs: shape [n_data] or scalar
    character: 'increasing', 'decreasing', or [] (for general sigmoid)
    '''
    ins = np.atleast_2d(ins)
    if ins.shape[0] != 1:
        ins = ins.T
    dim, n_data = ins.shape

    if character == 'increasing':
        opt = True
        num_vars = 4
    elif character == 'decreasing':
        opt = False
        num_vars = 4
    elif character == []:
        opt = None
        num_vars = 5
    else:
        raise ValueError('Invalid character. Options are "increasing", "decreasing", or [].')

    def sigmoid(params, x):
        if num_vars == 4:
            A, B, log_nu, log_Q = params
            nu = np.exp(log_nu)
            Q = np.exp(log_Q)
            K = int(not opt)
            return A + (K - A) / (1 + Q * np.exp((x-B)))**(1 / nu)
        else:
            A, K, B, nu, Q = params
            return A + (K - A) / (1 + Q * np.exp(-B * x))**(1 / nu)

    def evaluate_tensor_product(params, x):
        x = np.atleast_2d(x)
        if x.shape[0] != dim:
            x = x.T
        single_input = x.shape[1] == 1 and x.ndim == 2

        param_array = params.reshape(dim, num_vars)
        result = np.ones(x.shape[1])
        for d in range(dim):
            result *= sigmoid(param_array[d], x[d])
        return result[0] if single_input else result

    def residuals(params):
        return evaluate_tensor_product(params, ins) - outs

    x0 = np.full((dim * num_vars,), 0.5)
    result = least_squares(residuals, x0, loss='linear')

    return lambda x: evaluate_tensor_product(result.x, x)


def fit_polynomial_torch(ins, outs, order=5):
    """
    Fit a tensor product of 1-D polynomials to data (PyTorch version).
    Returns a callable that preserves gradients.
    
    ins: [dim, n_data] or [dim] — input variables
    outs: [n_data] or scalar — target values
    """
    ins = torch.atleast_2d(torch.as_tensor(ins, dtype=torch.float64))
    outs = torch.as_tensor(outs, dtype=torch.float64)
    
    if ins.shape[0] > ins.shape[1]:
        ins = ins.T
    dim, n_data = ins.shape
    
    # Initialize coefficients
    coeffs = nn.Parameter(torch.full((dim, order + 1), 0.5, dtype=torch.float64))
    
    def loss_fn():
        result = torch.ones(n_data, dtype=torch.float64)
        for d in range(dim):
            # Manual Vandermonde construction to avoid inplace operations
            x = ins[d]
            powers = torch.stack([x**i for i in range(order + 1)], dim=1)
            poly_vals = powers @ coeffs[d]
            result = result * poly_vals
        return torch.mean((result - outs) ** 2)
    
    # Optimize
    optimizer = LBFGS([coeffs], max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    optimizer.step(closure)
    
    # Detach optimized coefficients but keep them as parameters for gradient flow
    coeffs_opt = coeffs.detach().clone().requires_grad_(True)
    
    def evaluate(x):
        """Evaluates polynomial at x while preserving gradients."""
        # Convert to tensor while preserving gradients if already a tensor
        if torch.is_tensor(x):
            x_t = x.to(dtype=torch.float64) if x.dtype != torch.float64 else x
        else:
            x_t = torch.as_tensor(x, dtype=torch.float64)
        
        x_t = torch.atleast_2d(x_t)
        if x_t.shape[0] != dim:
            x_t = x_t.T
        
        single_input = (x_t.shape[1] == 1 and x_t.ndim == 2 and x_t.shape[0] == dim)
        
        result = torch.ones(x_t.shape[1], dtype=torch.float64, requires_grad=x_t.requires_grad)
        for d in range(dim):
            # Manual Vandermonde construction
            x_d = x_t[d]
            powers = torch.stack([x_d**i for i in range(order + 1)], dim=1)
            poly_vals = powers @ coeffs_opt[d]
            result = result * poly_vals
        
        # Always return 0-d tensor for single input (preserves gradients)
        return result.squeeze() if single_input else result
    
    return evaluate


def fit_sigmoid_torch(ins, outs, character=[]):
    """
    Fit a tensor product of 3- or 5-parameter sigmoids to data (PyTorch version).
    Returns a callable that preserves gradients.
    
    ins: shape [dim, n_data] or [dim]
    outs: shape [n_data] or scalar
    character: 'increasing', 'decreasing', or [] (for general sigmoid)
    """
    ins = torch.atleast_2d(torch.as_tensor(ins, dtype=torch.float64))
    outs = torch.as_tensor(outs, dtype=torch.float64)
    
    if ins.shape[0] != 1:
        ins = ins.T
    dim, n_data = ins.shape
    
    if character == 'increasing':
        opt = True
        num_vars = 4
    elif character == 'decreasing':
        opt = False
        num_vars = 4
    elif character == []:
        opt = None
        num_vars = 5
    else:
        raise ValueError('Invalid character. Options are "increasing", "decreasing", or [].')
    
    def sigmoid(params, x):
        if num_vars == 4:
            A, B, log_nu, log_Q = params
            nu = torch.exp(log_nu)
            Q = torch.exp(log_Q)
            K = int(not opt)
            return A + (K - A) / (1 + Q * torch.exp((x - B)))**(1 / nu)
        else:
            A, K, B, nu, Q = params
            return A + (K - A) / (1 + Q * torch.exp(-B * x))**(1 / nu)
    
    # Initialize parameters
    params = nn.Parameter(torch.full((dim, num_vars), 0.5, dtype=torch.float64))
    
    def loss_fn():
        result = torch.ones(n_data, dtype=torch.float64)
        for d in range(dim):
            result = result * sigmoid(params[d], ins[d])
        return torch.mean((result - outs) ** 2)
    
    # Optimize
    optimizer = LBFGS([params], max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    optimizer.step(closure)
    
    # Detach optimized parameters but keep them as tensors for gradient flow
    params_opt = params.detach().clone().requires_grad_(True)
    
    def evaluate(x):
        """Evaluates sigmoid at x while preserving gradients."""
        # Convert to tensor while preserving gradients if already a tensor
        if torch.is_tensor(x):
            x_t = x.to(dtype=torch.float64) if x.dtype != torch.float64 else x
        else:
            x_t = torch.as_tensor(x, dtype=torch.float64)
        
        x_t = torch.atleast_2d(x_t)
        if x_t.shape[0] != dim:
            x_t = x_t.T
        
        single_input = (x_t.shape[1] == 1 and x_t.ndim == 2)
        
        result = torch.ones(x_t.shape[1], dtype=torch.float64, requires_grad=x_t.requires_grad)
        for d in range(dim):
            result = result * sigmoid(params_opt[d], x_t[d])
        
        # Always return 0-d tensor for single input (preserves gradients)
        return result.squeeze() if single_input else result
    
    return evaluate


# ============================================================================
# BATCHED ARCHAKOV--HANSEN CORRELATION MATRIX OPERATIONS
# ============================================================================

def to_symmetric_tracefree_batch(lower_vecs, n):
    """
    Convert lower-triangular vectors to symmetric zero-diagonal matrices.

    lower_vecs:
        shape (B, n*(n-1)//2) or (n*(n-1)//2,)

    returns:
        shape (B, n, n)
    """
    if lower_vecs.ndim == 1:
        lower_vecs = lower_vecs[None, :]

    device = lower_vecs.device
    B_, m_ = lower_vecs.shape

    assert m_ == n * (n - 1) // 2

    L = lower_vecs.new_zeros((B_, n, n))
    tril_idx = torch.tril_indices(n, n, offset=-1, device=device)

    L[:, tril_idx[0], tril_idx[1]] = lower_vecs

    return L + L.transpose(1, 2)


class WarmStartedArchakovHansenMap(nn.Module):
    """
    Archakov--Hansen inverse map from unrestricted lower-log-correlation
    coordinates to correlation matrices.

    The diagonal fixed-point solve is:
        x <- x - log(diag(exp(A + diag(x))))

    To make the map usable inside the ACV optimization loop, the fixed-point
    iteration is:
      - deterministic;
      - warm-started;
      - performed under torch.no_grad();
      - followed by one differentiable matrix exponential with x_star detached.

    This gives an approximate derivative that ignores dx_star/dgamma, matching
    the spirit of the current detached implementation but much faster.
    """

    def __init__(self, n, tol=1e-8, max_iter=100, verbose=False):
        super().__init__()

        self.n = n
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        self._x_cache = None
        self.last_num_iters = None
        self.last_diag_error = None

    def reset_cache(self):
        self._x_cache = None

    def forward(self, lower_vecs):
        if lower_vecs.ndim == 1:
            lower_vecs = lower_vecs[None, :]

        A = to_symmetric_tracefree_batch(lower_vecs, self.n)
        x_star = self._solve_diagonal(A)

        # Final reconstruction. Gradients flow through A but not through x_star.
        M = A + torch.diag_embed(x_star)
        C = torch.linalg.matrix_exp(M)

        # Numerical symmetrization and unit-diagonal cleanup.
        C = 0.5 * (C + C.transpose(-1, -2))

        d = torch.sqrt(torch.clamp(C.diagonal(dim1=-2, dim2=-1), min=1e-300))
        C = C / d[..., :, None] / d[..., None, :]
        C = 0.5 * (C + C.transpose(-1, -2))

        return C

    def _solve_diagonal(self, A):
        B, n, _ = A.shape

        with torch.no_grad():
            A_detached = A.detach()

            if self._x_cache is not None and self._x_cache.shape == (B, n):
                x = self._x_cache.to(dtype=A.dtype, device=A.device).clone()
            else:
                x = torch.zeros(B, n, dtype=A.dtype, device=A.device)

            for iteration in range(1, self.max_iter + 1):
                M = A_detached + torch.diag_embed(x)
                C = torch.linalg.matrix_exp(M)

                diagC = C.diagonal(dim1=-2, dim2=-1)
                err = torch.log(torch.clamp(diagC, min=1e-300))

                x_next = x - err

                if torch.max(torch.abs(err)) < self.tol:
                    x = x_next
                    break

                x = x_next

            self._x_cache = x.detach().clone()
            self.last_num_iters = iteration
            self.last_diag_error = float(torch.max(torch.abs(err)).cpu())

            if self.verbose:
                print(
                    "AH fixed point:",
                    f"iters={self.last_num_iters}",
                    f"diag_error={self.last_diag_error:.3e}",
                )

        return x


def to_unique_corr_matrix_batch(lower_vecs, n):
    """
    Backward-compatible stateless wrapper. Prefer WarmStartedArchakovHansenMap
    when repeatedly evaluating nearby points.
    """
    ah_map = WarmStartedArchakovHansenMap(n)
    return ah_map(lower_vecs)


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class VeclNet(nn.Module):
    """
    Network that produces unrestricted Archakov--Hansen coordinates.

    Args
    ----
    dim:
        Input dimension. Currently 1 for a single trainable ROM basis size.
    hidden:
        Hidden layer size.
    n:
        Full correlation matrix size.
    """

    def __init__(self, dim, hidden, n):
        super().__init__()

        self.n = n
        self.m = n * (n - 1) // 2
        self.dim = dim
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.m),
        )

    def forward(self, z):
        output = self.net(z)
        assert output.shape[-1] == self.m
        return output

    def corr_matrix(self, z, ah_map=None):
        if ah_map is None:
            ah_map = WarmStartedArchakovHansenMap(self.n)
        return ah_map(self.forward(z))


def train_ah_matrix_model(
    model,
    ah_map,
    inputs,
    targets,
    weights,
    lr=1e-2,
    max_steps=1000,
    tol=1e-9,
    grad_clip=1.0,
    print_every=50,
    optimizer_cls=optim.Adam,
    optimizer_kwargs=None,
):
    """
    Train omega_model(s) through the AH map using weighted matrix-entry loss.

    The fixed-fixed entries should receive a large weight, while ROM-dependent
    entries receive order-one weight. This does not hard-enforce fixed-fixed
    correlations, but strongly regularizes them while preserving PSD through AH.

    The up-weighting of fixed-fixed entries (often ~1e3) makes the early-step
    gradient large and highly non-stationary. Left unclipped, Adam can overshoot
    past the target correlation and land in the saturated |corr| -> 1 region of
    the AH map, where the gradient collapses to ~0. The optimizer then stalls
    there and the delta-loss stopping criterion below misreads that stall as
    convergence, silently handing a badly wrong (but "converged") surrogate to
    the caller. grad_clip guards against this overshoot; the caller additionally
    validates the fit and retries with a fresh initialization if needed.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    inputs = inputs.to(dtype=torch.float64)
    targets = targets.to(dtype=torch.float64)
    weights = weights.to(dtype=torch.float64)

    model = model.to(dtype=torch.float64)

    prev_loss = None
    loss_history = []

    denom = torch.sum(weights * targets**2).clamp_min(1e-16)

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()

        gamma = model(inputs)
        P_pred = ah_map(gamma)

        loss = torch.sum(weights * (P_pred - targets) ** 2) / denom

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        cur_loss = float(loss.detach().cpu().numpy())
        loss_history.append(cur_loss)

        if step % print_every == 0:
            print(f"[step {step:4d}/{max_steps}] AH matrix loss = {cur_loss:.6e}")

        if prev_loss is not None and abs(prev_loss - cur_loss) < tol:
            print(
                f"Converged at step {step} | "
                f"Δ loss = {abs(prev_loss - cur_loss):.2e} < tol={tol:.2e}"
            )
            break

        prev_loss = cur_loss

    else:
        print(f"Reached max_steps={max_steps} | final loss = {cur_loss:.6e}")

    return model, loss_history


class AHMatrixCorrelationSurrogate(nn.Module):
    """
    Matrix-valued correlation surrogate.

    This is the default AH path:
        s -> gamma_theta(s) -> AH(gamma_theta(s)) -> P(s)

    Fixed-fixed entries are not replaced componentwise. Instead, they are
    strongly regularized during training so the returned matrix remains a valid
    AH correlation matrix.
    """

    def __init__(
        self,
        omega_model,
        ah_map,
        s_min,
        s_max,
        psd_check="none",
    ):
        super().__init__()

        self.omega_model = omega_model
        self.ah_map = ah_map

        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.psd_check = psd_check

    def normalize_s(self, s):
        return 2.0 * (s - self.s_min) / (self.s_max - self.s_min) - 1.0

    def forward(self, s_active):
        if torch.is_tensor(s_active):
            s = s_active.reshape(-1)[-1]
            dtype = s.dtype
            device = s.device
        else:
            s = torch.as_tensor(
                np.asarray(s_active).reshape(-1)[-1],
                dtype=torch.float64,
            )
            dtype = torch.float64
            device = s.device

        z = self.normalize_s(s).reshape(1, 1).to(dtype=dtype, device=device)

        gamma = self.omega_model(z)
        P = self.ah_map(gamma)[0]

        if self.psd_check != "none":
            self._check_psd(P)

        return P

    def _check_psd(self, P, tolerance=-1e-8):
        with torch.no_grad():
            eig_min = float(torch.linalg.eigvalsh(P).min().cpu())

        if eig_min < tolerance:
            msg = f"AH matrix surrogate produced min eigenvalue {eig_min:.3e}."
            if self.psd_check == "raise":
                raise RuntimeError(msg)
            if self.psd_check == "warn":
                print(f"Warning: {msg}")

        return eig_min


class SurrogateBuilder:
    """
    Builds cost and correlation surrogates.

    Supported correlation surrogate modes
    -------------------------------------

    ah_matrix:
        New default. Trains a matrix-valued AH surrogate and returns
        corr_matrix_fn(s). This preserves global admissibility of the returned
        correlation matrix.

    ah_componentwise_sigmoid:
        Legacy current behavior. Trains an AH/VeclNet matrix surrogate, samples
        its entries on a dense grid, then fits componentwise scalar surrogates.
        This does not preserve the global matrix guarantee after scalar refit.

    componentwise_sigmoid:
        Direct scalar sigmoid/polynomial fitting to pilot correlations and
        costs. Does not use AH.
    """

    def __init__(
        self,
        pilot_list,
        n_active,
        n_aux,
        work_dir=None,
        method="ah_matrix",
        tunable_range=None,
        use_torch=True,
        fixed_fixed_weight=1.0,
        ah_tol=1e-8,
        ah_max_iter=1000,
        ah_psd_check="none",
        ah_fixed_fixed_tol=0.05,
    ):
        self.pilot_list = list(pilot_list)
        self.n_active = n_active
        self.n_aux = n_aux
        self.n_models = 1 + n_aux + n_active
        self.work_dir = work_dir
        self.use_torch = use_torch
        self.tunable_range = tunable_range or [min(pilot_list), max(pilot_list)]
        self.fixed_fixed_weight = fixed_fixed_weight
        self.ah_tol = ah_tol
        self.ah_max_iter = ah_max_iter
        self.ah_psd_check = ah_psd_check
        self.ah_fixed_fixed_tol = ah_fixed_fixed_tol

        # Backward-compatible aliases.
        if method == "neural_network":
            method = "ah_componentwise_sigmoid"
        elif method == "sigmoid":
            method = "componentwise_sigmoid"

        valid_methods = {
            "ah_matrix",
            "ah_componentwise_sigmoid",
            "componentwise_sigmoid",
        }

        if method not in valid_methods:
            raise ValueError(
                f"Unknown surrogate method '{method}'. "
                f"Valid methods are {sorted(valid_methods)}."
            )

        self.method = method

        if work_dir:
            self.model_path = os.path.join(
                work_dir,
                f"vecl_correlation_model_{self.method}.pt",
            )
        else:
            self.model_path = None

    # ------------------------------------------------------------------
    # Public build
    # ------------------------------------------------------------------

    def build(self, data_npz):
        """
        Returns
        -------
        hf_corr_list, lf_corr_list, cost_list, corr_matrix_fn

        For scalar modes, corr_matrix_fn is None.

        For ah_matrix, hf_corr_list and lf_corr_list are None and the optimizer
        should use corr_matrix_fn.
        """
        with np.load(data_npz) as data:
            fom_aux_corrs = data["fom_aux_corrs"]
            aux_aux_corrs = data.get("aux_aux_corrs", np.array([]))
            fom_rom_corrs = data["fom_rom_corrs"]
            aux_rom_corrs_list = [
                data[f"aux{i}_rom_corrs"] for i in range(self.n_aux)
            ]
            norm_aux_times = data["normalized_aux_times"]
            norm_rom_times = data["normalized_rom_times"]

        if self.method == "componentwise_sigmoid":
            print("Building direct componentwise sigmoid surrogates")
            return self._build_componentwise_sigmoid(
                fom_aux_corrs,
                aux_aux_corrs,
                fom_rom_corrs,
                aux_rom_corrs_list,
                norm_aux_times,
                norm_rom_times,
            )

        if self.method == "ah_componentwise_sigmoid":
            print("Building legacy AH + componentwise sigmoid surrogates")
            return self._build_ah_componentwise_sigmoid(
                fom_aux_corrs,
                aux_aux_corrs,
                fom_rom_corrs,
                aux_rom_corrs_list,
                norm_aux_times,
                norm_rom_times,
            )

        print("Building matrix-valued Archakov--Hansen surrogate")
        return self._build_ah_matrix(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
            norm_aux_times,
            norm_rom_times,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _rom_input(self, s):
        """Extract ROM coordinate from state vector or pass through scalar."""
        if torch.is_tensor(s):
            return s if s.ndim == 0 else s[-1]
        return s if np.isscalar(s) else s[-1]

    def _wrap(self, func):
        """Wrap scalar surrogate to handle vector inputs."""
        def wrapped(s):
            s_input = self._rom_input(s)
            result = func(s_input)

            if self.use_torch and torch.is_tensor(s):
                if not torch.is_tensor(result):
                    result = torch.tensor(
                        result,
                        dtype=torch.float64,
                        device=s.device,
                    )

            return result

        return wrapped

    def _make_constant(self, value):
        """Create a constant function compatible with numpy and torch."""
        if self.use_torch:
            def const_fn(s):
                if torch.is_tensor(s):
                    return torch.tensor(
                        value,
                        dtype=torch.float64,
                        device=s.device,
                    )
                return value

            return const_fn

        return lambda s: value

    def _build_cost_list(self, norm_aux_times, norm_rom_times):
        if self.use_torch:
            cost_rom_surr = fit_polynomial_torch(
                np.array(self.pilot_list)[None, :],
                norm_rom_times,
                order=1,
            )
        else:
            cost_rom_surr = fit_polynomial(
                np.array(self.pilot_list)[None, :],
                norm_rom_times,
                order=1,
            )

        cost_list = [self._make_constant(float(t)) for t in norm_aux_times]
        cost_list.append(self._wrap(cost_rom_surr))

        return cost_list

    # ------------------------------------------------------------------
    # Direct scalar path
    # ------------------------------------------------------------------

    def _build_componentwise_sigmoid(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        norm_aux_times,
        norm_rom_times,
    ):
        pilots = np.array(self.pilot_list)

        if self.use_torch:
            fit_sig = fit_sigmoid_torch
        else:
            fit_sig = fit_sigmoid

        fom_rom_surr = fit_sig(pilots[None, :], fom_rom_corrs)
        aux_rom_surrs = [
            fit_sig(pilots[None, :], corrs) for corrs in aux_rom_corrs_list
        ]

        hf_corr_list = [self._make_constant(float(c)) for c in fom_aux_corrs]
        hf_corr_list.append(self._wrap(fom_rom_surr))

        lf_corr_list = [self._make_constant(float(c)) for c in aux_aux_corrs]
        lf_corr_list.extend([self._wrap(surr) for surr in aux_rom_surrs])

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times)

        return hf_corr_list, lf_corr_list, cost_list, None

    # ------------------------------------------------------------------
    # AH matrix path
    # ------------------------------------------------------------------

    def _build_ah_matrix(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        norm_aux_times,
        norm_rom_times,
    ):
        n = self.n_models
        hidden_size = 4

        model, ah_map = self._load_or_train_ah_matrix_model(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
            n,
            hidden_size,
        )

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times)

        corr_matrix_fn = AHMatrixCorrelationSurrogate(
            omega_model=model,
            ah_map=ah_map,
            s_min=self.tunable_range[0],
            s_max=self.tunable_range[1],
            psd_check=self.ah_psd_check,
        )

        return None, None, cost_list, corr_matrix_fn

    def _load_or_train_ah_matrix_model(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        n,
        hidden_size,
    ):
        # Needed both to validate a cached model and, if training is
        # required, to hand off to _train_ah_matrix_model.
        inputs = torch.tensor(self.pilot_list, dtype=torch.float64).reshape(-1, 1)
        inputs_norm = self._normalize_training_inputs(inputs)

        targets, weights = self._assemble_ah_targets_and_weights(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
        )

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location="cpu")

            compatible = (
                checkpoint.get("pilot_list") == self.pilot_list
                and checkpoint.get("n_active") == self.n_active
                and checkpoint.get("n_aux") == self.n_aux
                and checkpoint.get("method") == self.method
            )

            if compatible:
                model = VeclNet(1, hidden_size, n).double()
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                ah_map = WarmStartedArchakovHansenMap(
                    n,
                    tol=self.ah_tol,
                    max_iter=self.ah_max_iter,
                )

                # A cached checkpoint may predate this validate-and-retry
                # safeguard (e.g. saved by an earlier, unstable training
                # run). Loading it unconditionally would silently bypass
                # retraining forever, since the file looks "compatible" on
                # every subsequent run. Validate it exactly as a freshly
                # trained model would be validated.
                err = self._fixed_fixed_fit_error(
                    model, ah_map, inputs_norm, targets, weights
                )

                if err <= self.ah_fixed_fixed_tol:
                    print(
                        f"Loading AH matrix model from {self.model_path} "
                        f"(fixed-fixed fit error {err:.4f})\n"
                    )
                    return model, ah_map

                print(
                    f"Cached AH model at {self.model_path} fails fixed-fixed "
                    f"validation (error {err:.4f} > {self.ah_fixed_fixed_tol:.4f}); "
                    f"discarding and retraining\n"
                )
            else:
                print("Cached AH model incompatible; retraining\n")

            os.remove(self.model_path)

        return self._train_ah_matrix_model(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
            n,
            hidden_size,
        )

    def _train_ah_matrix_model(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        n,
        hidden_size,
        max_restarts=5,
        fixed_fixed_fit_tol=None,
    ):
        if fixed_fixed_fit_tol is None:
            fixed_fixed_fit_tol = self.ah_fixed_fixed_tol

        print(
            f"Training AH matrix surrogate: "
            f"n={n} (1 FOM + {self.n_aux} aux + {self.n_active} ROM)\n"
        )

        inputs = torch.tensor(
            self.pilot_list,
            dtype=torch.float64,
        ).reshape(-1, 1)

        inputs_norm = self._normalize_training_inputs(inputs)

        targets, weights = self._assemble_ah_targets_and_weights(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
        )

        # The fixed-fixed entries (FOM-aux, aux-aux) are known exactly from the
        # pilot data and should not depend on s at all. Training can
        # occasionally overshoot into the saturated |corr| -> 1 region of the
        # AH map and get stuck there (a spurious but low-gradient point that
        # the delta-loss stopping rule misreads as convergence). Rather than
        # silently pass a corrupted surrogate downstream to the ACV
        # optimizer, validate the fixed-fixed fit and retry with a fresh
        # random initialization if it is off.
        best_model, best_ah_map, best_err = None, None, np.inf

        for attempt in range(1, max_restarts + 1):
            model = VeclNet(1, hidden_size, n).double()

            ah_map = WarmStartedArchakovHansenMap(
                n,
                tol=self.ah_tol,
                max_iter=self.ah_max_iter,
            )

            model, _ = train_ah_matrix_model(
                model,
                ah_map,
                inputs_norm,
                targets,
                weights,
                lr=1e-2,
                max_steps=2000,
                tol=1e-9,
                grad_clip=1.0,
                print_every=50,
            )

            model.eval()

            err = self._fixed_fixed_fit_error(
                model, ah_map, inputs_norm, targets, weights
            )

            if err < best_err:
                best_model, best_ah_map, best_err = model, ah_map, err

            if err <= fixed_fixed_fit_tol:
                break

            print(
                f"AH matrix attempt {attempt}/{max_restarts}: "
                f"fixed-fixed fit error {err:.4f} exceeds tolerance "
                f"{fixed_fixed_fit_tol:.4f}; retrying with a new init\n"
            )
        else:
            print(
                f"Warning: AH matrix surrogate did not reach the fixed-fixed "
                f"fit tolerance after {max_restarts} attempts "
                f"(best error {best_err:.4f}). Using the best fit found.\n"
            )

        model, ah_map = best_model, best_ah_map

        if self.model_path:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pilot_list": self.pilot_list,
                    "n_active": self.n_active,
                    "n_aux": self.n_aux,
                    "n_models": n,
                    "hidden_size": hidden_size,
                    "method": self.method,
                    "fixed_fixed_weight": self.fixed_fixed_weight,
                },
                self.model_path,
            )

            print(f"Saved AH matrix model to {self.model_path}\n")

        return model, ah_map

    def _normalize_training_inputs(self, inputs):
        s_min, s_max = self.tunable_range
        return 2.0 * (inputs - s_min) / (s_max - s_min) - 1.0

    def _fixed_fixed_fit_error(self, model, ah_map, inputs_norm, targets, weights):
        """
        Max absolute error between the trained surrogate and the known
        fixed-fixed correlation targets (entries carrying fixed_fixed_weight).
        These entries do not depend on s, so a good fit should match them
        closely at every pilot point; a large error signals the training run
        landed in a saturated, low-gradient region of the AH map.
        """
        with torch.no_grad():
            ah_map.reset_cache()
            P_pred = ah_map(model(inputs_norm))

        mask = weights >= (0.5 * self.fixed_fixed_weight)

        if not torch.any(mask):
            return 0.0

        return float(torch.max(torch.abs(P_pred[mask] - targets[mask])))

    def _assemble_ah_targets_and_weights(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
    ):
        """
        Assemble target correlation matrices and matrix-entry weights.

        Current implementation supports one active ROM, matching the existing
        workflow. The layout is:

            index 0: FOM
            indices 1..n_aux: fixed auxiliary models
            final index: trainable ROM

        Fixed-fixed entries receive fixed_fixed_weight.
        ROM-dependent entries receive weight 1.
        """
        if self.n_active != 1:
            raise NotImplementedError(
                "The current AH target assembly supports one active ROM. "
                "Extend this function for multiple trainable ROMs."
            )

        pilots = np.array(self.pilot_list)
        G = len(pilots)

        n = self.n_models
        n_fixed = 1 + self.n_aux
        rom_idx = n - 1

        targets = np.zeros((G, n, n), dtype=float)
        weights = np.zeros((G, n, n), dtype=float)

        for g in range(G):
            P = np.eye(n, dtype=float)
            W = np.zeros((n, n), dtype=float)

            # FOM-aux fixed correlations.
            for i in range(self.n_aux):
                idx = i + 1
                P[idx, 0] = fom_aux_corrs[i]
                P[0, idx] = fom_aux_corrs[i]

                W[idx, 0] = self.fixed_fixed_weight
                W[0, idx] = self.fixed_fixed_weight

            # Aux-aux fixed correlations.
            aux_pair_idx = 0
            for i in range(self.n_aux):
                for j in range(i):
                    idx_i = i + 1
                    idx_j = j + 1

                    corr = aux_aux_corrs[aux_pair_idx]
                    P[idx_i, idx_j] = corr
                    P[idx_j, idx_i] = corr

                    W[idx_i, idx_j] = self.fixed_fixed_weight
                    W[idx_j, idx_i] = self.fixed_fixed_weight

                    aux_pair_idx += 1

            # FOM-ROM varying correlation.
            P[rom_idx, 0] = fom_rom_corrs[g]
            P[0, rom_idx] = fom_rom_corrs[g]

            W[rom_idx, 0] = 1.0
            W[0, rom_idx] = 1.0

            # Aux-ROM varying correlations.
            for i in range(self.n_aux):
                idx = i + 1
                corr = aux_rom_corrs_list[i][g]

                P[rom_idx, idx] = corr
                P[idx, rom_idx] = corr

                W[rom_idx, idx] = 1.0
                W[idx, rom_idx] = 1.0

            # Diagonal is guaranteed by AH, but a small weight can improve
            # numerical behavior. Keep this modest.
            for i in range(n):
                W[i, i] = 1.0

            targets[g] = P
            weights[g] = W

        return (
            torch.tensor(targets, dtype=torch.float64),
            torch.tensor(weights, dtype=torch.float64),
        )

    # ------------------------------------------------------------------
    # Legacy AH + componentwise sigmoid path
    # ------------------------------------------------------------------

    def _build_ah_componentwise_sigmoid(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        norm_aux_times,
        norm_rom_times,
    ):
        n = self.n_models
        hidden_size = 4

        # Reuse the AH matrix training routine, but then intentionally discard
        # the matrix-valued surrogate by fitting scalar surrogates to entries.
        model, ah_map = self._load_or_train_ah_matrix_model(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
            n,
            hidden_size,
        )

        hf_corr_list, lf_corr_list = self._fit_surrogates_to_model(
            model,
            ah_map,
            n,
        )

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times)

        return hf_corr_list, lf_corr_list, cost_list, None

    def _fit_surrogates_to_model(self, model, ah_map, n):
        """
        Legacy behavior: query AH matrix model on a dense grid and fit
        componentwise scalar surrogates. This path does not preserve global
        admissibility after the scalar refit.
        """
        s_grid = np.unique(
            np.concatenate(
                [
                    self.pilot_list,
                    np.linspace(
                        self.tunable_range[0],
                        self.tunable_range[1],
                        200,
                    ),
                ]
            )
        )

        s_tensor = torch.tensor(s_grid, dtype=torch.float64).reshape(-1, 1)
        s_tensor_norm = self._normalize_training_inputs(s_tensor)

        with torch.no_grad():
            ah_map.reset_cache()
            corr_matrices = model.corr_matrix(s_tensor_norm, ah_map).cpu().numpy()

        if self.work_dir:
            self._plot_correlations(corr_matrices, s_grid, n)

        if self.use_torch:
            surrogates = self._fit_torch_surrogates(corr_matrices, s_grid, n)
        else:
            surrogates = self._fit_numpy_surrogates(corr_matrices, s_grid, n)

        hf_corr_list = [surrogates[(i, 0)] for i in range(1, n)]
        lf_corr_list = [
            surrogates[(i, j)]
            for i in range(1, n)
            for j in range(1, i)
        ]

        return hf_corr_list, lf_corr_list

    def _fit_torch_surrogates(self, corr_matrices, s_grid, n):
        surrogates = {}

        for i in range(n):
            for j in range(i):
                values = corr_matrices[:, i, j]

                if np.std(values) < 0.01:
                    surrogates[(i, j)] = self._make_constant(float(np.mean(values)))
                else:
                    sig = fit_sigmoid_torch(s_grid[None, :], values)
                    surrogates[(i, j)] = lambda s, f=sig: f(self._rom_input(s))

        return surrogates

    def _fit_numpy_surrogates(self, corr_matrices, s_grid, n):
        from scipy.interpolate import interp1d

        surrogates = {}

        for i in range(n):
            for j in range(i):
                values = corr_matrices[:, i, j]

                if np.std(values) < 0.01:
                    surrogates[(i, j)] = self._make_constant(float(np.mean(values)))
                else:
                    try:
                        sig = fit_sigmoid(s_grid[None, :], values)
                        test = np.array([sig(s) for s in s_grid])

                        if np.mean((test - values) ** 2) < 0.01:
                            surrogates[(i, j)] = (
                                lambda s, f=sig: float(f(self._rom_input(s)))
                            )
                        else:
                            raise ValueError("Poor sigmoid fit")

                    except Exception:
                        interp = interp1d(
                            s_grid,
                            values,
                            kind="cubic",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        surrogates[(i, j)] = (
                            lambda s, f=interp: float(f(self._rom_input(s)))
                        )

        return surrogates

    def _plot_correlations(self, corr_matrices, s_grid, n):
        """
        Keep your existing diagnostic plotting implementation here.

        The body of your current _plot_correlations method can remain mostly
        unchanged. It consumes corr_matrices, s_grid, and n in the same way.
        """
        try:
            import matplotlib.pyplot as plt

            debug_dir = os.path.join(self.work_dir, "debug_plots")
            os.makedirs(debug_dir, exist_ok=True)

            with np.load(os.path.join(self.work_dir, "pilot_results.npz")) as data:
                fom_aux = data["fom_aux_corrs"]
                aux_aux = data.get("aux_aux_corrs", np.array([]))
                fom_rom = data["fom_rom_corrs"]
                aux_rom = [data[f"aux{i}_rom_corrs"] for i in range(self.n_aux)]

            pilot_data, names = {}, {}

            for i in range(self.n_aux):
                pilot_data[(i + 1, 0)] = np.full(len(self.pilot_list), fom_aux[i])
                names[(i + 1, 0)] = f"FOM-aux{i}"

            if self.n_aux > 1:
                idx = 0
                for i in range(self.n_aux):
                    for j in range(i):
                        pilot_data[(i + 1, j + 1)] = np.full(
                            len(self.pilot_list),
                            aux_aux[idx],
                        )
                        names[(i + 1, j + 1)] = f"aux{j}-aux{i}"
                        idx += 1

            rom_idx = n - 1
            pilot_data[(rom_idx, 0)] = fom_rom
            names[(rom_idx, 0)] = "FOM-ROM"

            for i in range(self.n_aux):
                pilot_data[(rom_idx, i + 1)] = aux_rom[i]
                names[(rom_idx, i + 1)] = f"aux{i}-ROM"

            n_plots = n * (n - 1) // 2
            ncols = min(3, n_plots)
            nrows = (n_plots + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6 * ncols, 5 * nrows),
            )

            axes = np.array([axes]).flatten() if n_plots == 1 else axes.flatten()

            plot_idx = 0
            for i in range(n):
                for j in range(i):
                    ax = axes[plot_idx]
                    nn_vals = corr_matrices[:, i, j]

                    ax.plot(
                        s_grid,
                        nn_vals,
                        "b-",
                        label="AH surrogate",
                        linewidth=2,
                        alpha=0.7,
                    )

                    if (i, j) in pilot_data:
                        ax.plot(
                            self.pilot_list,
                            pilot_data[(i, j)],
                            "ro",
                            label="Pilot",
                            markersize=8,
                            zorder=5,
                        )

                    ax.set_title(
                        names.get((i, j), f"({i},{j})"),
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.set_xlabel("ROM basis size")
                    ax.set_ylabel("Correlation")
                    ax.set_ylim([-1.05, 1.05])
                    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=9)

                    plot_idx += 1

            for idx in range(plot_idx, len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(debug_dir, "correlation_fits.png"),
                dpi=150,
                bbox_inches="tight",
            )
            print(f"\nSaved plot to {debug_dir}/correlation_fits.png\n")
            plt.close()

        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")