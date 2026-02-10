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
# BATCHED CORRELATION MATRIX OPERATIONS
# ============================================================================

def to_symmetric_tracefree_batch(lower_vecs, n):  
    """
    lower_vecs: (B, m)
    returns:    (B, n, n) symmetric, zero‐trace
    """
    device = lower_vecs.device
    B_, m_ = lower_vecs.shape
    assert m_ == n*(n-1)//2
    L = lower_vecs.new_zeros((B_, n, n))
    tril_idx = torch.tril_indices(n, n, offset=-1, device=device)
    L[:, tril_idx[0], tril_idx[1]] = lower_vecs
    return L + L.transpose(1, 2)


def fixed_point_solve_batch(A_batch, tol=1e-6, max_iter=2500):
    """
    A_batch: (B,n,n)
    returns: x* of shape (B,n)
    solves x = x - log diag exp(A + diag(x)) in batch
    """
    B_, n_, _ = A_batch.shape
    x = torch.randn(B_, n_, device=A_batch.device, dtype=A_batch.dtype).detach()
    for _ in range(max_iter):
        A_new  = A_batch + torch.diag_embed(x)
        C_new  = torch.linalg.matrix_exp(A_new)
        diagC  = C_new.diagonal(dim1=-2, dim2=-1)
        x_next = (x - torch.log(diagC)).detach()
        if torch.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next
    return x


def to_unique_corr_matrix_batch(lower_vecs, n):
    """
    lower_vecs: (B, m)
    returns:    (B, n, n) valid correlation matrices
    """
    A_batch = to_symmetric_tracefree_batch(lower_vecs, n)
    d_star  = fixed_point_solve_batch(A_batch, tol=1e-6, max_iter=2500)
    M       = A_batch + torch.diag_embed(d_star)
    return torch.linalg.matrix_exp(M)


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class VeclNet(nn.Module):
    """
    Network that produces full lower triangle of correlation matrix.
    
    Args:
        dim: input latent dimension
        hidden: hidden layer size
        n: matrix size (number of models)
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
            nn.Linear(hidden, self.m)
        )

    def forward(self, z):
        """
        z: (B, dim)
        returns: (B, m) - all lower triangle entries
        """
        output = self.net(z)
        assert output.shape[-1] == self.m
        return output

    def corr_matrix(self, z, n=None):
        """
        z: (B, dim)
        n: matrix size (should match self.n)
        returns: (B, n, n) valid correlation matrices
        """
        if n is None:
            n = self.n
        else:
            assert n == self.n
        
        lower_entries = self.forward(z)
        return to_unique_corr_matrix_batch(lower_entries, n)


def train_model(
    model,
    inputs,
    targets,
    n,
    lr=1e-2,
    max_steps=500,
    tol=1e-8,
    print_every=50,
    optimizer_cls=optim.Adam,
    optimizer_kwargs=None,
    random_seed=torch.manual_seed(2025)
):
    """
    Train model to approximate target correlation matrices.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    prev_loss = None
    loss_history = []

    targets.to(inputs.device)
    denom = F.mse_loss(targets, torch.zeros_like(targets))

    for step in range(1, max_steps+1):
        optimizer.zero_grad()

        vecl_batch = model(inputs)
        C_batch    = to_unique_corr_matrix_batch(vecl_batch, n)

        loss = F.mse_loss(C_batch, targets) / denom
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        loss_history.append(cur_loss)

        if step % print_every == 0:
            print(f"[step {step:4d}/{max_steps}] loss = {cur_loss:.6e}")

        if prev_loss is not None:
            if abs(prev_loss - cur_loss) < tol:
                print(f"Converged at step {step} | Δ loss = {abs(prev_loss - cur_loss):.2e} < tol={tol:.2e}")
                break

        prev_loss = cur_loss

    else:
        print(f"Reached max_steps={max_steps} | final loss = {cur_loss:.6e}")

    return model, loss_history


class SurrogateBuilder:
    """Builds surrogate models for correlation and cost functions.
    
    Surrogates are differentiable and compatible with PyTorch autograd.
    """
    
    def __init__(self, pilot_list, n_active, n_aux, work_dir=None, method='neural_network',
                 tunable_range=None, use_torch=True):
        self.pilot_list = pilot_list
        self.n_active = n_active
        self.n_aux = n_aux
        self.n_models = 1 + n_aux + n_active
        self.work_dir = work_dir
        self.method = method
        self.use_torch = use_torch  # Use PyTorch fitting for autodiff
        self.tunable_range = tunable_range or [min(pilot_list), max(pilot_list)]
        self.model_path = os.path.join(work_dir, "vecl_correlation_model.pt") if work_dir else None
        
        if method not in ['neural_network', 'sigmoid']:
            raise ValueError("method must be 'neural_network' or 'sigmoid'")
    
    def build(self, data_npz):
        """Build surrogate models from pilot data."""
        with np.load(data_npz) as data:
            fom_aux_corrs = data['fom_aux_corrs']
            aux_aux_corrs = data.get('aux_aux_corrs', np.array([]))
            fom_rom_corrs = data['fom_rom_corrs']
            aux_rom_corrs_list = [data[f'aux{i}_rom_corrs'] for i in range(self.n_aux)]
            norm_aux_times = data['normalized_aux_times']
            norm_rom_times = data['normalized_rom_times']
        
        if self.method == 'sigmoid':
            print('Building sigmoid surrogates')
            return self._build_sigmoid(fom_aux_corrs, aux_aux_corrs, fom_rom_corrs,
                                      aux_rom_corrs_list, norm_aux_times, norm_rom_times)
        else:
            print('Building neural network surrogate')
            return self._build_neural_net(fom_aux_corrs, aux_aux_corrs, fom_rom_corrs,
                                         aux_rom_corrs_list, norm_aux_times, norm_rom_times)
    
    def _rom_input(self, s):
        """Extract ROM coordinate from state vector or pass through scalar."""
        if torch.is_tensor(s):
            return s if s.ndim == 0 else s[-1]
        return s if np.isscalar(s) else s[-1]
    
    def _wrap(self, func):
        """Wrap function to handle both scalar and vector inputs while preserving tensor type."""
        def wrapped(s):
            s_input = self._rom_input(s)
            result = func(s_input)
            # Ensure result is a tensor if input was a tensor
            if self.use_torch and torch.is_tensor(s):
                if not torch.is_tensor(result):
                    result = torch.tensor(result, dtype=torch.float64, device=s.device)
            return result
        return wrapped
    
    def _build_sigmoid(self, fom_aux_corrs, aux_aux_corrs, fom_rom_corrs,
                      aux_rom_corrs_list, norm_aux_times, norm_rom_times):
        """Build surrogates by fitting sigmoids directly to pilot data."""
        pilots = np.array(self.pilot_list)
        
        # Choose fitting function based on use_torch flag
        if self.use_torch:
            fit_sig = fit_sigmoid_torch
            fit_poly = fit_polynomial_torch
        else:
            fit_sig = fit_sigmoid
            fit_poly = fit_polynomial
        
        # Fit ROM-dependent surrogates
        fom_rom_surr = fit_sig(pilots[None, :], fom_rom_corrs)
        aux_rom_surrs = [fit_sig(pilots[None, :], corrs) for corrs in aux_rom_corrs_list]
        cost_rom_surr = fit_poly(pilots[None, :], norm_rom_times, order=1)
        
        # HF correlations: FOM vs. auxiliaries (constant) and ROM (surrogate)
        hf_corr_list = [self._make_constant(float(c)) for c in fom_aux_corrs]
        hf_corr_list.append(self._wrap(fom_rom_surr))
        
        # LF correlations: aux-aux (constant) and aux-ROM (surrogate)
        lf_corr_list = [self._make_constant(float(c)) for c in aux_aux_corrs]
        lf_corr_list.extend([self._wrap(surr) for surr in aux_rom_surrs])
        
        # Costs: auxiliaries (constant) and ROM (surrogate)
        cost_list = [self._make_constant(float(t)) for t in norm_aux_times]
        cost_list.append(self._wrap(cost_rom_surr))
        
        return hf_corr_list, lf_corr_list, cost_list
    
    def _make_constant(self, value):
        """Create a constant function that works with both numpy and torch."""
        if self.use_torch:
            # Return a scalar tensor that can be stacked
            def const_fn(s):
                # Match dtype based on input type
                if torch.is_tensor(s):
                    return torch.tensor(value, dtype=torch.float64, device=s.device)
                else:
                    return value
            return const_fn
        else:
            return lambda s: value
    
    def _build_neural_net(self, fom_aux_corrs, aux_aux_corrs, fom_rom_corrs,
                        aux_rom_corrs_list, norm_aux_times, norm_rom_times):
        """Build surrogates using neural network then fitting sigmoids to output."""
        n = self.n_models
        m = n * (n - 1) // 2
        hidden_size = 4
        
        # Load or train model
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if (checkpoint.get('pilot_list') == self.pilot_list and
                checkpoint.get('n_active') == self.n_active and
                checkpoint.get('n_aux') == self.n_aux):
                print(f"Loading model from {self.model_path}\n")
                model = VeclNet(1, hidden_size, n)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            else:
                print("Cached model incompatible, retraining\n")
                os.remove(self.model_path)
                model = self._train_model(fom_aux_corrs, aux_aux_corrs, fom_rom_corrs,
                                         aux_rom_corrs_list, n, m, hidden_size)
        else:
            model = self._train_model(fom_aux_corrs, aux_aux_corrs, fom_rom_corrs,
                                     aux_rom_corrs_list, n, m, hidden_size)
        
        # Extract correlation surrogates from trained model
        hf_corr_list, lf_corr_list = self._fit_surrogates_to_model(model, n)
        
        # Build cost list
        if self.use_torch:
            cost_rom_surr = fit_polynomial_torch(np.array(self.pilot_list)[None, :], 
                                                 norm_rom_times, order=1)
        else:
            cost_rom_surr = fit_polynomial(np.array(self.pilot_list)[None, :], 
                                          norm_rom_times, order=1)
        
        cost_list = [self._make_constant(float(t)) for t in norm_aux_times]
        cost_list.append(self._wrap(cost_rom_surr))
        
        return hf_corr_list, lf_corr_list, cost_list
    
    def _train_model(self, fom_aux_corrs, aux_aux_corrs, fom_rom_corrs,
                    aux_rom_corrs_list, n, m, hidden_size):
        """Train a new VeclNet model."""
        print(f"Training VeclNet: n={n} (1 FOM + {self.n_aux} aux + {self.n_active} ROM), m={m}\n")
        
        inputs = torch.tensor(self.pilot_list, dtype=torch.float32).reshape(-1, 1)
        
        # Build correlation vector (lower triangle)
        half_entries = []
        for c in fom_aux_corrs:
            half_entries.append(torch.full((len(self.pilot_list),), float(c)))
        if self.n_aux > 1:
            for c in aux_aux_corrs:
                half_entries.append(torch.full((len(self.pilot_list),), float(c)))
        half_entries.append(torch.tensor(fom_rom_corrs, dtype=torch.float32))
        for corrs in aux_rom_corrs_list:
            half_entries.append(torch.tensor(corrs, dtype=torch.float32))
        
        half = torch.stack(half_entries, dim=1)
        target_matrices = to_symmetric_tracefree_batch(half, n)
        target_matrices += torch.diag_embed(torch.ones(inputs.shape[0], n))
        
        # Train
        model = VeclNet(1, hidden_size, n)
        model, _ = train_model(model, inputs, target_matrices, n=n, lr=1e-1,
                             max_steps=2000, tol=1e-8, print_every=50)
        model.eval()
        
        # Save
        if self.model_path:
            torch.save({'model_state_dict': model.state_dict(), 'pilot_list': self.pilot_list,
                       'n_active': self.n_active, 'n_aux': self.n_aux,
                       'n_models': n, 'hidden_size': hidden_size}, self.model_path)
            print(f"Saved model to {self.model_path}\n")
        
        return model
    
    def _fit_surrogates_to_model(self, model, n):
        """Query model on dense grid and fit surrogates to each correlation entry."""
        # Create dense grid
        s_grid = np.unique(np.concatenate([
            self.pilot_list, np.linspace(self.tunable_range[0], self.tunable_range[1], 200)
        ]))
        
        # Query model
        s_tensor = torch.tensor(s_grid, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            corr_matrices = model.corr_matrix(s_tensor, n).numpy()
        
        if self.work_dir:
            self._plot_correlations(corr_matrices, s_grid, n)
        
        # Choose fitting approach
        if self.use_torch:
            surrogates = self._fit_torch_surrogates(corr_matrices, s_grid, n)
        else:
            surrogates = self._fit_numpy_surrogates(corr_matrices, s_grid, n)
        
        # Extract HF and LF lists
        hf_corr_list = [surrogates[(i, 0)] for i in range(1, n)]
        lf_corr_list = [surrogates[(i, j)] for i in range(1, n) for j in range(1, i)]
        
        return hf_corr_list, lf_corr_list
    
    def _fit_torch_surrogates(self, corr_matrices, s_grid, n):
        """Fit PyTorch-based surrogates that preserve gradients."""
        
        surrogates = {}
        for i in range(n):
            for j in range(i):
                values = corr_matrices[:, i, j]
                
                if np.std(values) < 0.01:  # Nearly constant
                    surrogates[(i, j)] = self._make_constant(float(np.mean(values)))
                else:
                    # Fit sigmoid (PyTorch version for autodiff)
                    sig = fit_sigmoid_torch(s_grid[None, :], values)
                    surrogates[(i, j)] = lambda s, f=sig: f(self._rom_input(s))
        
        return surrogates
    
    def _fit_numpy_surrogates(self, corr_matrices, s_grid, n):
        """Fit numpy-based surrogates (original approach)."""
        from scipy.interpolate import interp1d
        
        surrogates = {}
        for i in range(n):
            for j in range(i):
                values = corr_matrices[:, i, j]
                
                if np.std(values) < 0.01:  # Nearly constant
                    surrogates[(i, j)] = self._make_constant(float(np.mean(values)))
                else:
                    try:  # Try sigmoid
                        sig = fit_sigmoid(s_grid[None, :], values)
                        test = np.array([sig(s) for s in s_grid])
                        if np.mean((test - values) ** 2) < 0.01:
                            surrogates[(i, j)] = lambda s, f=sig: float(f(self._rom_input(s)))
                        else:
                            raise ValueError("Poor fit")
                    except:  # Fall back to interpolation
                        interp = interp1d(s_grid, values, kind='cubic', 
                                        bounds_error=False, fill_value='extrapolate')
                        surrogates[(i, j)] = lambda s, f=interp: float(f(self._rom_input(s)))
        
        return surrogates
    
    def _plot_correlations(self, corr_matrices, s_grid, n):
        """Generate diagnostic plots."""
        try:
            import matplotlib.pyplot as plt
            debug_dir = os.path.join(self.work_dir, "debug_plots")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Load pilot data
            with np.load(os.path.join(self.work_dir, "pilot_results.npz")) as data:
                fom_aux = data['fom_aux_corrs']
                aux_aux = data.get('aux_aux_corrs', np.array([]))
                fom_rom = data['fom_rom_corrs']
                aux_rom = [data[f'aux{i}_rom_corrs'] for i in range(self.n_aux)]
            
            # Map data to indices
            pilot_data, names = {}, {}
            for i in range(self.n_aux):
                pilot_data[(i+1, 0)] = np.full(len(self.pilot_list), fom_aux[i])
                names[(i+1, 0)] = f"FOM-aux{i}"
            if self.n_aux > 1:
                idx = 0
                for i in range(1, self.n_aux):
                    for j in range(1, i):
                        pilot_data[(i, j)] = np.full(len(self.pilot_list), aux_aux[idx])
                        names[(i, j)] = f"aux{j-1}-aux{i-1}"
                        idx += 1
            pilot_data[(n-1, 0)] = fom_rom
            names[(n-1, 0)] = "FOM-ROM"
            for i in range(self.n_aux):
                pilot_data[(n-1, i+1)] = aux_rom[i]
                names[(n-1, i+1)] = f"aux{i}-ROM"
            
            # Create plots
            n_plots = n * (n - 1) // 2
            ncols = min(3, n_plots)
            nrows = (n_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
            axes = np.array([axes]).flatten() if n_plots == 1 else axes.flatten()
            
            plot_idx = 0
            for i in range(n):
                for j in range(i):
                    ax = axes[plot_idx]
                    nn_vals = corr_matrices[:, i, j]
                    ax.plot(s_grid, nn_vals, 'b-', label='Neural Net', linewidth=2, alpha=0.7)
                    
                    if (i, j) in pilot_data:
                        ax.plot(self.pilot_list, pilot_data[(i, j)], 
                               'ro', label='Pilot', markersize=10, zorder=5)
                        nn_at_pilots = nn_vals[np.isin(s_grid, self.pilot_list)]
                        if len(nn_at_pilots) == len(pilot_data[(i, j)]):
                            rmse = np.sqrt(np.mean((nn_at_pilots - pilot_data[(i, j)])**2))
                            ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}', transform=ax.transAxes,
                                   fontsize=9, va='top', bbox=dict(boxstyle='round', 
                                   facecolor='wheat', alpha=0.5))
                    
                    ax.set_title(f"{names.get((i, j), f'({i},{j})')}", fontsize=12, fontweight='bold')
                    ax.set_xlabel('ROM Basis Size')
                    ax.set_ylabel('Correlation')
                    ax.set_ylim([-1.05, 1.05])
                    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=9)
                    plot_idx += 1
            
            for idx in range(plot_idx, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, 'correlation_fits.png'), dpi=150, bbox_inches='tight')
            print(f"\nSaved plot to {debug_dir}/correlation_fits.png\n")
            plt.close()
        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")