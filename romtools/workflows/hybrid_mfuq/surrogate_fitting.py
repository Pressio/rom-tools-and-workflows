"""
Curve-fitting surrogates for Multi-Fidelity UQ.

Tensor-product polynomial and sigmoid fits (numpy and PyTorch backends),
used by SurrogateBuilder for scalar cost/correlation surrogates.

Split out of surrogate_methods.py (see hybrid_mfuq_simplification_plan.md, T6).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from scipy.optimize import least_squares


def _prepare_inputs(ins, outs, expect_dim=None):
    """
    Reshape ins to [dim, n_data], the input-reshaping step shared by
    fit_polynomial and fit_sigmoid.

    expect_dim=None reproduces fit_polynomial's original heuristic:
    transpose only if that makes the array wider (shape[0] > shape[1]).
    expect_dim=1 reproduces fit_sigmoid's original heuristic: transpose
    unless the first axis already has length 1 (sigmoid fits are always
    1-D). outs is accepted but not modified here; numpy fits use it
    as-is in their residuals closures.
    """
    ins = np.atleast_2d(ins)

    if expect_dim is None:
        if ins.shape[0] > ins.shape[1]:
            ins = ins.T
    else:
        if ins.shape[0] != expect_dim:
            ins = ins.T

    dim, n_data = ins.shape
    return ins, dim, n_data


def fit_polynomial(ins, outs, order=5):
    '''
    Fit a tensor product of 1-D polynomials to data.
    ins: [dim, n_data] or [dim] — input variables
    outs: [n_data] or scalar — target values
    '''
    ins, dim, n_data = _prepare_inputs(ins, outs)

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


def _sigmoid_character(character):
    """
    Validate/interpret the sigmoid 'character' argument shared by
    fit_sigmoid and fit_sigmoid_torch. Purely a lookup, no numpy/torch
    dependence.
    """
    if character == 'increasing':
        return True, 4
    elif character == 'decreasing':
        return False, 4
    elif character == []:
        return None, 5
    else:
        raise ValueError('Invalid character. Options are "increasing", "decreasing", or [].')


def fit_sigmoid(ins, outs, character=[]):
    '''
    Fit a tensor product of 3- or 5-parameter sigmoids to data.
    
    ins: shape [dim, n_data] or [dim]
    outs: shape [n_data] or scalar
    character: 'increasing', 'decreasing', or [] (for general sigmoid)
    '''
    ins, dim, n_data = _prepare_inputs(ins, outs, expect_dim=1)
    opt, num_vars = _sigmoid_character(character)

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


def _prepare_inputs_torch(ins, outs, expect_dim=None):
    """
    Torch-tensor analogue of _prepare_inputs: converts ins/outs to
    float64 tensors and reshapes ins to [dim, n_data] using the same
    transpose-detection heuristic (expect_dim=None for the polynomial
    fits' shape[0] > shape[1] check, expect_dim=1 for the sigmoid fits'
    always-dim-1 check).
    """
    ins = torch.atleast_2d(torch.as_tensor(ins, dtype=torch.float64))
    outs = torch.as_tensor(outs, dtype=torch.float64)

    if expect_dim is None:
        if ins.shape[0] > ins.shape[1]:
            ins = ins.T
    else:
        if ins.shape[0] != expect_dim:
            ins = ins.T

    dim, n_data = ins.shape
    return ins, outs, dim, n_data


def fit_polynomial_torch(ins, outs, order=5):
    """
    Fit a tensor product of 1-D polynomials to data (PyTorch version).
    Returns a callable that preserves gradients.
    
    ins: [dim, n_data] or [dim] — input variables
    outs: [n_data] or scalar — target values
    """
    ins, outs, dim, n_data = _prepare_inputs_torch(ins, outs)
    
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
    ins, outs, dim, n_data = _prepare_inputs_torch(ins, outs, expect_dim=1)
    opt, num_vars = _sigmoid_character(character)
    
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

