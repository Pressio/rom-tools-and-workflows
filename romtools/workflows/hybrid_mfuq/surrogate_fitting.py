"""
Curve-fitting surrogates for Multi-Fidelity UQ.

Two fits live here, matching the two scalar surrogate families of the
writeup:

    fit_cost_polynomial   Sec. 4.3.1, normalized cost w_tilde_omega(s_omega)
    fit_sigmoid           Sec. 4.3.2, generalized sigmoid correlations

Both return a callable that accepts numpy scalars/arrays or torch tensors and
dispatches on the argument type, so the numpy and torch optimizer backends
share one fitted model rather than each fitting their own. Fitting itself is
always done in numpy (SciPy least squares); the returned closure holds the
fitted parameters as constants, so autograd flows through a tensor input
without the parameters needing to be torch objects.

Earlier revisions carried a parallel `*_torch` implementation of each fit that
optimized with LBFGS. That produced genuinely different parameters from the
SciPy fit, so `use_torch=True` and `use_torch=False` were running different
surrogates rather than the same surrogate two ways. Those variants are gone.
"""

import numpy as np
import torch
from scipy.optimize import least_squares


def _prepare_inputs(ins, outs, expect_dim=None):
    """
    Reshape `ins` to [dim, n_data].

    expect_dim=k forces the input dimension. expect_dim=None auto-detects by
    transposing only when that makes the array wider, which is correct
    whenever n_data > dim -- true for every fit here, since a tensor-product
    fit needs more data points than inputs. Pass expect_dim explicitly if a
    call site can present fewer data points than dimensions.

    `outs` is accepted for signature symmetry and returned unmodified.
    """
    ins = np.atleast_2d(ins)

    if expect_dim is None:
        if ins.shape[0] > ins.shape[1]:
            ins = ins.T
    elif ins.shape[0] != expect_dim:
        ins = ins.T

    dim, n_data = ins.shape
    return ins, dim, n_data


def _sigmoid_character(character):
    """
    Interpret the sigmoid `character` argument: returns (increasing, n_params).

    'increasing'/'decreasing' pin one asymptote and fit 4 parameters; [] fits
    the full 5-parameter generalized sigmoid of Sec. 4.3.2, which is what
    every call site currently uses.
    """
    if character == 'increasing':
        return True, 4
    if character == 'decreasing':
        return False, 4
    if character == []:
        return None, 5

    raise ValueError(
        'Invalid character. Options are "increasing", "decreasing", or [].'
    )


def fit_cost_polynomial(s_grid, costs, order=1):
    """
    Fit a single 1-D polynomial w_hat(s) = sum_a c_a s^a by linear least
    squares, per writeup Sec. 4.3.1.

    Deliberately not a tensor product: Sec. 4.3.1 gives each trainable model a
    cost that depends only on its own basis size, so there is no cross-model
    coupling to represent and extra input dimensions would only add degrees of
    freedom the model does not have. Callers fit one of these per trainable
    ROM. Deliberately not an iterative solve either: the coefficients enter
    linearly, so the normal equations give them exactly.

    s_grid : (G,) pilot basis sizes B_omega
    costs  : (G,) normalized pilot costs w_tilde_omega(s_omega,g)
    order  : polynomial order r (r = 1 in the reported experiments)
    """
    s = np.asarray(s_grid, dtype=float).reshape(-1)
    w = np.asarray(costs, dtype=float).reshape(-1)

    if s.size != w.size:
        raise ValueError(
            f"fit_cost_polynomial: {s.size} basis sizes vs {w.size} cost values"
        )

    order = int(min(order, max(s.size - 1, 0)))

    coeffs, *_ = np.linalg.lstsq(
        np.vander(s, N=order + 1, increasing=True), w, rcond=None
    )

    def evaluate(x):
        if torch.is_tensor(x):
            c = torch.as_tensor(coeffs, dtype=torch.float64, device=x.device)
            x_t = x.to(dtype=torch.float64)

            result = torch.zeros_like(x_t)
            for a in range(order, -1, -1):
                result = result * x_t + c[a]

            return result

        x_np = np.asarray(x, dtype=float)

        result = np.zeros_like(x_np, dtype=float)
        for a in range(order, -1, -1):
            result = result * x_np + coeffs[a]

        return result if result.ndim else float(result)

    evaluate.coeffs = coeffs
    evaluate.order = order

    return evaluate


def _sigmoid(params, x, n_params, increasing, backend):
    """
    Generalized sigmoid of Sec. 4.3.2, evaluated with `backend` (np or torch).

    The 5-parameter form is the one the writeup specifies:
        sigma(s; A, K, B, nu, Q) = A + (K - A) / (1 + Q exp(-B s))^(1/nu)
    """
    if n_params == 4:
        A, B, log_nu, log_Q = params
        nu = backend.exp(log_nu)
        Q = backend.exp(log_Q)
        K = int(not increasing)
        return A + (K - A) / (1 + Q * backend.exp(x - B)) ** (1 / nu)

    A, K, B, nu, Q = params
    return A + (K - A) / (1 + Q * backend.exp(-B * x)) ** (1 / nu)


def fit_sigmoid(ins, outs, character=[], expect_dim=None):
    """
    Fit a tensor product of generalized sigmoids to correlation data
    (writeup Sec. 4.3.2), by nonlinear least squares.

    dim = 1 gives the fixed-trainable surrogate p_hat_{i,omega}(s_omega).
    dim = 2 gives the trainable-trainable tensor product
    p_hat_{omega,q}(s_omega, s_q) = sigma_omega(s_omega) sigma_q(s_q).

    ins:  [dim, n_data] or [dim]
    outs: [n_data] or scalar
    character: 'increasing', 'decreasing', or [] for the general sigmoid
    expect_dim: force an input dimension (see _prepare_inputs)

    The returned callable accepts numpy or torch input and preserves
    gradients w.r.t. a tensor argument.
    """
    ins, dim, n_data = _prepare_inputs(ins, outs, expect_dim=expect_dim)
    increasing, n_params = _sigmoid_character(character)

    def evaluate_with(params, x, backend):
        params = params.reshape(dim, n_params)

        result = None
        for d in range(dim):
            term = _sigmoid(params[d], x[d], n_params, increasing, backend)
            result = term if result is None else result * term

        return result

    def residuals(flat_params):
        return evaluate_with(flat_params, ins, np) - outs

    result = least_squares(
        residuals, np.full((dim * n_params,), 0.5), loss='linear'
    )
    params_opt = result.x

    def evaluate(x):
        if torch.is_tensor(x):
            x_t = torch.atleast_2d(x.to(dtype=torch.float64))
            if x_t.shape[0] != dim:
                x_t = x_t.T

            params = torch.as_tensor(
                params_opt, dtype=torch.float64, device=x_t.device
            )
            values = evaluate_with(params, x_t, torch)

            return values.squeeze() if x_t.shape[1] == 1 else values

        x_np = np.atleast_2d(np.asarray(x, dtype=float))
        if x_np.shape[0] != dim:
            x_np = x_np.T

        values = evaluate_with(params_opt, x_np, np)

        return values[0] if x_np.shape[1] == 1 else values

    evaluate.params = params_opt
    evaluate.dim = dim

    return evaluate
