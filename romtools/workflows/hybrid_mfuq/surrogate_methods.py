import numpy as np
from scipy.optimize import least_squares


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# original routines used for surrogate fit
# ─────────────────────────────────────────────────────────────────────────────

def fit_polynomial(ins, outs, order=5):
    '''
    Fit a tensor product of 1-D polynomials to data.
    ins: [dim, n_data] or [dim] — input variables
    outs: [n_data] or scalar — target values
    '''
    ins = np.atleast_2d(ins)
    if ins.shape[0] > ins.shape[1]:
        ins = ins.T  # Ensure shape is (dim, n_data)
    dim, n_data = ins.shape

    def evaluate_tensor_product(coeffs, x):
        '''
        Evaluate tensor product polynomial on input x.
        coeffs: flat array [dim * (order+1)]
        x: [dim, n_data] or [dim]
        Returns: [n_data] or scalar
        '''
        coeffs = coeffs.reshape(dim, order + 1)
        x = np.atleast_2d(x)
        if x.shape[0] != dim:
            x = x.T  # ensure (dim, n_data)
        single_input = False
        if x.shape[1] == 1:
            single_input = x.ndim == 2 and x.shape[1] == 1 and x.shape[0] == dim

        result = np.ones(x.shape[1])
        for d in range(dim):
            powers = np.vander(x[d], N=order+1, increasing=True)  # [n_data, order+1]
            poly_vals = powers @ coeffs[d]  # [n_data]
            result *= poly_vals

        if single_input:
            return result[0]
        return result

    def residuals(coeffs):
        return evaluate_tensor_product(coeffs, ins) - outs

    x0 = np.full((dim * (order + 1),), 0.5)  # initial guess
    result = least_squares(residuals, x0, loss='linear')

    def fitted(x):
        '''
        Evaluate fitted polynomial.
        x: [dim] or [dim, n_data]
        Returns: scalar or [n_data]
        '''
        return evaluate_tensor_product(result.x, x)

    return fitted


# r is always a vector of one number per model
# s is currently assumed a vector of one number per model
# could use tensor product of sigmoids for multiple s per model?
def fit_sigmoid(ins, outs, character=[]):
    '''
    Fit a tensor product of 3- or 5-parameter sigmoids to data.
    
    ins: shape [dim, n_data] or [dim]
    outs: shape [n_data] or scalar
    character: 'increasing', 'decreasing', or [] (for general sigmoid)
    
    Returns:
        fitted(x): callable that accepts x of shape (dim,) or (dim, n_data)
    '''
    ins = np.atleast_2d(ins)
    if ins.shape[0] > ins.shape[1]:
        ins = ins.T  # ensure shape (dim, n_data)
    dim, n_data = ins.shape

    # Define sigmoid type
    if character == 'increasing':
        opt = True
        num_vars = 4
    elif character == 'decreasing':
        opt = False
        num_vars = 4
    elif character == []:
        opt = None  # general
        num_vars = 5
    else:
        raise ValueError('Invalid character. Options are "increasing", "decreasing", or [].')

    def sigmoid(params, x):
        '''
        Vectorized evaluation of sigmoid function over x.
        x: shape [n_data]
        '''
        if num_vars == 4:
            A, B, log_nu, log_Q = params
            nu = np.exp(log_nu)
            Q = np.exp(log_Q)
            # A = int(opt)
            K = int(not opt)
            return A + (K - A) / (1 + Q * np.exp((x-B)))**(1 / nu)
        else:
            A, K, B, nu, Q = params
            return A + (K - A) / (1 + Q * np.exp(-B * x))**(1 / nu)

    def evaluate_tensor_product(params, x):
        '''
        Evaluate tensor product of sigmoids over input x.
        x: shape [dim] or [dim, n_data]
        params: flat array of sigmoid parameters
        '''
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
    fitted_params = result.x

    def fitted(x):
        return evaluate_tensor_product(fitted_params, x)

    return fitted



# ─────────────────────────────────────────────────────────────────────────────
# un-batched 4 routines, no longer used
# ─────────────────────────────────────────────────────────────────────────────

def to_symmetric_tracefree(lower_vec, n):
    L = torch.zeros((n, n), dtype=lower_vec.dtype, device=lower_vec.device)
    tril = torch.tril_indices(n, n, offset=-1)
    L[tril[0], tril[1]] = lower_vec
    return L + L.T

def fixed_point_step(A, x):
    A_new = A + torch.diag(x)
    C_new = torch.linalg.matrix_exp(A_new)
    return x - torch.log(torch.diag(C_new))

def fixed_point_solve(f, x0, tol=1e-6, max_iter=100):
    x = x0.clone().detach()
    for _ in range(max_iter):
        x_next = f(x).detach()
        if torch.norm(x_next - x) < tol:
            break
        x = x_next
    return x

def to_unique_corr_matrix(vecl, n):
    A    = to_symmetric_tracefree(vecl, n)
    func = lambda x: fixed_point_step(A, x)
    x0   = torch.randn(n, device=vecl.device, dtype=vecl.dtype)
    d    = fixed_point_solve(func, x0, tol=1e-9, max_iter=100)
    return torch.linalg.matrix_exp(A + torch.diag(d))


# ─────────────────────────────────────────────────────────────────────────────
# new routines batched over inputs
# ─────────────────────────────────────────────────────────────────────────────

# def to_symmetric_tracefree_batch(lower_vecs, n):
#     B, m = lower_vecs.shape
#     assert m == n*(n-1)//2

#     # make sure everything is on the same device/dtype
#     device, dtype = lower_vecs.device, lower_vecs.dtype

#     # flattened index into an (n*n)-long vector
#     rows, cols = torch.tril_indices(n, n, offset=-1, device=device)
#     idx = rows * n + cols        # shape (m,)

#     # create empty flat matrix (B, n*n)
#     flat = torch.zeros(B, n*n, device=device, dtype=dtype)

#     # scatter the lower-triangle entries into flat
#     # scatter_ is out-of-place in terms of grad, but in-place
#     # on the flat storage—no device mismatch.
#     flat.scatter_(1,
#                   idx.unsqueeze(0).expand(B, -1),
#                   lower_vecs)

#     # reshape to (B,n,n) and symmetrize
#     L = flat.view(B, n, n)
#     return L + L.transpose(1, 2)


def to_symmetric_tracefree_batch(lower_vecs, n):  # in-place, might not propagate gradients
    """
    lower_vecs: (B, m)
    returns:    (B, n, n) symmetric, zero‐trace
    """
    device = lower_vecs.device  # make sure everything is on same device
    B_, m_ = lower_vecs.shape
    assert m_ == n*(n-1)//2
    L = lower_vecs.new_zeros((B_, n, n))
    tril_idx = torch.tril_indices(n, n, offset=-1, device=device)
    L[:, tril_idx[0], tril_idx[1]] = lower_vecs
    return L + L.transpose(1, 2)     # (B,n,n)


def fixed_point_solve_batch(A_batch, tol=1e-6, max_iter=100):
    """
    A_batch: (B,n,n)
    returns: x* of shape (B,n)
    solves x = x - log diag exp(A + diag(x)) in batch
    """
    B_, n_, _ = A_batch.shape
    # initialize x
    x = torch.randn(B_, n_, device=A_batch.device, dtype=A_batch.dtype).detach()
    for _ in range(max_iter):
        A_new  = A_batch + torch.diag_embed(x)          # (B,n,n)
        C_new  = torch.linalg.matrix_exp(A_new)         # (B,n,n)
        diagC  = C_new.diagonal(dim1=-2, dim2=-1)       # (B,n)
        x_next = (x - torch.log(diagC)).detach()        # no grads through solver (IFT TODO)
        if torch.norm(x_next - x) < tol:                # stopping based on norm of batch OK?
            x = x_next
            break
        x = x_next
    return x  # (B,n)


def to_unique_corr_matrix_batch(lower_vecs, n):
    """
    lower_vecs: (B, m)
    returns:    (B, n, n) valid correlation matrices
    """
    A_batch = to_symmetric_tracefree_batch(lower_vecs, n)     # (B,n,n)
    d_star  = fixed_point_solve_batch(A_batch, tol=1e-9, max_iter=100)  # (B,n)
    M       = A_batch + torch.diag_embed(d_star)              # (B,n,n)
    return torch.linalg.matrix_exp(M)                         # (B,n,n)


# ─────────────────────────────────────────────────────────────────────────────
# shallow network class that produces log-transformed correlations 
# ─────────────────────────────────────────────────────────────────────────────

class VeclNet(nn.Module):
    """
    dim:    input latent dimension
    hidden: hidden‐layer size
    m:      output vector size (# unique off-diag entries)
    """
    def __init__(self, dim, hidden, m):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, m)
        )

    def forward(self, z):
        """
        z:       (B, dim)
        returns: (B, m)
        """
        return self.net(z)

    def corr_matrix(self, z, n):
        """
        z:       (B, dim)
        n:       scalar matrix size
        returns: (B, n, n) valid correlation matrices via
                 to_unique_corr_matrix_batch
        """
        return to_unique_corr_matrix_batch(self.net(z), n)  # (B,n,n)


# ─────────────────────────────────────────────────────────────────────────────
# training function which takes inputs/targets and produces 
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    inputs,
    targets,
    n,
    lr=1e-2,
    max_steps=500,
    tol=1e-6,
    print_every=50,
    optimizer_cls=optim.Adam,
    optimizer_kwargs=None,
    random_seed=torch.manual_seed(2025)
):
    """
    Train `model` so that model(inputs) -> vecl_batch -> to_unique_corr_matrix_batch(vecl_batch, n)
    approximates `targets` correlations.

    Args:
      model            -- an nn.Module mapping inputs to a batch of length-m vectors
      inputs           -- Tensor of shape (B, dim), the latent inputs Z
      targets          -- Tensor of shape (B, n, n), the target correlation matrices
      n                -- int, matrix‐size parameter for to_unique_corr_matrix_batch
      lr               -- learning rate passed to the optimizer
      max_steps        -- maximum number of training iterations
      tol              -- tolerance on absolute change in loss for early stopping
      print_every      -- how often to print status (in steps)
      optimizer_cls    -- optimizer class (default: torch.optim.Adam)
      optimizer_kwargs -- dict of extra args to optimizer constructor
      random_seed      -- seed for reproducibility

    Returns:
      model            -- the trained model (in‐place)
      loss_history     -- list of loss floats, one per step
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    prev_loss = None
    loss_history = []

    targets.to(inputs.device)

    for step in range(1, max_steps+1):
        optimizer.zero_grad()

        vecl_batch = model(inputs)                                # (B, m)
        C_batch    = to_unique_corr_matrix_batch(vecl_batch, n)   # (B, n, n)

        loss = F.mse_loss(C_batch, targets)   # using full matrices (not necessary)
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



if __name__ == "__main__":
    torch.manual_seed(0)

    n             = 4
    m             = n*(n-1)//2   # number of unique off‐diag entries
    B             = 10           # batch size
    dim           = 2            # latent dimension for z
    hidden        = 16
    model         = VeclNet(dim, hidden, m)

    # latent inputs
    Z = torch.randn(B, dim)

    # helper to build random SPD→corr matrices
    # def random_corr(n):
    #     G = torch.randn(n, n)
    #     S = G @ G.T
    #     d = torch.sqrt(torch.diag(S))
    #     return S / (d.unsqueeze(0)*d.unsqueeze(1) + 1e-9)
    def random_corr(n):
        vecl = torch.randn(n*(n-1)//2).unsqueeze(0)
        return to_unique_corr_matrix_batch(vecl, n).squeeze(0)

    # batch of targets
    targets = torch.stack([random_corr(n) for _ in range(B)], dim=0)  # (B,n,n)

    trained_model, history = train_model(
        model,
        inputs=Z,
        targets=targets,
        n=n,
        lr=1e-1,
        max_steps=1000,
        tol=1e-8,
        print_every=50,
    )

    C_preds = trained_model.corr_matrix(Z, n)
    idx = 5
    print(torch.linalg.eigvals(C_preds[idx]))
    print(torch.linalg.eigvals(targets[idx]))
