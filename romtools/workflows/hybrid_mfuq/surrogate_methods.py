import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# original 4 routines, no longer used
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

def to_symmetric_tracefree_batch(lower_vecs, n):
    """
    lower_vecs: (B, m)
    returns:    (B, n, n) symmetric, zero‐trace
    """
    B_, m_ = lower_vecs.shape
    assert m_ == n*(n-1)//2
    L = lower_vecs.new_zeros((B_, n, n))
    tril_idx = torch.tril_indices(n, n, offset=-1)
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
