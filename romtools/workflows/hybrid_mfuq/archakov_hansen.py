"""
Batched Archakov--Hansen correlation-matrix machinery for Multi-Fidelity UQ.

Provides the differentiable, warm-started AH map from unrestricted
lower-triangular coordinates to valid correlation matrices, the VeclNet
network that produces those coordinates, the training loop for the
matrix-valued surrogate, and the resulting corr_matrix_fn wrapper.

Split out of surrogate_methods.py (see hybrid_mfuq_simplification_plan.md, T6).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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



