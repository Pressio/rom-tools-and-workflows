"""
Compute a POD basis by calling DistributedSvd directly.
"""

import numpy as np
from mpi4py import MPI

from romtools.linalg import DistributedSvd


def main():
    """Compute and verify the thin SVD of a row-distributed matrix."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_row_count = rank + 2
    snapshot_count = 6
    rng = np.random.default_rng(1000 + rank)
    local_snapshot_matrix = rng.normal(
        size=(local_row_count, snapshot_count)
    )

    distributed_svd = DistributedSvd(comm)
    U_local, singular_values, Vh = distributed_svd(
        local_snapshot_matrix,
        full_matrices=False,
        compute_uv=True,
        hermitian=False,
    )

    local_pod_basis = U_local

    local_reconstruction = (U_local*singular_values) @ Vh
    local_error_squared = np.linalg.norm(
        local_reconstruction - local_snapshot_matrix
    )**2
    local_norm_squared = np.linalg.norm(local_snapshot_matrix)**2
    global_error_squared = comm.allreduce(local_error_squared, op=MPI.SUM)
    global_norm_squared = comm.allreduce(local_norm_squared, op=MPI.SUM)
    relative_reconstruction_error = np.sqrt(
        global_error_squared/global_norm_squared
    )

    local_gram = local_pod_basis.T @ local_pod_basis
    global_gram = np.empty_like(local_gram)
    comm.Allreduce(local_gram, global_gram, op=MPI.SUM)
    orthonormality_error = np.linalg.norm(
        global_gram - np.eye(singular_values.size)
    )

    if rank == 0:
        print("DistributedSvd example")
        print(f"singular values: {singular_values}")
        print(
            "relative reconstruction error: "
            f"{relative_reconstruction_error:.3e}"
        )
        print(f"orthonormality error: {orthonormality_error:.3e}")


if __name__ == "__main__":
    main()
