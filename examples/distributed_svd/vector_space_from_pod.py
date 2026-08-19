"""
Construct a distributed VectorSpaceFromPOD with DistributedSvd.
"""

import numpy as np
from mpi4py import MPI

from romtools.linalg import DistributedSvd
from romtools.vector_space import VectorSpaceFromPOD
from romtools.vector_space.utils import BasisSizeTruncater


def main():
    """Build and verify a truncated, row-distributed POD vector space."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    number_of_variables = 1
    local_spatial_extent = rank + 2
    snapshot_count = 6
    rng = np.random.default_rng(2000 + rank)
    local_snapshots = rng.normal(
        size=(number_of_variables, local_spatial_extent, snapshot_count)
    )

    global_spatial_extent = comm.allreduce(
        local_spatial_extent, op=MPI.SUM
    )
    basis_dimension = min(3, global_spatial_extent, snapshot_count)

    vector_space = VectorSpaceFromPOD(
        local_snapshots,
        truncater=BasisSizeTruncater(basis_dimension),
        svdFnc=DistributedSvd(comm),
    )

    local_basis_tensor = vector_space.get_basis()
    local_basis_matrix = local_basis_tensor.reshape(
        number_of_variables*local_spatial_extent,
        basis_dimension,
    )

    local_gram = local_basis_matrix.T @ local_basis_matrix
    global_gram = np.empty_like(local_gram)
    comm.Allreduce(local_gram, global_gram, op=MPI.SUM)
    orthonormality_error = np.linalg.norm(
        global_gram - np.eye(basis_dimension)
    )

    if rank == 0:
        print("VectorSpaceFromPOD with DistributedSvd example")
        print(f"basis dimension: {basis_dimension}")
        print(
            "singular values: "
            f"{vector_space.get_singular_values()}"
        )
        print(f"orthonormality error: {orthonormality_error:.3e}")


if __name__ == "__main__":
    main()
