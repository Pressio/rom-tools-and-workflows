import numpy as np
import pytest

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.linalg import _local_column_range, _streaming_pod
from romtools.vector_space.utils.svd_method_of_snapshots import SvdMethodOfSnapshots


@pytest.mark.mpi(min_size=3)
def test_streaming_pod_row_distributed():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    snapshots = np.random.default_rng(33).normal(size=(5, 8))
    start_row, end_row = _local_column_range(rank, size, snapshots.shape[0])
    local_snapshots = snapshots[start_row:end_row]
    loader = lambda start, end: local_snapshots[:, start:end]

    np.random.seed(327)
    local_U, S, Vt, total_energy = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        n_snapshots=8,
        max_basis_dimension=5,
        svdFnc=SvdMethodOfSnapshots(comm),
        comm=comm,
    )

    gathered_U = comm.gather(local_U, root=0)
    if rank == 0:
        U = np.vstack(gathered_U)
        exact_U, exact_S, _ = np.linalg.svd(snapshots, full_matrices=False)
        assert U.shape == (5, 5)
        assert S.shape == (5,)
        assert Vt.shape == (5, 8)
        assert np.allclose(S, exact_S)
        assert np.isclose(total_energy, np.sum(snapshots**2))
        assert np.allclose(
            np.abs(U.T @ exact_U), np.eye(5), atol=1e-10
        )


@pytest.mark.mpi(min_size=3)
def test_streaming_pod_distributed_requires_svd_functor():
    comm = MPI.COMM_WORLD
    snapshots = np.ones((2, 4))
    loader = lambda start, end: snapshots[:, start:end]

    with pytest.raises(ValueError, match="svdFnc is required"):
        _streaming_pod(
            snapshot_loader=loader,
            block_size=2,
            n_snapshots=4,
            max_basis_dimension=2,
            comm=comm,
        )
