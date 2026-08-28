import pytest
import numpy as np
import romtools as rt
from helper_scripts import helpers
import romtools.vector_space.utils as utils
from romtools.linalg.linalg import _local_column_range
from romtools.vector_space.utils.svd_method_of_snapshots import SvdMethodOfSnapshots
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


def construct_snapshots(comm):
    rank = comm.Get_rank()
    if rank == 0:
        myData = np.random.normal(size=(3, 3, 5))
    elif rank==1:
        myData = np.random.normal(size=(3, 5, 5))
    else:
        myData = np.random.normal(size=(3, 3, 5))

    return myData

class MyFakeSvd:
    def __init__(self, comm):
        self.comm_ = comm

    def __call__(self, A, full_matrices=False, compute_uv=True, hermitian=False):
        # this is totally fake, just for testing

        rank = self.comm_.Get_rank()
        nr, nc = A.shape[0], A.shape[1]
        lsv = np.ones((nr, 2))*rank
        svals = np.ones(2)*rank
        return lsv, svals, None


@pytest.mark.mpi(min_size=3)
def test_vector_space_from_pod_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() == 3:
        snaps = construct_snapshots(comm)
        myVectorSpace = rt.VectorSpaceFromPOD(snaps, svdFnc=MyFakeSvd(comm))
        U = myVectorSpace.get_basis()
        k = myVectorSpace.extents()[-1]
        if rank == 0:
            assert np.allclose(U, np.zeros((3, 3, 2)))
            assert np.allclose(2, k)
        elif rank == 1:
            assert np.allclose(U, np.ones((3, 5, 2)))
            assert np.allclose(2, k)
        elif rank == 2:
            assert np.allclose(U, np.ones((3, 3, 2))*2)
            assert np.allclose(2, k)
    else:
        helpers.mpi_skipped_test_mismatching_commsize(comm, "test_vector_space_from_pod_mpi", 3)


@pytest.mark.mpi(min_size=3)
def test_streaming_vector_space_row_distributed_shifting_and_scaling():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    snapshots = np.random.default_rng(52).normal(size=(2, 6, 5))
    snapshots[..., 3:] *= 10.0
    start, end = _local_column_range(rank, size, snapshots.shape[1])
    local_snapshots = snapshots[:, start:end].copy()
    original_local_snapshots = local_snapshots.copy()
    loader = lambda first, last: local_snapshots[..., first:last]
    shifter = utils.create_streaming_average_shifter()
    scaler = utils.VariableScaler("variance")

    vector_space = rt.VectorSpaceFromStreamingPOD(
        snapshot_loader=loader,
        block_size=2,
        n_snapshots=5,
        max_basis_dimension=4,
        truncater=utils.BasisSizeTruncater(3),
        shifter=shifter,
        scaler=scaler,
        svdFnc=SvdMethodOfSnapshots(comm),
        comm=comm,
    )

    shift_vector = np.mean(snapshots, axis=2)
    shifted_snapshots = snapshots - shift_vector[..., None]
    expected_scales = np.std(shifted_snapshots, axis=(1, 2))
    assert np.allclose(scaler.var_scales_, expected_scales)
    assert np.allclose(
        vector_space.get_shift_vector(), shift_vector[:, start:end]
    )
    assert np.array_equal(local_snapshots, original_local_snapshots)
    assert vector_space.extents() == (2, end - start, 3)

    gathered_basis = comm.gather(
        (start, end, vector_space.get_basis()), root=0
    )
    if rank == 0:
        global_basis = np.empty((2, 6, 3))
        for local_start, local_end, local_basis in gathered_basis:
            global_basis[:, local_start:local_end] = local_basis

        in_memory_space = rt.VectorSpaceFromPOD(
            snapshots=snapshots.copy(),
            truncater=utils.BasisSizeTruncater(3),
            shifter=utils.create_average_shifter(snapshots),
            scaler=utils.VariableScaler("variance"),
        )
        expected_basis = in_memory_space.get_basis()
        for mode in range(3):
            correlation = abs(
                global_basis[..., mode].ravel()
                @ expected_basis[..., mode].ravel()
            )
            correlation /= (
                np.linalg.norm(global_basis[..., mode])
                * np.linalg.norm(expected_basis[..., mode])
            )
            assert np.isclose(correlation, 1.0, atol=1e-10)
        assert np.allclose(
            vector_space.get_singular_values(),
            in_memory_space.get_singular_values()[:4],
        )

if __name__ == "__main__":
    test_vector_space_from_pod_mpi()
