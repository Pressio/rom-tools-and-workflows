import numpy as np
import pytest
from mpi4py import MPI

from romtools.linalg import DistributedSvd
from romtools.vector_space import VectorSpaceFromPOD
from romtools.vector_space.utils import BasisSizeTruncater, NoOpTruncater


def _uneven_snapshot_partition(global_snapshots, comm):
    """Distribute the snapshot tensor's spatial axis nearly evenly by rank.

    ``global_snapshots`` has shape ``(n_variables, spatial_extent, n_samples)``.
    ``VectorSpaceFromPOD`` later collapses the first two axes into matrix rows;
    this test uses one variable, so partitioning spatial indices here is exactly
    equivalent to partitioning the resulting snapshot matrix by rows.

    As in the matrix helper, each rank receives ``spatial_extent // comm_size``
    entries and the first ``spatial_extent % comm_size`` ranks receive one
    extra. The returned global slice is used to select the matching rows from
    the serial NumPy POD basis.
    """
    spatial_extent = global_snapshots.shape[1]
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    # Compute the number of spatial entries owned by every rank.
    counts = np.full(comm_size, spatial_extent // comm_size, dtype=int)
    counts[:spatial_extent % comm_size] += 1

    # Turn the counts into global spatial-axis boundaries. For example, counts
    # [4, 3, 3] produce offsets [0, 4, 7, 10].
    offsets = np.concatenate(([0], np.cumsum(counts)))
    local_slice = slice(offsets[rank], offsets[rank + 1])

    # Keep every variable and sample, selecting only this rank's spatial range.
    # The copy also lets the integration test detect accidental input mutation.
    return global_snapshots[:, local_slice, :].copy(), local_slice, counts


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("basis_dimension", [None, 3])
def test_vector_space_from_pod_with_distributed_svd_matches_numpy(
        basis_dimension):
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(424242)

    # For the CI sizes of 3 and 4 ranks, this produces uneven row distribution.
    spatial_extent = 2*comm.Get_size() + 7
    snapshot_count = 6

    # A single variable makes VectorSpaceFromPOD's tensor-to-matrix operation
    # exactly the same row distribution used by the reference matrix below.
    global_snapshots = rng.normal(
        size=(1, spatial_extent, snapshot_count)
    )
    # Extract the spacial rows owned by this rank.
    local_snapshots, local_slice, counts = _uneven_snapshot_partition(
        global_snapshots, comm
    )
    local_snapshots_before = local_snapshots.copy()

    if comm.Get_size() > 1:
        assert np.unique(counts).size > 1

    # Select the POD truncation policy.
    truncater = (
        NoOpTruncater()
        if basis_dimension is None
        else BasisSizeTruncater(basis_dimension)
    )
    vector_space = VectorSpaceFromPOD(
        local_snapshots,
        truncater=truncater,
        svdFnc=DistributedSvd(comm),
    )

    # Compute the trusted serial NumPy reference SVD.
    expected_u, expected_s, _ = np.linalg.svd(
        global_snapshots.reshape(spatial_extent, snapshot_count),
        full_matrices=False,
        compute_uv=True,
        hermitian=False,
    )
    expected_dimension = (
        expected_s.size
        if basis_dimension is None
        else basis_dimension
    )
    actual_local_basis = vector_space.get_basis()[0]

    # Check the shape of the distributed POD basis.
    assert vector_space.extents() == (
        1, local_snapshots.shape[1], expected_dimension
    )
    np.testing.assert_allclose(
        vector_space.get_singular_values(),
        expected_s,
        rtol=2e-12,
        atol=2e-12,
    )

    # Compare the POD spaces through their principal angles. This is invariant
    # to signs and to rotations within repeated-singular-value subspaces.
    expected_local_basis = expected_u[local_slice, :expected_dimension]
    local_overlap = expected_local_basis.T @ actual_local_basis
    global_overlap = np.empty_like(local_overlap)
    comm.Allreduce(local_overlap, global_overlap, op=MPI.SUM)
    principal_cosines = np.linalg.svd(global_overlap, compute_uv=False)
    np.testing.assert_allclose(
        principal_cosines,
        np.ones(expected_dimension),
        rtol=2e-12,
        atol=2e-12,
    )

    # Check global orthonormality of the distributed basis.
    local_gram = actual_local_basis.T @ actual_local_basis
    global_gram = np.empty_like(local_gram)
    comm.Allreduce(local_gram, global_gram, op=MPI.SUM)
    np.testing.assert_allclose(
        global_gram,
        np.eye(expected_dimension),
        rtol=2e-12,
        atol=2e-12,
    )

    # Check that POD construction did not modify the input snapshots.
    np.testing.assert_array_equal(local_snapshots, local_snapshots_before)
