import numpy as np
import pytest

from mpi4py import MPI

from romtools.linalg import DistributedSvd


def _matrix_for_case(case, comm_size):
    """Build a deterministic global matrix for one numerical test scenario.

    ``comm_size`` is included in the dimensions so the same test remains
    meaningful at each CI communicator size.

    The returned matrix is global reference data: every rank constructs the
    same matrix from the same random seed.
    """
    # A fixed seed makes every MPI rank independently construct identical
    # reference data; no broadcast of the global test matrix is required.
    rng = np.random.default_rng(8675309)

    if case == "tall-real":
        # More rows than columns, using NumPy's default float64 dtype.
        return rng.normal(size=(2*comm_size + 7, 5))
    if case == "wide-real":
        # More columns than rows exercises the general reduced-QR path.
        return rng.normal(size=(comm_size + 2, 2*comm_size + 9))
    if case == "tall-complex":
        # Build complex128 data from independent real and imaginary parts.
        shape = (2*comm_size + 7, 4)
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)
    if case == "wide-complex64":
        # Exercise both a wide shape and NumPy's lower-precision complex dtype.
        shape = (comm_size + 2, 2*comm_size + 7)
        return (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(
            np.complex64
        )
    if case == "rank-deficient":
        # The matrix product has rank at most three. The appended column is a
        # linear combination of two existing columns, so it adds no rank.
        row_count = 2*comm_size + 7
        core = rng.normal(size=(row_count, 3)) @ rng.normal(size=(3, 6))
        return np.column_stack((core, core[:, 0] - 3.0*core[:, 2]))
    if case == "repeated-singular-values":
        # Orthonormal left/right factors and an explicitly chosen spectrum make
        # the first two singular values exactly equal. Their individual
        # singular vectors are nonunique, but their two-dimensional subspace is.
        row_count = 2*comm_size + 7
        column_count = 7
        left, _ = np.linalg.qr(rng.normal(size=(row_count, 4)))
        right, _ = np.linalg.qr(rng.normal(size=(column_count, 4)))
        return (left*np.array([9.0, 9.0, 2.5, 0.75])) @ right.T
    raise ValueError(f"unknown matrix case {case}")


def _uneven_partition(global_matrix, comm):
    """Distribute contiguous matrix rows nearly evenly across MPI ranks.

    The number of rows in ``global_matrix`` is divided by the number of MPI
    ranks. Every rank receives at least
    ``global_matrix.shape[0] // comm.Get_size()`` rows. If the number of rows is
    not divisible by the number of ranks, the remaining rows are assigned one at
    a time to the lowest-numbered ranks. Consequently, the local row counts
    differ by at most one.

    Args:
        global_matrix (np.ndarray):
            Complete two-dimensional matrix to partition. Every MPI rank is
            expected to hold the same matrix.
        comm (MPI.Comm):
            MPI communicator whose ranks will own the row partitions.

    Returns:
        tuple:
            A tuple ``(local_matrix, local_slice, counts)`` containing:

            * ``local_matrix``: A copy of the contiguous rows assigned to the
              current rank.
            * ``local_slice``: A slice locating those rows in
              ``global_matrix``.
            * ``counts``: An integer array containing the number of rows
              assigned to every MPI rank.

    Example:
        The following example distributes an 11-row matrix across three ranks:

        .. code-block:: python

            import numpy as np
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            global_matrix = np.arange(22).reshape(11, 2)

            local_matrix, local_slice, counts = _uneven_partition(
                global_matrix,
                comm,
            )

            print(
                f"rank={rank}, "
                f"slice={local_slice}, "
                f"local_shape={local_matrix.shape}"
            )

        Run the example with:

        .. code-block:: console

            mpiexec -n 3 python example.py

        The 11 rows are distributed as follows:

        .. code-block:: text

            counts  = [4, 4, 3]
            offsets = [0, 4, 8, 11]

            rank 0 owns global_matrix[0:4]
            rank 1 owns global_matrix[4:8]
            rank 2 owns global_matrix[8:11]

        Rank 0 and rank 1 receive four rows each, while rank 2 receives three
        rows. The difference between any two local row counts is at most one.

    Notes:
        ``local_matrix`` is returned as a copy. This prevents it from sharing
        memory with ``global_matrix`` and allows tests to detect whether
        ``DistributedSvd`` accidentally modifies its input.

        ``local_slice`` is used to compare a rank's distributed result with the
        corresponding rows of a serial NumPy result.
    """
    row_count = global_matrix.shape[0]
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    counts = np.full(comm_size, row_count // comm_size, dtype=int)

    # Distribute the remainder one row at a time to the lowest-numbered ranks.
    counts[:row_count % comm_size] += 1

    offsets = np.concatenate(([0], np.cumsum(counts)))

    # Rank r owns offsets[r]:offsets[r + 1].
    local_slice = slice(offsets[rank], offsets[rank + 1])

    # Copying makes input-preservation assertions meaningful: a mutation by
    # DistributedSvd cannot be hidden by aliasing the global reference matrix.
    return global_matrix[local_slice].copy(), local_slice, counts


def _positive_singular_value_clusters(singular_values, matrix_shape):
    """Group numerically positive, equal singular values into index slices.

    Singular vectors are not uniquely determined. A real singular vector may
    change sign, a complex singular vector may change phase, and singular
    vectors associated with a repeated singular value may be rotated within
    their shared invariant subspace.

    Numerically zero singular values are excluded because different valid SVD
    algorithms may produce different bases for the corresponding null space.

    Args:
        singular_values: Singular values in descending order.
        matrix_shape: Shape of the matrix from which the singular values were
            computed. It is used to calculate a scale-dependent numerical-zero
            tolerance.

    Returns:
        A list of slices. Each slice identifies a group of numerically equal,
        positive singular values.

    Example:
        Consider a rank-deficient matrix whose singular values are:

        .. code-block:: python

            singular_values = np.array([9.0, 9.0, 2.5, 0.75, 0.0])
            matrix_shape = (20, 7)

            clusters = _positive_singular_value_clusters(
                singular_values,
                matrix_shape,
            )

        The resulting clusters are equivalent to:

        .. code-block:: python

            clusters = [
                slice(0, 2),  # Repeated singular values [9.0, 9.0]
                slice(2, 3),  # Singular value [2.5]
                slice(3, 4),  # Singular value [0.75]
            ]

        The singular values in each cluster can be inspected with:

        .. code-block:: python

            for cluster in clusters:
                print(singular_values[cluster])

        which produces:

        .. code-block:: text

            [9. 9.]
            [2.5]
            [0.75]

        The zero singular value is not included. The first two singular values
        are grouped because their individual singular vectors are not unique,
        although the two-dimensional subspace spanned by those vectors is
        unique.

    Notes:
        Singular values that differ only by floating-point roundoff are treated
        as equal. Values below the scale-dependent tolerance are treated as
        numerically zero.
    """
    if singular_values.size == 0:
        return []

    # This is a conventional scale-aware numerical-rank threshold: machine
    # precision multiplied by problem size and the largest singular value. The
    # factor 100 gives the QR+SVD sequence some floating-point margin.
    tolerance = (100.0*np.finfo(singular_values.dtype).eps
                 * max(matrix_shape)*singular_values[0])
    positive_count = np.count_nonzero(singular_values > tolerance)
    clusters = []
    cluster_start = 0

    # NumPy returns singular values in descending order, so equal values occur
    # next to one another and can be represented by ordinary Python slices.
    while cluster_start < positive_count:
        cluster_end = cluster_start + 1
        while (cluster_end < positive_count
               and np.isclose(
                   singular_values[cluster_end],
                   singular_values[cluster_start],
                   rtol=1e-7,
                   atol=tolerance,
               )):
            cluster_end += 1
        clusters.append(slice(cluster_start, cluster_end))
        cluster_start = cluster_end

    return clusters


def _assert_distributed_factors_match_numpy(global_matrix, local_matrix,
                                            local_slice, comm):
    """Validate one distributed SVD against a serial NumPy reference SVD.

    This helper is called collectively: every rank holds the same small
    ``global_matrix`` for reference purposes, but passes only ``local_matrix``
    to ``DistributedSvd``. ``local_slice`` identifies where the local rows
    occur in the global reference matrix.

    The checks cover (1) output shapes, (2) singular values, (3) input
    preservation, (4) reconstruction, (5) orthonormality, and (6) left/right
    singular subspaces.

    Args:
        global_matrix: Complete matrix used only by the test to compute the
            serial NumPy reference result.
        local_matrix: Rows of ``global_matrix`` owned by the calling MPI rank.
        local_slice: Slice locating ``local_matrix`` inside ``global_matrix``.
        comm: MPI communicator over which the matrix rows are distributed.
    """
    # Compute the trusted serial result. All ranks construct the same small
    # reference matrix, so each rank independently obtains the same NumPy SVD.
    expected_u, expected_s, expected_vh = np.linalg.svd(
        global_matrix, full_matrices=False, compute_uv=True, hermitian=False
    )

    # Keep an exact copy to verify that the distributed implementation treats
    # its local input as read-only.
    local_matrix_before = local_matrix.copy()

    # Compute the distributed result. U remains row-distributed, whereas the
    # singular values and Vh are replicated on every rank.
    actual_u, actual_s, actual_vh = DistributedSvd(comm)(
        local_matrix,
        full_matrices=False,
        compute_uv=True,
        hermitian=False,
    )

    # Single-precision real and complex inputs need a looser tolerance than
    # float64/complex128 inputs because QR and SVD accumulate rounding errors.
    tolerance = (2e-5 if global_matrix.dtype in (np.float32, np.complex64)
                 else 2e-12)
    thin_rank = min(global_matrix.shape)

    # (1) A thin SVD of an m-by-n matrix contains k=min(m, n) singular triplets.
    # Only U's row count is local; s and Vh have their complete global shapes.
    assert actual_u.shape == (local_matrix.shape[0], thin_rank)
    assert actual_s.shape == (thin_rank,)
    assert actual_vh.shape == (thin_rank, global_matrix.shape[1])

    # (2) Singular values are unique and ordered, so they can be compared
    # directly.
    np.testing.assert_allclose(actual_s, expected_s,
                               rtol=tolerance, atol=tolerance)

    # (3) Input preservation is exact rather than approximate.
    np.testing.assert_array_equal(local_matrix, local_matrix_before)

    # (4) Multiplying U by the one-dimensional s array scales each column of U.
    # It is equivalent to U @ diag(s), without constructing the diagonal matrix.
    # Each rank reconstructs and checks only the rows that it owns.
    local_reconstruction = (actual_u*actual_s) @ actual_vh
    np.testing.assert_allclose(
        local_reconstruction,
        global_matrix[local_slice],
        rtol=tolerance,
        atol=tolerance,
    )

    # (5) If U is split into row blocks U_i, then U^H U equals the sum of
    # U_i^H U_i. Allreduce forms this global Gram matrix on every rank.
    local_gram = actual_u.conj().T @ actual_u
    global_gram = np.empty_like(local_gram)
    comm.Allreduce(local_gram, global_gram, op=MPI.SUM)
    np.testing.assert_allclose(
        global_gram,
        np.eye(thin_rank),
        rtol=tolerance,
        atol=tolerance,
    )

    # (5) Vh is replicated rather than distributed, so its row orthonormality
    # can be checked locally without communication. conjugate-transpose supports
    # both real and complex matrices.
    np.testing.assert_allclose(
        actual_vh @ actual_vh.conj().T,
        np.eye(thin_rank),
        rtol=tolerance,
        atol=tolerance,
    )

    # (6) Compare invariant subspaces one singular-value cluster at a time. This
    # permits NumPy and TSQR to choose different signs, complex phases, or
    # bases within a repeated-singular-value subspace.
    for cluster in _positive_singular_value_clusters(expected_s,
                                                      global_matrix.shape):
        expected_right = expected_vh[cluster, :]
        actual_right = actual_vh[cluster, :]

        # Two orthonormal bases span the same right singular subspace exactly
        # when their orthogonal projectors are equal. Projectors remove the
        # sign/phase/rotation ambiguity of individual singular vectors.
        np.testing.assert_allclose(
            actual_right.conj().T @ actual_right,
            expected_right.conj().T @ expected_right,
            rtol=10*tolerance,
            atol=10*tolerance,
        )

        # Each rank contributes the overlap between its NumPy and distributed
        # blocks of U. Summation reconstructs the global overlap matrix
        # expected_U^H @ actual_U without gathering either global U matrix.
        expected_local_left = expected_u[local_slice, cluster]
        local_overlap = expected_local_left.conj().T @ actual_u[:, cluster]
        global_overlap = np.empty_like(local_overlap)
        comm.Allreduce(local_overlap, global_overlap, op=MPI.SUM)

        # The singular values of the overlap are cosines of the principal
        # angles between the two subspaces. All ones means the spaces agree.
        overlap_singular_values = np.linalg.svd(
            global_overlap, compute_uv=False
        )
        np.testing.assert_allclose(
            overlap_singular_values,
            np.ones(overlap_singular_values.size),
            rtol=10*tolerance,
            atol=10*tolerance,
        )


####################
### Define Tests ###
####################

@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize(
    "case",
    [
        "tall-real",
        "wide-real",
        "tall-complex",
        "wide-complex64",
        "rank-deficient",
        "repeated-singular-values",
    ],
)
def test_distributed_svd_matches_numpy_for_uneven_partitions(case):
    comm = MPI.COMM_WORLD
    global_matrix = _matrix_for_case(case, comm.Get_size())
    local_matrix, local_slice, counts = _uneven_partition(global_matrix, comm)

    if comm.Get_size() > 1:
        assert np.unique(counts).size > 1

    _assert_distributed_factors_match_numpy(
        global_matrix, local_matrix, local_slice, comm
    )


@pytest.mark.mpi(min_size=1)
def test_distributed_svd_supports_zero_local_rows():
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()
    rng = np.random.default_rng(112358)

    if comm_size == 1:
        global_matrix = np.empty((0, 5))
        counts = np.array([0])
    else:
        global_matrix = rng.normal(size=(2*comm_size + 3, 5))
        counts = np.zeros(comm_size, dtype=int)
        rows_on_nonempty_ranks = global_matrix.shape[0]
        counts[1:] = rows_on_nonempty_ranks // (comm_size - 1)
        counts[1:1 + rows_on_nonempty_ranks % (comm_size - 1)] += 1

    offsets = np.concatenate(([0], np.cumsum(counts)))
    local_slice = slice(offsets[rank], offsets[rank + 1])
    local_matrix = global_matrix[local_slice].copy()

    assert counts[0] == 0
    _assert_distributed_factors_match_numpy(
        global_matrix, local_matrix, local_slice, comm
    )


@pytest.mark.mpi(min_size=1)
def test_distributed_svd_supports_zero_global_columns():
    comm = MPI.COMM_WORLD
    global_matrix = np.empty((2*comm.Get_size() + 1, 0))
    local_matrix, local_slice, _ = _uneven_partition(global_matrix, comm)

    _assert_distributed_factors_match_numpy(
        global_matrix, local_matrix, local_slice, comm
    )


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("full_matrices", [False, True])
@pytest.mark.parametrize("complex_input", [False, True])
def test_distributed_compute_uv_false_matches_numpy(full_matrices,
                                                    complex_input):
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(173205)
    shape = (2*comm.Get_size() + 5, 6)
    global_matrix = rng.normal(size=shape)
    if complex_input:
        global_matrix = global_matrix + 1j*rng.normal(size=shape)
    local_matrix, _, _ = _uneven_partition(global_matrix, comm)

    expected = np.linalg.svd(
        global_matrix,
        full_matrices=full_matrices,
        compute_uv=False,
        hermitian=False,
    )
    actual = DistributedSvd(comm)(
        local_matrix,
        full_matrices=full_matrices,
        compute_uv=False,
        hermitian=False,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.mpi(min_size=1)
def test_distributed_svd_preserves_communicator():
    comm = MPI.COMM_WORLD
    global_matrix = _matrix_for_case("tall-real", comm.Get_size())
    local_matrix, _, _ = _uneven_partition(global_matrix, comm)
    distributed_svd = DistributedSvd(comm)

    distributed_svd(local_matrix, full_matrices=False)

    assert distributed_svd._comm is comm
    comm.Barrier()
    assert comm.Get_size() >= 1


class _GatherRecordingCommunicator:
    """MPI communicator proxy that records the shapes passed to ``gather``.

    ``DistributedSvd`` only relies on a small subset of the communicator API.
    This test double forwards those operations to a real communicator while
    intercepting ``gather`` to record what each rank sends. Communication still
    occurs normally, but the test can verify that the algorithm gathers reduced
    R factors instead of the original local input matrices.

    Attributes:
        gathered_shapes: Shapes of all values supplied by this rank to
            ``gather``, in call order.
    """

    def __init__(self, comm):
        self._comm = comm
        self.gathered_shapes = []

    def allgather(self, value):
        return self._comm.allgather(value)

    def gather(self, value, root=0):
        """Record the outgoing shape, then perform the real gather."""
        self.gathered_shapes.append(value.shape)
        return self._comm.gather(value, root=root)

    def bcast(self, value, root=0):
        return self._comm.bcast(value, root=root)

    def scatter(self, value, root=0):
        return self._comm.scatter(value, root=root)

    def Get_rank(self):
        return self._comm.Get_rank()


@pytest.mark.mpi(min_size=1)
def test_distributed_svd_gathers_only_reduced_r_factors():
    comm = MPI.COMM_WORLD

    recording_comm = _GatherRecordingCommunicator(comm)
    column_count = 4

    # Use a tall local matrix. Its reduced QR factor R has shape (4, 4), which
    # is deliberately different from the local input shape (7, 4).
    local_matrix = np.arange(
        (column_count + 3)*column_count, dtype=float
    ).reshape(column_count + 3, column_count)
    local_matrix += 100.0*comm.Get_rank()

    DistributedSvd(recording_comm)(local_matrix, full_matrices=False)

    # Exactly one gather occurred, and it carried the small square R factor
    # rather than this rank's full local input matrix.
    assert recording_comm.gathered_shapes == [(column_count, column_count)]
    assert recording_comm.gathered_shapes[0] != local_matrix.shape


###########################
### Test error handling ###
###########################

@pytest.mark.mpi(min_size=1)
def test_distributed_full_matrices_true_with_vectors_is_rejected_collectively():
    comm = MPI.COMM_WORLD
    with pytest.raises(NotImplementedError, match="full_matrices=False"):
        DistributedSvd(comm)(
            np.eye(3), full_matrices=True, compute_uv=True, hermitian=False
        )


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("compute_uv", [False, True])
def test_distributed_hermitian_true_is_rejected_collectively(compute_uv):
    comm = MPI.COMM_WORLD
    with pytest.raises(NotImplementedError, match="hermitian=True"):
        DistributedSvd(comm)(
            np.eye(3),
            full_matrices=False,
            compute_uv=compute_uv,
            hermitian=True,
        )


@pytest.mark.mpi(min_size=1)
def test_distributed_inconsistent_column_counts_raise_collectively():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        pytest.skip("column-count inconsistency requires multiple ranks")

    local_columns = 4 if comm.Get_rank() == 0 else 5
    with pytest.raises(ValueError, match="same number of matrix columns"):
        DistributedSvd(comm)(
            np.ones((3, local_columns)), full_matrices=False
        )


@pytest.mark.mpi(min_size=1)
def test_distributed_inconsistent_dtypes_raise_collectively():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        pytest.skip("dtype inconsistency requires multiple ranks")

    local_dtype = np.float64 if comm.Get_rank() == 0 else np.complex128
    with pytest.raises(ValueError, match="same matrix dtype"):
        DistributedSvd(comm)(
            np.ones((3, 4), dtype=local_dtype), full_matrices=False
        )


@pytest.mark.mpi(min_size=1)
def test_distributed_inconsistent_options_raise_collectively():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        pytest.skip("option inconsistency requires multiple ranks")

    compute_uv = comm.Get_rank() != 0
    with pytest.raises(ValueError, match="same SVD options"):
        DistributedSvd(comm)(
            np.ones((3, 4)),
            full_matrices=False,
            compute_uv=compute_uv,
        )


@pytest.mark.mpi(min_size=1)
def test_distributed_invalid_dimension_raises_collectively():
    comm = MPI.COMM_WORLD
    local_matrix = (np.ones(5) if comm.Get_rank() == 0
                    else np.ones((2, 5)))

    with pytest.raises(ValueError, match="two-dimensional"):
        DistributedSvd(comm)(local_matrix, full_matrices=False)
