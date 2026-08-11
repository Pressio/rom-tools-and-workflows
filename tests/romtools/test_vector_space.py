import copy
import pytest
import numpy as np
import romtools as rt
import romtools.vector_space.utils as utils


# @pytest.mark.mpi_skip
# def test_list_snapshots_to_array():
#    snapshots = np.random.normal(size=(15,7))
#    snapshot_data = PythonSnapshotData(snapshots)
#    matrix = snapshot_data.getSnapshotsAsArray()
#    assert matrix.shape[0] == 15
#    assert matrix.shape[1] == 7


def _tensor_to_matrix(tensor_input):
    return tensor_input.reshape(tensor_input.shape[0] * tensor_input.shape[1], tensor_input.shape[2])


@pytest.mark.mpi_skip
def test_dictionary_vector_space():
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    # default test
    my_vector_space = rt.DictionaryVectorSpace(snapshots)
    assert np.allclose(my_vector_space.get_basis().ravel(), snapshots.ravel())
    assert np.allclose(my_vector_space.get_shift_vector(), 0)
    assert np.allclose(my_vector_space.extents()[-1], 6)

    # test with a shift
    my_shifter = utils.create_average_shifter(snapshots)
    my_vector_space = rt.DictionaryVectorSpace(snapshots, my_shifter)
    assert np.allclose(my_vector_space.get_basis().ravel(), (original_snapshots - np.mean(original_snapshots, axis=2)[:, :, None]).ravel())
    assert np.allclose(my_vector_space.get_shift_vector(), np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.extents()[-1], 6)

    # test with a shift and orthogonalization
    my_shifter = utils.create_average_shifter(snapshots)
    my_orthogonalizer = utils.EuclideanL2Orthogonalizer()
    my_vector_space = rt.DictionaryVectorSpace(snapshots, my_shifter, my_orthogonalizer)
    assert np.allclose(my_vector_space.get_shift_vector(), np.mean(snapshots, axis=2))
    assert np.allclose(my_vector_space.extents()[2], 6)
    basis = my_vector_space.get_basis()
    basis = _tensor_to_matrix(basis)
    assert np.allclose(basis.transpose() @ basis, np.eye(6))


@pytest.mark.mpi_skip
def test_vector_space_from_pod():
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    my_vector_space = rt.VectorSpaceFromPOD(snapshots)
    # truth vector space
    snapshotMatrix = _tensor_to_matrix(original_snapshots)
    u, s, v = np.linalg.svd(snapshotMatrix, full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(6, my_vector_space.extents()[-1])
    assert np.allclose(0, my_vector_space.get_shift_vector())

    # test with a shift
    my_shifter = utils.create_average_shifter(snapshots)
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter)
    u, s, v = np.linalg.svd(snapshotMatrix - np.mean(snapshotMatrix, axis=1)[:, None], full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)  # FAILS
    assert np.allclose(my_vector_space.get_shift_vector(), np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.extents()[-1], 6)

    # test with a shift and orthogonalization
    snapshots = np.random.normal(size=(3, 8, 6))
    original_snapshots = snapshots.copy()
    snapshotMatrix = _tensor_to_matrix(original_snapshots)
    my_shifter = utils.create_average_shifter(snapshots)
    weighting = np.abs(np.random.normal(size=24))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter, orthogonalizer=my_orthogonalizer)
    u, s, v = np.linalg.svd(snapshotMatrix - np.mean(snapshotMatrix, axis=1)[:, None], full_matrices=False)
    u = my_orthogonalizer.orthogonalize(u)
    basis_tensor = my_vector_space.get_basis()
    assert np.allclose(u.reshape(basis_tensor.shape), basis_tensor)
    assert np.allclose(my_vector_space.get_shift_vector(), np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.extents()[2], 6)


@pytest.mark.mpi_skip
def test_trial_space_from_scaled_pod():
    snapshots = np.random.normal(size=(3, 8, 6))
    my_scaler = utils.VariableScaler("max_abs")
    my_vector_space = rt.VectorSpaceFromPOD(copy.deepcopy(snapshots), scaler=my_scaler)
    my_scaler.pre_scale(snapshots)
    snapshotMatrix = _tensor_to_matrix(snapshots)
    u, s, v = np.linalg.svd(snapshotMatrix, full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    my_scaler.post_scale(u)
    assert np.allclose(u, basis_tensor), print(u, my_vector_space.get_basis())
    assert np.allclose(6, my_vector_space.extents()[-1])
    assert np.allclose(0, my_vector_space.get_shift_vector())

    # test with a shift
    snapshots = np.random.normal(size=(3, 8, 6))
    shifted_snapshots = snapshots.copy()
    original_snapshots = snapshots.copy()
    my_shifter = utils.create_average_shifter(snapshots)
    my_scaler = utils.VariableScaler("max_abs")
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter, scaler=my_scaler)
    my_shifter.apply_shift(shifted_snapshots)
    my_scaler.pre_scale(shifted_snapshots)
    snapshot_matrix = _tensor_to_matrix(shifted_snapshots)
    u, s, v = np.linalg.svd(snapshot_matrix, full_matrices=False)
    basis_tensor = my_vector_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    my_scaler.post_scale(u)
    assert np.allclose(basis_tensor, u)  # FAILS
    assert np.allclose(my_vector_space.get_shift_vector(), np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.extents()[-1], 6)

    # test with a shift and orthogonalization
    snapshots = np.random.normal(size=(3, 8, 6))
    shifted_snapshots = snapshots.copy()
    original_snapshots = snapshots.copy()
    my_scaler = utils.VariableScaler("max_abs")
    my_shifter = utils.create_average_shifter(snapshots)
    weighting = np.abs(np.random.normal(size=24))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    my_vector_space = rt.VectorSpaceFromPOD(snapshots, shifter=my_shifter, scaler=my_scaler, orthogonalizer=my_orthogonalizer)
    my_shifter.apply_shift(shifted_snapshots)
    my_scaler = utils.VariableScaler("max_abs")
    my_scaler.pre_scale(shifted_snapshots)
    snapshot_matrix = _tensor_to_matrix(shifted_snapshots)
    u, s, v = np.linalg.svd(snapshot_matrix, full_matrices=False)
    ushp = u.shape
    basis_tensor = my_vector_space.get_basis()
    u = u.reshape(basis_tensor.shape)
    my_scaler.post_scale(u)
    u = my_orthogonalizer.orthogonalize(u.reshape(ushp))
    u = u.reshape(basis_tensor.shape)
    assert np.allclose(basis_tensor, u)
    assert np.allclose(my_vector_space.get_shift_vector(), np.mean(original_snapshots, axis=2))
    assert np.allclose(my_vector_space.extents()[2], 6)


@pytest.mark.mpi_skip
def test_vector_space_from_streaming_pod():
    rng = np.random.default_rng(42)
    snapshots = rng.normal(size=(3, 8, 6))
    requested_ranges = []

    class Loader:
        def __call__(self, start, end):
            requested_ranges.append((start, end))
            return snapshots[..., start:end]

    loader = Loader()

    vector_space = rt.VectorSpaceFromStreamingPOD(
        snapshot_loader=loader,
        block_size=4,
        n_snapshots=6,
        basis_dimension=3,
        oversampling=3,
    )

    snapshot_matrix = _tensor_to_matrix(snapshots)
    exact_basis, exact_svals, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)
    basis_matrix = _tensor_to_matrix(vector_space.get_basis())

    assert vector_space.extents() == (3, 8, 3)
    assert np.allclose(vector_space.get_shift_vector(), 0.0)
    assert np.allclose(vector_space.get_singular_values(), exact_svals[:3])
    assert np.allclose(
        np.abs(basis_matrix.T @ exact_basis[:, :3]),
        np.eye(3),
        atol=1e-10,
    )
    assert requested_ranges == [(0, 4), (4, 6)] * 2


@pytest.mark.mpi_skip
def test_streaming_vector_space_orthogonalizer():
    rng = np.random.default_rng(13)
    snapshots = rng.normal(size=(2, 5, 4))
    loader = lambda start, end: snapshots[..., start:end]

    vector_space = rt.VectorSpaceFromStreamingPOD(
        snapshot_loader=loader,
        block_size=3,
        n_snapshots=4,
        basis_dimension=2,
        oversampling=2,
        orthogonalizer=utils.EuclideanL2Orthogonalizer(),
    )

    basis_matrix = _tensor_to_matrix(vector_space.get_basis())
    assert np.allclose(basis_matrix.T @ basis_matrix, np.eye(2))


@pytest.mark.mpi_skip
def test_streaming_vector_space_requires_tensor_blocks():
    snapshots = np.ones((6, 4))
    loader = lambda start, end: snapshots[:, start:end]

    with pytest.raises(ValueError, match="three-dimensional"):
        rt.VectorSpaceFromStreamingPOD(
            snapshot_loader=loader,
            block_size=2,
            n_snapshots=4,
            basis_dimension=1,
            oversampling=0,
        )


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "scaler_factory, expected_passes",
    [
        (lambda: utils.NoOpScaler(), 2),
        (lambda: utils.ScalarScaler(2.5), 2),
        (lambda: utils.VectorScaler(np.linspace(0.5, 1.5, 5)), 2),
        (lambda: utils.VariableScaler("max_abs"), 3),
        (lambda: utils.VariableScaler("mean_abs"), 3),
        (lambda: utils.VariableScaler("variance"), 3),
        (
            lambda: utils.VariableAndVectorScaler(
                np.linspace(0.5, 1.5, 5), "max_abs"
            ),
            3,
        ),
    ],
)
def test_streaming_vector_space_scalers(scaler_factory, expected_passes):
    rng = np.random.default_rng(91)
    snapshots = rng.normal(size=(3, 5, 6))
    snapshots[..., 4:] *= 20.0
    original_snapshots = snapshots.copy()
    requested_ranges = []

    def loader(start, end):
        requested_ranges.append((start, end))
        return snapshots[..., start:end]

    streaming_space = rt.VectorSpaceFromStreamingPOD(
        snapshot_loader=loader,
        block_size=2,
        n_snapshots=6,
        basis_dimension=3,
        oversampling=3,
        scaler=scaler_factory(),
    )
    in_memory_space = rt.VectorSpaceFromPOD(
        snapshots=snapshots.copy(),
        truncater=utils.BasisSizeTruncater(3),
        scaler=scaler_factory(),
    )

    streaming_basis = _tensor_to_matrix(streaming_space.get_basis())
    in_memory_basis = _tensor_to_matrix(in_memory_space.get_basis())
    for mode in range(3):
        correlation = abs(
            streaming_basis[:, mode] @ in_memory_basis[:, mode]
        )
        correlation /= (
            np.linalg.norm(streaming_basis[:, mode])
            * np.linalg.norm(in_memory_basis[:, mode])
        )
        assert np.isclose(correlation, 1.0, atol=1e-10)

    assert np.allclose(
        streaming_space.get_singular_values(),
        in_memory_space.get_singular_values()[:3],
    )
    assert np.array_equal(snapshots, original_snapshots)
    assert requested_ranges == [(0, 2), (2, 4), (4, 6)] * expected_passes


if __name__ == "__main__":
    test_dictionary_vector_space()
    test_vector_space_from_pod()
    test_vector_space_from_scaled_pod()
