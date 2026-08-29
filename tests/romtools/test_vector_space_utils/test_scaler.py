import copy
import pytest
import numpy as np
from romtools.vector_space.utils.scaler import NoOpScaler, ScalarScaler, VectorScaler, VariableScaler, VariableAndVectorScaler
import romtools.linalg.linalg as la
from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


@pytest.mark.mpi_skip
def test_noop_scaler():
    scaler = NoOpScaler()
    my_snapshots = np.random.normal(size=(3, 10, 2))
    orig_snapshots = copy.deepcopy(my_snapshots)
    scaler.pre_scale(my_snapshots)
    assert np.allclose(my_snapshots, orig_snapshots)
    scaler.post_scale(my_snapshots)
    assert np.allclose(my_snapshots, orig_snapshots)


@pytest.mark.mpi(min_size=3)
def test_noop_scaler_mpi():
    comm = MPI.COMM_WORLD
    scaler = NoOpScaler()
    local_snapshots, _ = generate_random_local_and_global_arrays_impl((3, 10, 2), comm=comm)
    orig_snapshots = copy.deepcopy(local_snapshots)
    scaler.pre_scale(local_snapshots)
    assert np.allclose(local_snapshots, orig_snapshots)
    scaler.post_scale(local_snapshots)
    assert np.allclose(local_snapshots, orig_snapshots)


def scaling_op(scaling_type, arg):
    if scaling_type == "max_abs":
        return la.max(np.abs(arg))
    elif scaling_type == "mean_abs":
        return la.mean(np.abs(arg))
    elif scaling_type == "variance":
        return la.std(arg)


@pytest.mark.mpi_skip
def test_scalar_scaler():
    n_var = 3
    nx = 5
    my_snapshots = np.random.normal(size=(n_var, nx, 8))
    my_scaling_factor = np.random.rand()
    my_initial_snapshots = copy.deepcopy(my_snapshots)
    scaler = ScalarScaler(my_scaling_factor)

    scaler.pre_scale(my_snapshots)
    assert np.allclose(my_snapshots, 1.0 / my_scaling_factor * my_initial_snapshots)

    scaler.post_scale(my_snapshots)
    assert np.allclose(my_snapshots, my_initial_snapshots)


@pytest.mark.mpi_skip
def test_vector_scaler():
    n_var = 3
    nx = 5
    my_snapshots = np.random.normal(size=(n_var, nx, 5))
    my_scaling_vector = np.abs(np.random.normal(size=nx))
    my_initial_snapshots = copy.deepcopy(my_snapshots)
    scaler = VectorScaler(my_scaling_vector)

    scaler.pre_scale(my_snapshots)
    assert np.allclose(my_snapshots, 1.0 / my_scaling_vector[None, :, None] * my_initial_snapshots)

    scaler.post_scale(my_snapshots)
    assert np.allclose(my_snapshots, my_initial_snapshots)


@pytest.mark.mpi(min_size=3)
def test_vector_scaler_mpi():
    comm = MPI.COMM_WORLD
    n_var = 3
    nx = 5
    local_snapshots, _ = generate_random_local_and_global_arrays_impl((n_var, nx, 5), comm=comm)
    my_scaling_vector = np.abs(np.random.normal(size=local_snapshots.shape[1]))
    initial_local_snapshots = copy.deepcopy(local_snapshots)
    scaler = VectorScaler(my_scaling_vector)

    scaler.pre_scale(local_snapshots)
    assert np.allclose(local_snapshots, 1.0 / my_scaling_vector[None, :, None] * initial_local_snapshots)

    scaler.post_scale(local_snapshots)
    assert np.allclose(local_snapshots, initial_local_snapshots)


@pytest.mark.mpi_skip
def test_variable_scaler():
    n_var = 3
    nx = 5

    def run_test(scaling_type):
        my_snapshots = np.random.normal(size=(n_var, nx, 5))
        scales = np.zeros(n_var)
        for i in range(0, n_var):
            scales[i] = scaling_op(scaling_type, my_snapshots[i])

        my_initial_snapshots = copy.deepcopy(my_snapshots)
        scaler = VariableScaler(scaling_type)
        scaler.pre_scale(my_snapshots)
        for i in range(0, n_var):
            assert np.allclose(my_snapshots[i], 1.0 / scales[i] * my_initial_snapshots[i])

        scaler.post_scale(my_snapshots)
        assert np.allclose(scales, scaler.var_scales_)
        assert np.allclose(my_initial_snapshots, my_snapshots)

    run_test("max_abs")
    run_test("mean_abs")
    run_test("variance")


@pytest.mark.mpi(min_size=3)
def test_variable_scaler_mpi():
    comm = MPI.COMM_WORLD
    n_var = 3
    nx = 5

    def run_test(scaling_type):
        local_snapshots, _ = generate_random_local_and_global_arrays_impl((n_var, nx, 5), comm=comm)
        scales = np.zeros(n_var)
        for i in range(0, n_var):
            scales[i] = scaling_op(scaling_type, local_snapshots[i])

        initial_local_snapshots = copy.deepcopy(local_snapshots)
        scaler = VariableScaler(scaling_type)
        scaler.pre_scale(local_snapshots)
        for i in range(0, n_var):
            assert np.allclose(local_snapshots[i], 1.0 / scales[i] * initial_local_snapshots[i])

        scaler.post_scale(local_snapshots)
        assert np.allclose(scales, scaler.var_scales_)
        assert np.allclose(initial_local_snapshots, local_snapshots)

    run_test("max_abs")
    run_test("mean_abs")
    run_test("variance")


@pytest.mark.mpi_skip
def test_variable_and_vector_scaler():
    def run_test(scaling_type):
        n_var = 3
        nx = 5
        my_snapshots = np.random.normal(size=(n_var, nx, 5))
        my_scaling_vector = np.abs(np.random.normal(size=nx))

        scales = np.zeros(n_var)
        for i in range(0, n_var):
            scales[i] = scaling_op(scaling_type, my_snapshots[i])

        my_initial_snapshots = copy.deepcopy(my_snapshots)
        scaler = VariableAndVectorScaler(my_scaling_vector, scaling_type)
        scaler.pre_scale(my_snapshots)
        for i in range(0, n_var):
            assert np.allclose(my_snapshots[i], 1.0 / scales[i] * (1.0 / my_scaling_vector[None, :, None] * my_initial_snapshots)[i])

        scaler.post_scale(my_snapshots)
        assert np.allclose(my_initial_snapshots, my_snapshots)

    run_test("max_abs")
    run_test("mean_abs")
    run_test("variance")


@pytest.mark.mpi(min_size=3)
def test_variable_and_vector_scaler_mpi():
    def run_test(scaling_type):
        comm = MPI.COMM_WORLD
        n_var = 3
        nx = 5
        local_snapshots, global_snapshots = generate_random_local_and_global_arrays_impl((n_var, nx, 5), comm=comm)
        my_scaling_vector = np.abs(np.random.normal(size=local_snapshots.shape[1]))

        scales = np.zeros(n_var)
        for i in range(0, n_var):
            scales[i] = scaling_op(scaling_type, local_snapshots[i])

        initial_local_snapshots = copy.deepcopy(local_snapshots)
        scaler = VariableAndVectorScaler(my_scaling_vector, scaling_type)
        scaler.pre_scale(local_snapshots)
        for i in range(0, n_var):
            assert np.allclose(local_snapshots[i], 1.0 / scales[i] * (1.0 / my_scaling_vector[None, :, None] * initial_local_snapshots)[i])

        scaler.post_scale(local_snapshots)
        assert np.allclose(initial_local_snapshots, local_snapshots)

    run_test("max_abs")
    run_test("mean_abs")
    run_test("variance")


@pytest.mark.mpi_skip
@pytest.mark.parametrize("scaling_type", ["max_abs", "mean_abs", "variance"])
def test_variable_scaler_initializes_from_snapshot_loader(scaling_type):
    snapshots = np.array(
        [
            np.zeros((3, 5)),
            [
                [1.0, 2.0, 3.0, 40.0, 50.0],
                [-2.0, 4.0, 6.0, 80.0, 100.0],
                [3.0, 6.0, 9.0, 120.0, 150.0],
            ],
        ]
    )
    loader = lambda start, end: snapshots[..., start:end]
    scaler = VariableScaler(scaling_type)

    scaler.initialize_scalings_from_loader(
        snapshot_loader=loader,
        block_size=2,
        n_snapshots=5,
    )

    if scaling_type == "max_abs":
        expected = np.max(np.abs(snapshots), axis=(1, 2))
    elif scaling_type == "mean_abs":
        expected = np.mean(np.abs(snapshots), axis=(1, 2))
    else:
        expected = np.std(snapshots, axis=(1, 2))
    expected = np.where(expected < 1e-10, 1.0, expected)

    assert np.allclose(scaler.var_scales_, expected)


@pytest.mark.mpi_skip
def test_variable_scaler_streaming_rejects_inconsistent_shapes():
    snapshots = np.ones((2, 3, 4))
    call_count = 0

    def loader(start, end):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return snapshots[..., start:end]
        return np.ones((2, 4, end - start))

    scaler = VariableScaler("max_abs")
    with pytest.raises(ValueError, match="inconsistent state dimensions"):
        scaler.initialize_scalings_from_loader(loader, block_size=2, n_snapshots=4)


if __name__ == "__main__":
    test_noop_scaler()
    test_noop_scaler_mpi()
    test_vector_scaler()
    test_vector_scaler_mpi()
    test_variable_scaler()
    test_variable_scaler_mpi()
    test_variable_and_vector_scaler()
