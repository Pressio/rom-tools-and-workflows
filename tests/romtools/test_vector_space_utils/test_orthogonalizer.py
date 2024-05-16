import pytest
import numpy as np
import scipy.sparse
from romtools.vector_space.utils.orthogonalizer import NoOpOrthogonalizer, \
                                                       EuclideanL2Orthogonalizer, \
                                                       EuclideanVectorWeightedL2Orthogonalizer
from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl, \
                                           distribute_array_impl
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


class ParallelQR:
    def __init__(self, comm):
        self.__comm = comm

    def __call__(self, local_arr: np.ndarray, mode: str):
        # Gather all of the local arrays
        local_arrays = self.__comm.allgather(local_arr)

        # Reconstruct the global array
        global_array = np.vstack(local_arrays)

        # Perform qr decomp on global array
        global_qr, _ = np.linalg.qr(global_array, mode=mode)

        # Then distribute back to the original processes and return (along with a dummy variable)
        local_qr = distribute_array_impl(global_qr, comm=self.__comm, dist_axis=0)
        return local_qr, _


@pytest.mark.mpi_skip
def test_noop_orthogonalizer():
    orthogonalizer = NoOpOrthogonalizer()
    my_basis = np.random.normal(size=(10, 2))
    my_orthogonalized_basis = orthogonalizer.orthogonalize(my_basis)
    assert np.allclose(my_orthogonalized_basis, my_basis)


@pytest.mark.mpi(min_size=3)
def test_noop_orthogonalizer_mpi():
    comm = MPI.COMM_WORLD
    orthogonalizer = NoOpOrthogonalizer()
    basis_shape = (10, 2)
    local_basis, global_basis = generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
    local_orthogonalized_basis = orthogonalizer.orthogonalize(local_basis)
    global_orthogonalized_basis = orthogonalizer.orthogonalize(global_basis)
    dist_global_orthogonalized_basis = distribute_array_impl(global_orthogonalized_basis, comm=comm, dist_axis=0)
    assert np.allclose(local_orthogonalized_basis, dist_global_orthogonalized_basis)


@pytest.mark.mpi_skip
def test_euclidean_l2_orthogonalizer():
    orthogonalizer = EuclideanL2Orthogonalizer()
    my_basis = np.random.normal(size=(10, 2))
    my_orthogonalized_basis = orthogonalizer.orthogonalize(my_basis)
    should_be_eye = my_orthogonalized_basis.transpose() @ my_orthogonalized_basis
    assert np.allclose(should_be_eye, np.eye(2))


@pytest.mark.mpi(min_size=3)
def test_euclidean_l2_orthogonalizer_mpi():
    comm = MPI.COMM_WORLD
    parallel_qr = ParallelQR(comm)
    parallel_orthogonalizer = EuclideanL2Orthogonalizer(qrFnc=parallel_qr)
    serial_orthogonalizer = EuclideanL2Orthogonalizer()
    basis_shape = (10, 2)
    local_basis, global_basis = generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
    local_orthogonalized_basis = parallel_orthogonalizer.orthogonalize(local_basis)
    global_orthogonalized_basis = serial_orthogonalizer.orthogonalize(global_basis)
    dist_global_orthogonalized_basis = distribute_array_impl(global_orthogonalized_basis, comm=comm, dist_axis=0)
    assert np.allclose(local_orthogonalized_basis, dist_global_orthogonalized_basis)


@pytest.mark.mpi_skip
def test_euclidean_vector_weighted_l2_orthogonalizer():
    np.random.seed(1)
    vec_to_orthogonalize_against = np.abs(np.random.normal(size=10))
    orthogonalizer = EuclideanVectorWeightedL2Orthogonalizer(vec_to_orthogonalize_against)
    my_basis = np.random.normal(size=(10, 2))
    my_orthogonalized_basis = orthogonalizer.orthogonalize(my_basis)
    should_be_eye = my_orthogonalized_basis.transpose() @ (scipy.sparse.diags(vec_to_orthogonalize_against) @ my_orthogonalized_basis)
    assert np.allclose(should_be_eye, np.eye(2)), should_be_eye


@pytest.mark.mpi(min_size=3)
def test_euclidean_vector_weighted_l2_orthogonalizer_mpi():
    comm = MPI.COMM_WORLD
    basis_shape = (10, 2)
    local_basis, global_basis = generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
    np.random.seed(1)

    vec_to_orthogonalize_against = np.abs(np.random.normal(size=10))
    serial_orthogonalizer = EuclideanVectorWeightedL2Orthogonalizer(vec_to_orthogonalize_against)
    global_orthogonalized_basis = serial_orthogonalizer.orthogonalize(global_basis)
    dist_global_orthogonalized_basis = distribute_array_impl(global_orthogonalized_basis, comm=comm, dist_axis=0)

    parallel_qr = ParallelQR(comm)
    local_vec_to_orthogonalize_against = distribute_array_impl(vec_to_orthogonalize_against, comm=comm)
    parallel_orthogonalizer = EuclideanVectorWeightedL2Orthogonalizer(local_vec_to_orthogonalize_against, parallel_qr)
    local_orthogonalized_basis = parallel_orthogonalizer.orthogonalize(local_basis)

    assert np.allclose(local_orthogonalized_basis, dist_global_orthogonalized_basis)


if __name__ == "__main__":
    test_noop_orthogonalizer()
    test_noop_orthogonalizer_mpi()
    test_euclidean_l2_orthogonalizer()
    test_euclidean_l2_orthogonalizer_mpi()
    test_euclidean_vector_weighted_l2_orthogonalizer()
    test_euclidean_vector_weighted_l2_orthogonalizer_mpi()
