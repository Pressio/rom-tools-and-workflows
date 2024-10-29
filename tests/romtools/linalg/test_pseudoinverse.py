import numpy as np
import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl
from romtools.linalg.linalg import _transposed_pseudoinverse_via_python, _basic_product_via_python

def test_transposed_pseudoinverse():
    matrix = np.random.rand(7,5)
    pinv = _transposed_pseudoinverse_via_python(matrix)
    product = np.zeros((pinv.shape[1], matrix.shape[1]))
    _basic_product_via_python("T", "N", 1, pinv, matrix, 0, product)
    assert np.allclose(product, np.eye(product.shape[0]))

@pytest.mark.mpi(min_size=3)
def test_transposed_pseudoinverse_dist():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_A, A = generate_random_local_and_global_arrays_impl((18, 4), comm)
    pinv_T = _transposed_pseudoinverse_via_python(local_A, comm)
    # let A be the matrix that we are distributing of which local_A is the local part
    # if pinv_T = A^* then we should have A^* A = I
    local_result = np.zeros((pinv_T.shape[1], local_A.shape[1]), dtype=local_A.dtype)
    _basic_product_via_python("T", "N", 1, pinv_T, local_A, 0, local_result, comm)
    assert np.allclose(local_result, np.eye(pinv_T.shape[1]))

    # now verify the result matches the pinv computed on the full A
    global_pinv = _transposed_pseudoinverse_via_python(A)
    local_pinvs = comm.allgather(pinv_T)
    assembled_pinv = np.vstack(local_pinvs)
    assert np.allclose(global_pinv, assembled_pinv)

@pytest.mark.mpi(min_size=3)
def test_transposed_pseudoinverse_dist_single_row_arrays():
    '''This test recreates the sampling of DEIM.'''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if comm.Get_size != 3:
        return

    local_A, A = generate_random_local_and_global_arrays_impl((18, 4), comm)

    sliced_A = A[np.array([0, 6, 7, 12, 13])]
    slicing_indices = np.array([0]) if rank == 0 else np.array([0,1])
    sliced_local_A = local_A[slicing_indices] # this is what DEIM does

    pinv_T = _transposed_pseudoinverse_via_python(sliced_local_A, comm)

    local_result = np.zeros((pinv_T.shape[1], sliced_local_A.shape[1]), dtype=sliced_local_A.dtype)
    _basic_product_via_python("T", "N", 1, pinv_T, sliced_local_A, 0, local_result, comm)
    assert np.allclose(local_result, np.eye(pinv_T.shape[1]))

    global_pinv = _transposed_pseudoinverse_via_python(sliced_A)
    local_pinvs = comm.allgather(pinv_T)
    assembled_pinv = np.vstack(local_pinvs)
    assert np.allclose(global_pinv, assembled_pinv)

@pytest.mark.mpi(min_size=3)
def test_transposed_pseudoinverse_dist_empty_rank():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if comm.Get_size() != 3:
        return

    local_A, global_A = generate_random_local_and_global_arrays_impl((18, 4), comm)
    sliced_A = global_A[np.array([0, 1, 2, 6, 7])]

    slicing_indices = []
    if rank == 0:
        slicing_indices.extend([0,1,2])
    elif rank == 1:
        slicing_indices.extend([0,1])

    sliced_local_A = local_A[slicing_indices]

    pinv_T = _transposed_pseudoinverse_via_python(sliced_local_A, comm=comm)

    local_result = np.zeros((pinv_T.shape[1], sliced_local_A.shape[1]), dtype=sliced_local_A.dtype)
    _basic_product_via_python("T", "N", 1, pinv_T, sliced_local_A, 0, local_result, comm)
    assert np.allclose(local_result, np.eye(pinv_T.shape[1]))

    global_pinv = _transposed_pseudoinverse_via_python(sliced_A)

    local_pinvs = comm.allgather(pinv_T)
    assembled_pinv = np.vstack(local_pinvs)

    assert np.allclose(global_pinv, assembled_pinv)

if __name__ == "__main__":
    test_transposed_pseudoinverse()
    test_transposed_pseudoinverse_dist()
    test_transposed_pseudoinverse_dist_single_row_arrays
    test_transposed_pseudoinverse_dist_empty_rank
