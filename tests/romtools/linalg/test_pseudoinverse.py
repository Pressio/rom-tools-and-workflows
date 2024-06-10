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
    print("matrix shape: ", matrix.shape)
    print("product shape: ", product.shape)
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

if __name__ == "__main__":
    test_transposed_pseudoinverse()
    test_transposed_pseudoinverse_dist()
