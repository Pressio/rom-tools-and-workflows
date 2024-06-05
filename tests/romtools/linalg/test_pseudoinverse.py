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

    local_matrix, global_matrix = generate_random_local_and_global_arrays_impl((18, 4), comm)

    pinv = _transposed_pseudoinverse_via_python(local_matrix, comm)
    global_pinv = _transposed_pseudoinverse_via_python(global_matrix)

    local_result = np.zeros((pinv.shape[1], local_matrix.shape[1]), dtype=local_matrix.dtype)
    _basic_product_via_python("T", "N", 1, pinv, local_matrix, 0, local_result, comm)

    assert np.allclose(local_result, np.eye(pinv.shape[1]))

    local_pinvs = comm.allgather(pinv)
    assembled_pinv = np.vstack(local_pinvs)

    assert np.allclose(global_pinv, assembled_pinv)

if __name__ == "__main__":
    test_transposed_pseudoinverse()
    test_transposed_pseudoinverse_dist()
