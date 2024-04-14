import numpy as np
import random

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.linalg import _basic_product_via_python


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_basic_product_via_python_mat_mat():
    '''Tests 2A^T A where A is row-distributed'''
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    n_rows = num_processes * 2
    n_cols = 3
    n_local_rows = n_rows // num_processes

    A = np.random.rand(n_rows, n_cols)
    A = comm.bcast(A, root=0)

    A_dist = np.zeros((n_local_rows, n_cols))
    comm.Scatter(A, A_dist, root=0)
    C = np.zeros((n_cols,n_cols))

    _basic_product_via_python("T", "N", 2, A_dist, A_dist, 0, C, comm)   # 2A^T A
    expected = np.dot(2*A.transpose(), A)

    assert np.allclose(C, expected)

def test_basic_product_via_python_constraints():
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    m = 2
    n = 3
    l = 4

    A_corr = np.random.rand(m,n)

    B_corr = np.random.rand(n,l)
    C_corr = np.zeros((m,l))

    B_wrong = np.random.rand(m,l) # should be (n,l)
    C_wrong = np.zeros((m,n))     # should be (m,l)

    try:
        _basic_product_via_python("Transpose", "N", 1, A_corr, B_corr, 1, C_corr, comm)
    except ValueError as e:
        assert str(e) == f"flagA not recognized; use either 'N' or 'T'"

    try:
        _basic_product_via_python("N", "Transpose", 1, A_corr, B_corr, 1, C_corr, comm)
    except ValueError as e:
        assert str(e) == f"flagB not recognized; use either 'N' or 'T'"

    try:
        _basic_product_via_python("N", "N", 1, A_corr, B_wrong, 1, C_corr, comm)
    except ValueError as e:
        assert str(e) == f"Invalid input array size. For A (m x n), B must be (n x l)."

    try:
        _basic_product_via_python("N", "N", 1, A_corr, B_corr, 1, C_wrong, comm)
    except ValueError as e:
        assert str(e) == f"Size of output array C ({np.shape(C_wrong)}) is invalid. For A (m x n) and B (n x l), C has dimensions (m x l))."

    a_vec = np.random.rand(m)

    try:
        _basic_product_via_python("N", "N", 1, a_vec, B_corr, 1, C_corr, comm)
    except ValueError as e:
        assert str(e) == f"This operation currently supports rank-2 tensors."

def test_basic_product_serial():
    '''Tests 2A^T A'''

    n_rows = 6
    n_cols = 3

    A = np.random.rand(n_rows, n_cols)
    C = np.zeros((n_cols,n_cols))

    _basic_product_via_python("T", "N", 2, A, A, 0, C)
    expected = np.dot(2*A.transpose(), A)

    assert np.allclose(C, expected)

if __name__ == "__main__":
    test_basic_product_via_python_mat_mat()
    test_basic_product_via_python_constraints()
    test_basic_max_serial()
