import math
import warnings
import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl
from romtools.linalg.linalg import move_distributed_linear_system_to_rank_zero

@pytest.mark.mpi(min_size=3)
def test_move_linsys_to_rank_zero():
    np.random.seed(9243)
    comm = MPI.COMM_WORLD
    myRank = comm.Get_rank()

    scenarios = ['every_rank_has_rows', 'rank_0_empty', 'rank_1_empty', 
        'rank_2_empty', 'rank_01_empty', 'rank_02_empty']

    for num_rows in [5, 11]:
        for scenario in scenarios:
            A_l, Agold = generate_random_local_and_global_arrays_impl((num_rows, 5), comm)
            b_l, bgold = generate_random_local_and_global_arrays_impl((num_rows, 1), comm)
            comm.barrier()

            if scenario == 'every_rank_has_rows':
                A2, b2 = move_distributed_linear_system_to_rank_zero(A_l, b_l, comm)
                if myRank == 0:
                    assert np.allclose(A2, Agold)
                    assert np.allclose(b2, bgold[:,0])

            if scenario == 'rank_0_empty':
                if myRank == 0:
                    A_l, b_l = np.array([]), np.array([])
                A2, b2 = move_distributed_linear_system_to_rank_zero(A_l, b_l, comm)
                if myRank == 0:
                    rows = [4,5,6,7,8,9,10] if num_rows == 11 else [2,3,4]
                    assert np.allclose(A2, Agold[rows,:])
                    assert np.allclose(b2, bgold[rows,0])

            if scenario == 'rank_1_empty':
                if myRank == 1:
                    A_l, b_l = np.array([]), np.array([])
                A2, b2 = move_distributed_linear_system_to_rank_zero(A_l, b_l, comm)
                if myRank == 0:
                    rows = [0,1,2,3,8,9,10] if num_rows == 11 else [0,1,4]
                    assert np.allclose(A2, Agold[rows,:])
                    assert np.allclose(b2, bgold[rows,0])

            if scenario == 'rank_2_empty':
                if myRank == 2:
                    A_l, b_l = np.array([]), np.array([])
                A2, b2 = move_distributed_linear_system_to_rank_zero(A_l, b_l, comm)
                if myRank == 0:
                    rows = [0,1,2,3,4,5,6,7] if num_rows == 11 else [0,1,2,3]
                    assert np.allclose(A2, Agold[rows,:])
                    assert np.allclose(b2, bgold[rows,0])

            if scenario == 'rank_01_empty':
                if myRank <= 1:
                    A_l, b_l = np.array([]), np.array([])
                A2, b2 = move_distributed_linear_system_to_rank_zero(A_l, b_l, comm)
                if myRank == 0:
                    rows = [8,9,10] if num_rows == 11 else [4]
                    assert np.allclose(A2, Agold[rows,:])
                    assert np.allclose(b2, bgold[rows,0])

            if scenario == 'rank_02_empty':
                if myRank == 0 or myRank == 2:
                    A_l, b_l = np.array([]), np.array([])
                A2, b2 = move_distributed_linear_system_to_rank_zero(A_l, b_l, comm)
                if myRank == 0:
                    rows = [4,5,6,7] if num_rows == 11 else [2,3]
                    assert np.allclose(A2, Agold[rows,:])
                    assert np.allclose(b2, bgold[rows,0])


if __name__ == "__main__":
    test_move_linsys_to_rank_zero()