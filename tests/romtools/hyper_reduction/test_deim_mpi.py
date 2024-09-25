
import numpy as np
import pytest
from romtools.hyper_reduction import deim
from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl, distribute_array_impl
import romtools.linalg.linalg as la

try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


def _map_local_and_rank_to_global_assuming_indices_increase_over_increasing_ranks(comm, localCount, localIndices, ranks):
    assert len(localIndices) == len(ranks)
    numRanks = comm.Get_size()
    allCounts = comm.allgather(localCount)
    countUntilTargetRank = np.cumsum(allCounts)
    result = []
    for i in range(len(localIndices)):
        index = localIndices[i]
        r = ranks[i]
        gi = countUntilTargetRank[r-1] if r > 0 else 0
        result.append( gi + index )
    return np.array(result)


@pytest.mark.mpi(min_size=3)
def test_deim_indices_mpi():
    np.random.seed(9243)
    comm = MPI.COMM_WORLD
    myRank = comm.Get_rank()

    for numColumns in [1,2,4,8]:
        U_l, U_g = generate_random_local_and_global_arrays_impl((10, numColumns), comm)

        indices_shmem = deim.deim_get_indices(U_g)
        indices, ranks = deim.deim_get_indices(U_l, comm)
        global_indices = _map_local_and_rank_to_global_assuming_indices_increase_over_increasing_ranks(comm, U_l.shape[0], indices, ranks)
        if myRank == 0:
            print("indices_shmem = ", indices_shmem)
            print("global_indices = ", global_indices)
        assert(np.allclose(global_indices,indices_shmem))

@pytest.mark.mpi(min_size=3)
def test_deim_basis_mpi():
    comm = MPI.COMM_WORLD

    if comm.Get_size() != 3:
        return

    rank = comm.Get_rank()

    U_l, U_g = generate_random_local_and_global_arrays_impl((10, 5), comm)
    Phi_l, Phi_g = generate_random_local_and_global_arrays_impl((10, 3), comm)

    distributions = [(0,3), (4,6), (7,9)]

    # Test with an empty rank
    global_indices = np.array([0,1,2,3,8,9])
    local_indices = np.array(
        [idx - distributions[rank][0] for idx in global_indices if distributions[rank][0] <= idx <= distributions[rank][1]]
    )
    deimPhi_g = deim.deim_get_test_basis(Phi_g, U_g, global_indices)
    deimPhi = deim.deim_get_test_basis(Phi_l, U_l, local_indices, comm)
    assert np.allclose(deimPhi, deimPhi_g)

    # Test with all ranks populated
    global_indices = np.array([0,1,5,6,8,9])
    local_indices = np.array(
        [idx - distributions[rank][0] for idx in global_indices if distributions[rank][0] <= idx <= distributions[rank][1]]
    )
    deimPhi_g = deim.deim_get_test_basis(Phi_g, U_g, global_indices)
    deimPhi = deim.deim_get_test_basis(Phi_l, U_l, local_indices, comm)
    assert np.allclose(deimPhi, deimPhi_g)

@pytest.mark.mpi(min_size=3)
def test_multi_state_deim_indices_mpi():
    np.random.seed(9243)
    comm = MPI.COMM_WORLD
    myRank = comm.Get_rank()

    for globalExtent in [10, 13, 17, 49]:
        for numBasis in [2, 4, 7, 8]:
            # by default, the distribution is done along axis=1
            U_l, U_g = generate_random_local_and_global_arrays_impl((3, globalExtent, numBasis), comm)
            indices_shmem = deim.multi_state_deim_get_indices(U_g)
            indices, ranks = deim.multi_state_deim_get_indices(U_l, comm)

            global_indices = _map_local_and_rank_to_global_assuming_indices_increase_over_increasing_ranks(comm, U_l.shape[1], indices, ranks)
            assert(np.allclose(global_indices, indices_shmem))

if __name__ == "__main__":
    test_deim_indices_mpi()
    test_deim_basis_mpi()
    test_multi_state_deim_indices_mpi()
