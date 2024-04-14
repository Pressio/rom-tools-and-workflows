import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.parallel_utils import distribute_array_impl
from romtools.linalg.linalg import _thin_svd


########################
###  Set up problem  ###
########################

def create_snapshots(comm):
    num_processes = comm.Get_size()
    global_snapshots = np.array([np.arange(0, num_processes)]).T
    local_snapshots = distribute_array_impl(global_snapshots, comm)
    return global_snapshots, local_snapshots

def get_serial_solution(snapshots):
    gram_matrix = np.dot(snapshots.T, snapshots)
    eigenvalues, eigenvectors = np.linalg.eig(gram_matrix)
    sigma = np.sqrt(eigenvalues)
    modes = np.zeros(np.shape(snapshots))
    modes[:] = np.dot(snapshots, np.dot(eigenvectors, np.diag(1./sigma)))
    ordering = np.argsort(sigma)[::-1]
    return modes[:, ordering], sigma[ordering]

########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_basic_svd_method_of_snapshots_impl_via_python():
    # Solve in parallel
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_processes = comm.Get_size()
    global_snapshots, local_snapshots = create_snapshots(comm)
    local_modes, mpi_sigma = _thin_svd(local_snapshots, comm, method='method_of_snapshots')

    # Get serial solution
    test_modes, test_sigma = get_serial_solution(global_snapshots)

    # Compare values
    assert np.allclose(local_modes, test_modes[rank])
    assert mpi_sigma == test_sigma

def test_basic_svd_serial():
    snapshots = np.array([np.arange(0, 3)]).transpose()
    modes, sigma = _thin_svd(snapshots, method='method_of_snapshots')
    test_modes, test_sigma = get_serial_solution(snapshots)
    assert np.allclose(modes, test_modes)
    assert sigma == test_sigma


if __name__ == "__main__":
    test_basic_svd_method_of_snapshots_impl_via_python()
    test_basic_svd_serial()
