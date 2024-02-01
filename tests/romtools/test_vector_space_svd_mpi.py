import pytest
import numpy as np
import romtools as rt
from helper_scripts import helpers
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


def construct_snapshots(comm):
    rank = comm.Get_rank()
    if rank == 0:
        myData = np.random.normal(size=(3, 3, 5))
    elif rank==1:
        myData = np.random.normal(size=(3, 5, 5))
    else:
        myData = np.random.normal(size=(3, 3, 5))

    return myData 

class MyFakeSvd:
    def __init__(self, comm):
        self.comm_ = comm

    def __call__(self, A, full_matrices=False, compute_uv=True, hermitian=False):
        # this is totally fake, just for testing

        rank = self.comm_.Get_rank()
        nr, nc = A.shape[0], A.shape[1]
        lsv = np.ones((nr, 2))*rank
        svals = np.ones(2)*rank
        return lsv, svals, None


@pytest.mark.mpi(min_size=3)
def test_vector_space_from_pod_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() == 3:
        snaps = construct_snapshots(comm)
        myVectorSpace = rt.VectorSpaceFromPOD(snaps, svdFnc=MyFakeSvd(comm))
        U = myVectorSpace.get_basis()
        k = myVectorSpace.get_dimension()
        if rank == 0:
            assert np.allclose(U, np.zeros((3, 3, 2)))
            assert np.allclose(2, k)
        elif rank == 1:
            assert np.allclose(U, np.ones((3, 5, 2)))
            assert np.allclose(2, k)
        elif rank == 2:
            assert np.allclose(U, np.ones((3, 3, 2))*2)
            assert np.allclose(2, k)
    else:
        helpers.mpi_skipped_test_mismatching_commsize(comm, "test_vector_space_from_pod_mpi", 3)

if __name__ == "__main__":
    test_vector_space_from_pod_mpi()
