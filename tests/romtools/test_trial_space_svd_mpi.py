import pytest
import numpy as np
import romtools as rt
from helper_scripts import helpers
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


class DistributedSnapshots(rt.AbstractSnapshotData):
    def __init__(self, myData, myGIDs):
        self.snapshots = myData

    def getSnapshotsAsListOfArrays(self):
        return self.snapshots

    def getMeshGids(self):
        return myGIDs

    def getVariableNames(self):
        return ['u','v','w']

    def getNumVars(self) -> int:
        return 3

def construct_snapshots(comm):
    rank = comm.Get_rank()
    if rank == 0:
        myGids = np.array([0,1,2])
        myData = [np.random.normal(size=(9,5))]
    elif rank==1:
        myGids = np.array([6,7,8,9,10])
        myData = [np.random.normal(size=(15,5))]
    else:
        myGids = np.array([3,4,5])
        myData = [np.random.normal(size=(9,5))]

    return DistributedSnapshots(myData, myGids)

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
def test_trial_space_from_pod_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() == 3:
        snaps = construct_snapshots(comm)
        myTrialSpace = rt.TrialSpaceFromPOD(snaps, svdFnc=MyFakeSvd(comm))
        U = myTrialSpace.getBasis()
        k = myTrialSpace.getDimension()
        if rank == 0:
            assert np.allclose(U, np.zeros((9,2)))
            assert np.allclose(2, k)
        elif rank==1:
            assert(np.allclose(U, np.ones((15,2))))
            assert np.allclose(2, k)
        elif rank==2:
            assert(np.allclose(U, np.ones((9,2))*2))
            assert np.allclose(2, k)
    else:
        helpers.mpi_skipped_test_mismatching_commsize(comm, "test_trial_space_from_pod_mpi", 3)

if __name__=="__main__":
    test_trial_space_from_pod_mpi()
