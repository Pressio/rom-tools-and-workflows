import romtools as rt
import copy
import numpy as np
import pytest

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

@pytest.mark.mpi(min_size=3)
def test_mpi_snapshots():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() == 3:
        if rank == 0:
            myGids = np.array([0,1,2])
            myData = [np.random.normal(size=(9,5))]
        elif rank==1:
            myGids = np.array([6,7,8,9,10])
            myData = [np.random.normal(size=(15,5))]
        else:
            myGids = np.array([3,4,5])
            myData = [np.random.normal(size=(9,5))]

        sd = DistributedSnapshots(myData, myGids)
        matrix = sd.getSnapshotsAsArray()
        assert matrix.shape[0] == myGids.shape[0] * sd.getNumVars()
        assert matrix.shape[1] == 5

if __name__=="__main__":
    test_mpi_snapshots()
