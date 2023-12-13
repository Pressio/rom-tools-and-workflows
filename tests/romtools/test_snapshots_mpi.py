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

    def getSnapshotTensor(self):
        return self.snapshots

    def getMeshGids(self):
        return myGIDs



@pytest.mark.mpi(min_size=3)
def test_mpi_snapshots():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() == 3:
        if rank == 0:
            myGids = np.array([0,1,2])
            myData = np.random.normal(size=(3,3,5))
        elif rank==1:
            myGids = np.array([6,7,8,9,10])
            myData = np.random.normal(size=(3,5,5))
        else:
            myGids = np.array([3,4,5])
            myData = np.random.normal(size=(3,3,5))

        sd = DistributedSnapshots(myData, myGids)
        tensor = sd.getSnapshotTensor()
        assert tensor.shape[0] == 3 
        assert tensor.shape[1] == myGids.shape[0] 
        assert tensor.shape[2] == 5

if __name__=="__main__":
    test_mpi_snapshots()
