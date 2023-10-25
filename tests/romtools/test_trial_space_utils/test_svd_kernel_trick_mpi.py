import romtools as rt
import romtools.trial_space_utils as utils
from helper_scripts import helpers
import copy
import numpy as np
import pytest

try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


def construct_distributed_data(comm):
    rank = comm.Get_rank()
    np.random.seed(rank)
    if rank == 0:
        myData = np.random.normal(size=(9,5))
    elif rank==1:
        myData = np.random.normal(size=(15,5))
    else:
        myData = np.random.normal(size=(9,5))
    return myData

def construct_full_data():
    np.random.seed(0)
    myDataOne = np.random.normal(size=(9,5))
    np.random.seed(1)
    myDataTwo = np.random.normal(size=(15,5))
    np.random.seed(2)
    myDataThree = np.random.normal(size=(9,5))
    myData = np.append(myDataOne,myDataTwo,axis=0)
    myData = np.append(myData,myDataThree,axis=0)
    return myData

@pytest.mark.mpi(min_size=3)
def test_parallel_kernel_trick_on_three_cores():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() == 3:
        data = construct_distributed_data(comm)
        data_full = construct_full_data()
        U,sigma = utils.svdMethodOfSnapshotsImpl(data,comm)
        Uf,sigmaf,_ = np.linalg.svd(data_full,full_matrices=False)

        if rank == 0:
          assert(np.allclose(sigma,sigmaf))
          ## Use absolute value due to non-uniqueness of sign
          assert(np.allclose(np.abs(U),np.abs(Uf)[0:9]))
        if rank == 1:
          assert(np.allclose(sigma,sigmaf))
          assert(np.allclose(np.abs(U),np.abs(Uf)[9:24]))
        if rank == 2:
          assert(np.allclose(sigma,sigmaf))
          assert(np.allclose(np.abs(U),np.abs(Uf)[24::]))
    else:
        helpers.mpi_skipped_test_mismatching_commsize(comm, "test_parallel_kernel_trick_on_three_cores", 3)

@pytest.mark.mpi(min_size=3)
def test_class_parallel_kernel_trick_on_three_cores():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() == 3:
        data = construct_distributed_data(comm)
        data_full = construct_full_data()
        mySvd = utils.svdMethodOfSnapshots(comm)
        U,sigma,_ = mySvd(data,full_matrices=False)
        Uf,sigmaf,_ = np.linalg.svd(data_full,full_matrices=False)

        if rank == 0:
          assert(np.allclose(sigma,sigmaf))
          ## Use absolute value due to non-uniqueness of sign
          assert(np.allclose(np.abs(U),np.abs(Uf)[0:9]))
        if rank == 1:
          assert(np.allclose(sigma,sigmaf))
          assert(np.allclose(np.abs(U),np.abs(Uf)[9:24]))
        if rank == 2:
          assert(np.allclose(sigma,sigmaf))
          assert(np.allclose(np.abs(U),np.abs(Uf)[24::]))
    else:
        helpers.mpi_skipped_test_mismatching_commsize(comm, "test_class_parallel_kernel_trick_on_three_cores", 3)


if __name__=="__main__":
    test_parallel_kernel_trick_on_three_cores()
    test_class_parallel_kernel_trick_on_three_cores()
