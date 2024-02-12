import pytest
import numpy as np
import romtools as rt
import pressiolinalg.linalg as pla

def test_pressiolinalg_serial():
    test_arr = np.array([1,2,3,4,5,6])

    np_max = np.max(test_arr)
    pla_max = pla.max(test_arr)
    assert(np_max == pla_max)

    np_min = np.min(test_arr)
    pla_min = pla.min(test_arr)
    assert(np_min == pla_min)

# Functionality is tested in pressio-linalg; this is just checking that MPI works
@pytest.mark.mpi(min_size=3)
def test_pressiolinalg_parallel():
    import mpi4py
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    global_vector = np.arange(0,size*2)
    local_vector = np.zeros(2)
    comm.Scatter(global_vector, local_vector, root=0)

    dist_min = pla.min(local_vector, comm)
    serial_min = np.min(global_vector)

    assert dist_min == serial_min

if __name__ == "__main__":
    test_pressiolinalg_serial()
    test_pressiolinalg_parallel()
