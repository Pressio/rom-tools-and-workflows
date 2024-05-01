import numpy as np
import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl,\
                                           generate_local_and_global_arrays_from_example_impl
from romtools.linalg.linalg import _basic_argmax_via_python


# ------------------------------------------------------------------------------

def _parallel_argmax_test(ndim, comm):
    shape = (7,5,6)[:ndim]

    rank = comm.Get_rank()
    local_arr, global_arr = generate_random_local_and_global_arrays_impl(shape, comm)

    linalg_result = _basic_argmax_via_python(local_arr, comm=comm)

    numpy_max_index = np.argmax(global_arr)
    numpy_max_val = global_arr.ravel()[numpy_max_index]

    linalg_max_index = linalg_result[0]
    rank_with_max = linalg_result[1]

    if rank == rank_with_max:
        assert local_arr.ravel()[linalg_max_index] == numpy_max_val

def _serial_argmax_test(ndim):
    shape = (7,5,6)[:ndim]

    array = np.random.rand(*shape)

    linalg_result = _basic_argmax_via_python(array)
    numpy_result = np.argmax(array)

    assert linalg_result == numpy_result

# ------------------------------------------------------------------------------


@pytest.mark.mpi(min_size=3)
def test_argmax_mpi():
    comm = MPI.COMM_WORLD
    for i in range(1, 4):
        _parallel_argmax_test(ndim=i, comm=comm)

@pytest.mark.mpi_skip
def test_argmax_serial():
    for i in range(1, 4):
        _serial_argmax_test(ndim=i)

@pytest.mark.mpi(min_size=3)
def test_parallel_argmax_examples():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 3:
        rank = comm.Get_rank()
        slices = [(0,2), (2,6), (6,7)]

        # Example 1
        local_arr_1, global_arr_1 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=1)
        res_ex_1 = _basic_argmax_via_python(local_arr_1, comm=comm)
        assert res_ex_1 == (1, 1)

        # Example 2
        local_arr_2, global_arr_2 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=2)
        res_ex_2 = _basic_argmax_via_python(local_arr_2, comm=comm)
        assert res_ex_2 == (3, 1)

        # Example 3
        local_arr_2, global_arr_2 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=3)
        res_ex_2 = _basic_argmax_via_python(local_arr_2, comm=comm)
        assert res_ex_2 == (20, 1)


if __name__ == "__main__":
    test_argmax_mpi()
    test_argmax_serial()
    test_parallel_argmax_examples()