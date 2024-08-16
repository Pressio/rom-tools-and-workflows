import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl,\
                                           generate_local_and_global_arrays_from_example_impl
from romtools.linalg.linalg import _basic_std_via_python


########################
###  Set up problem  ###
########################

def _std_setup(ndim, dtype=None, axis=None, comm=None):
    n_procs = comm.Get_size()
    shape = (7,5,6)
    local_arr, global_arr = generate_random_local_and_global_arrays_impl(shape[:ndim], comm)

    std_result = _basic_std_via_python(local_arr, dtype=dtype, axis=axis, comm=comm)
    return std_result, np.std(global_arr, dtype=dtype, axis=axis)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_std_examples_mpi():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 3:
        rank = comm.Get_rank()
        slices = [(0,2), (2,6), (6,7)]

        # Example 1
        local_arr_1, global_arr_1 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=1)
        res_ex1 = _basic_std_via_python(local_arr_1, comm=comm)
        np.testing.assert_almost_equal(res_ex1, np.std(global_arr_1), decimal=10)

        # Example 2
        local_arr_2, global_arr_2 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=2)

        res_ex2_ax0 = _basic_std_via_python(local_arr_2, axis=0, comm=comm)
        exp_ex2_ax0 = np.std(global_arr_2, axis=0)
        assert np.allclose(res_ex2_ax0, exp_ex2_ax0)

        res_ex2_ax1 = _basic_std_via_python(local_arr_2, axis=1, comm=comm)
        full_ex2_ax1_std = np.std(global_arr_2, axis=1)
        exp_ex2_ax1 = full_ex2_ax1_std[slices[rank][0]:slices[rank][1]]
        assert np.allclose(res_ex2_ax1, exp_ex2_ax1)

        # Example 3
        local_arr_3, global_arr_3 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=3)
        res_ex3_ax0 = _basic_std_via_python(local_arr_3, axis=0, comm=comm)
        full_ex3_ax0_std = np.std(global_arr_3, axis=0)
        exp_ex3_ax0 = full_ex3_ax0_std[slices[rank][0]:slices[rank][1],:]
        assert np.allclose(res_ex3_ax0, exp_ex3_ax0)

        res_ex3_ax1 = _basic_std_via_python(local_arr_3, axis=1, comm=comm)
        exp_ex3_ax1 = np.std(global_arr_3, axis=1)
        assert np.allclose(res_ex3_ax1, exp_ex3_ax1)

        res_ex3_ax2 = _basic_std_via_python(local_arr_3, axis=2, comm=comm)
        full_ex3_ax2_std = np.std(global_arr_3, axis=2)
        exp_ex3_ax2 = full_ex3_ax2_std[:,slices[rank][0]:slices[rank][1]]
        assert np.allclose(res_ex3_ax2, exp_ex3_ax2)

@pytest.mark.mpi(min_size=3)
def test_python_std_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _std_setup(ndim=1, comm=comm)
    np.testing.assert_almost_equal(result, expected, decimal=10)

@pytest.mark.mpi(min_size=3)
def test_python_std_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _std_setup(ndim=2, dtype=np.float32, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _std_setup(ndim=3, comm=comm)
    assert np.allclose(result_02, expected_02)

@pytest.mark.mpi(min_size=3)
def test_python_std_array_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _std_setup(ndim=2, dtype=np.float32, axis=0, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _std_setup(ndim=2, dtype=np.float32, axis=1, comm=comm)
    assert len(np.setdiff1d(result_02, expected_02)) == 0

    result_03, expected_03 = _std_setup(ndim=3, axis=2, comm=comm)
    assert len(np.setdiff1d(result_03, expected_03)) == 0

def test_python_std_serial():
    vector = np.random.rand(10)
    np.testing.assert_almost_equal(_basic_std_via_python(vector), np.std(vector), decimal=10)

    array = np.random.rand(3, 10)
    np.testing.assert_almost_equal(_basic_std_via_python(array), np.std(array), decimal=10)


if __name__ == "__main__":
    test_python_std_examples_mpi()
    test_python_std_vector_mpi()
    test_python_std_array_mpi()
    test_python_std_array_axis_mpi()
    test_python_std_serial()
