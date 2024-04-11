import math
import warnings
import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.parallel_utils import *
from romtools.linalg.linalg import _basic_mean_via_python


########################
###  Set up problem  ###
########################

def _mean_setup(ndim, dtype=None, axis=None, comm=None):
    shape = (7,5,6)
    local_arr, global_arr = generate_random_local_and_global_arrays_impl(shape[:ndim], comm)
    mean_result = _basic_mean_via_python(local_arr, dtype=dtype, axis=axis, comm=comm)
    return mean_result, np.mean(global_arr, dtype=dtype, axis=axis)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_mean_examples_mpi():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 3:
        rank = comm.Get_rank()
        slices = [(0,2), (2,6), (6,7)]

        # Example 1
        local_arr_1, global_arr_1 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=1)

        res_ex1 = _basic_mean_via_python(local_arr_1, comm=comm)
        assert res_ex1 == np.mean(global_arr_1)

        # Example 2
        local_arr_2, global_arr_2 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=2)

        res_ex2_ax0 = _basic_mean_via_python(local_arr_2, axis=0, comm=comm)
        exp_ex2_ax0 = np.mean(global_arr_2, axis=0)
        assert np.allclose(res_ex2_ax0, exp_ex2_ax0)

        res_ex2_ax1 = _basic_mean_via_python(local_arr_2, axis=1, comm=comm)
        full_ex2_ax1_mean = np.mean(global_arr_2, axis=1)
        exp_ex2_ax1 = full_ex2_ax1_mean[slices[rank][0]:slices[rank][1]]
        assert np.allclose(res_ex2_ax1, exp_ex2_ax1)

        # Example 3
        local_arr_3, global_arr_3 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=3)
        res_ex3_ax0 = _basic_mean_via_python(local_arr_3, axis=0, comm=comm)
        full_ex3_ax0_mean = np.mean(global_arr_3, axis=0)
        exp_ex3_ax0 = full_ex3_ax0_mean[slices[rank][0]:slices[rank][1],:]
        assert np.allclose(res_ex3_ax0, exp_ex3_ax0)

        res_ex3_ax1 = _basic_mean_via_python(local_arr_3, axis=1, comm=comm)
        exp_ex3_ax1 = np.mean(global_arr_3, axis=1)
        assert np.allclose(res_ex3_ax1, exp_ex3_ax1)

        res_ex3_ax2 = _basic_mean_via_python(local_arr_3, axis=2, comm=comm)
        full_ex3_ax2_mean = np.mean(global_arr_3, axis=2)
        exp_ex3_ax2 = full_ex3_ax2_mean[:,slices[rank][0]:slices[rank][1]]
        assert np.allclose(res_ex3_ax2, exp_ex3_ax2)

@pytest.mark.mpi(min_size=3)
def test_python_mean_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _mean_setup(ndim=1, comm=comm)
    np.testing.assert_almost_equal(result, expected, decimal=10)

@pytest.mark.mpi(min_size=3)
def test_python_mean_null_vector_mpi():
    comm = MPI.COMM_WORLD

    # Both la.mean and np.mean will output warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, expected = _mean_setup(ndim=0, comm=comm)
        assert math.isnan(result)
        assert math.isnan(expected)

@pytest.mark.mpi(min_size=3)
def test_python_mean_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _mean_setup(ndim=2, dtype=np.float32, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _mean_setup(ndim=3, comm=comm)
    assert np.allclose(result_02, expected_02)

@pytest.mark.mpi(min_size=3)
def test_python_mean_array_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _mean_setup(ndim=2, axis=0, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _mean_setup(ndim=2, axis=1, comm=comm)
    assert len(np.setdiff1d(result_02, expected_02)) == 0

@pytest.mark.mpi(min_size=3)
def test_python_mean_tensor_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _mean_setup(ndim=3, axis=0, comm=comm)
    assert len(np.setdiff1d(result_01, expected_01)) == 0

    result_02, expected_02 = _mean_setup(ndim=3, axis=1, comm=comm)
    assert np.allclose(result_02, expected_02)

    result_03, expected_03 = _mean_setup(ndim=3, axis=2, comm=comm)
    assert len(np.setdiff1d(result_03, expected_03)) == 0

def test_python_mean_serial():
    vector = np.random.rand(10)
    np.testing.assert_almost_equal(_basic_mean_via_python(vector), np.mean(vector), decimal=10)

    array = np.random.rand(3, 10)
    np.testing.assert_almost_equal(_basic_mean_via_python(array), np.mean(array), decimal=10)


if __name__ == "__main__":
    test_python_mean_examples_mpi()
    test_python_mean_null_vector_mpi()
    test_python_mean_vector_mpi()
    test_python_mean_array_mpi()
    test_python_mean_array_axis_mpi()
    test_python_mean_tensor_axis_mpi()
    test_python_mean_serial()
