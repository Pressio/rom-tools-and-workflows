import numpy as np

import pytest
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl,\
                                           generate_local_and_global_arrays_from_example_impl
from romtools.linalg.linalg import _basic_max_via_python
from romtools.linalg.linalg import _basic_min_via_python


########################
###  Set up problem  ###
########################

def _min_max_setup(operation, ndim, axis=None, comm=None):
    shape = (7,5,6)
    local_arr, global_arr = generate_random_local_and_global_arrays_impl(shape[:ndim], comm)

    if operation == "min":
        min_result = _basic_min_via_python(local_arr, comm=comm)
        return min_result, np.min(global_arr)
    elif operation == "max":
        max_result = _basic_max_via_python(local_arr, axis=axis, comm=comm)
        return max_result, np.max(global_arr, axis=axis)
    else:
        return None, max(global_arr)


########################
###   Define Tests   ###
########################

@pytest.mark.mpi(min_size=3)
def test_python_max_examples_mpi():
    '''Specifically tests the documented examples in _basic_max_via_python.'''
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 3:
        rank = comm.Get_rank()
        slices = [(0,2), (2,6), (6,7)]

        # Example 1
        local_arr_1, global_arr_1 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=1)

        res_ex1 = _basic_max_via_python(local_arr_1, comm=comm)
        assert res_ex1 == np.max(global_arr_1)

        # Example 2
        local_arr_2, global_arr_2 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=2)

        res_ex2_ax0 = _basic_max_via_python(local_arr_2, axis=0, comm=comm)
        exp_ex2_ax0 = np.max(global_arr_2, axis=0)
        assert np.allclose(res_ex2_ax0, exp_ex2_ax0)

        res_ex2_ax1 = _basic_max_via_python(local_arr_2, axis=1, comm=comm)
        full_ex2_ax1_max = np.max(global_arr_2, axis=1)
        exp_ex2_ax1 = full_ex2_ax1_max[slices[rank][0]:slices[rank][1]]
        assert np.allclose(res_ex2_ax1, exp_ex2_ax1)

        # Example 3
        local_arr_3, global_arr_3 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=3)
        res_ex3_ax0 = _basic_max_via_python(local_arr_3, axis=0, comm=comm)
        full_ex3_ax0_max = np.max(global_arr_3, axis=0)
        exp_ex3_ax0 = full_ex3_ax0_max[slices[rank][0]:slices[rank][1],:]
        assert np.allclose(res_ex3_ax0, exp_ex3_ax0)

        res_ex3_ax1 = _basic_max_via_python(local_arr_3, axis=1, comm=comm)
        exp_ex3_ax1 = np.max(global_arr_3, axis=1)
        assert np.allclose(res_ex3_ax1, exp_ex3_ax1)

        res_ex3_ax2 = _basic_max_via_python(local_arr_3, axis=2, comm=comm)
        full_ex3_ax2_max = np.max(global_arr_3, axis=2)
        exp_ex3_ax2 = full_ex3_ax2_max[:,slices[rank][0]:slices[rank][1]]
        assert np.allclose(res_ex3_ax2, exp_ex3_ax2)

@pytest.mark.mpi(min_size=3)
def test_python_min_examples_mpi():
    '''Specifically tests the documented examples in _basic_min_via_python.'''
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 3:
        rank = comm.Get_rank()
        slices = [(0,2), (2,6), (6,7)]

        # Example 1
        local_arr_1, global_arr_1 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=1)

        res_ex1 = _basic_min_via_python(local_arr_1, comm=comm)
        assert res_ex1 == np.min(global_arr_1)

        # Example 2
        local_arr_2, global_arr_2 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=2)

        res_ex2_ax0 = _basic_min_via_python(local_arr_2, axis=0, comm=comm)
        exp_ex2_ax0 = np.min(global_arr_2, axis=0)
        assert np.allclose(res_ex2_ax0, exp_ex2_ax0)

        res_ex2_ax1 = _basic_min_via_python(local_arr_2, axis=1, comm=comm)
        full_ex2_ax1_min = np.min(global_arr_2, axis=1)
        exp_ex2_ax1 = full_ex2_ax1_min[slices[rank][0]:slices[rank][1]]
        assert np.allclose(res_ex2_ax1, exp_ex2_ax1)

        # Example 3
        local_arr_3, global_arr_3 = generate_local_and_global_arrays_from_example_impl(rank, slices, example=3)
        res_ex3_ax0 = _basic_min_via_python(local_arr_3, axis=0, comm=comm)
        full_ex3_ax0_min = np.min(global_arr_3, axis=0)
        exp_ex3_ax0 = full_ex3_ax0_min[slices[rank][0]:slices[rank][1],:]
        assert np.allclose(res_ex3_ax0, exp_ex3_ax0)

        res_ex3_ax1 = _basic_min_via_python(local_arr_3, axis=1, comm=comm)
        exp_ex3_ax1 = np.min(global_arr_3, axis=1)
        assert np.allclose(res_ex3_ax1, exp_ex3_ax1)

        res_ex3_ax2 = _basic_min_via_python(local_arr_3, axis=2, comm=comm)
        full_ex3_ax2_min = np.min(global_arr_3, axis=2)
        exp_ex3_ax2 = full_ex3_ax2_min[:,slices[rank][0]:slices[rank][1]]
        assert np.allclose(res_ex3_ax2, exp_ex3_ax2)

@pytest.mark.mpi(min_size=3)
def test_python_max_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected = _min_max_setup(operation="max", ndim=1, comm=comm)
    assert result == expected

@pytest.mark.mpi(min_size=3)
def test_python_min_vector_mpi():
    comm = MPI.COMM_WORLD
    result, expected_min = _min_max_setup(operation="min", ndim=1, comm=comm)
    assert result == expected_min

@pytest.mark.mpi(min_size=3)
def test_python_max_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="max", ndim=2, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _min_max_setup(operation="max", ndim=3, comm=comm)
    assert np.allclose(result_02, expected_02)

@pytest.mark.mpi(min_size=3)
def test_python_max_on_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="max", ndim=2, axis=0, comm=comm)
    assert np.allclose(result_01, expected_01)

    # Make sure the result is a subset of the full max along the axis
    result_02, expected_02 = _min_max_setup(operation="max", ndim=3, axis=1, comm=comm)
    assert len(np.setdiff1d(result_02, expected_02)) == 0


@pytest.mark.mpi(min_size=3)
def test_python_min_on_axis_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="min", ndim=2, axis=0, comm=comm)
    assert np.allclose(result_01, expected_01)

    # Make sure the result is a subset of the full min along the axis
    result_02, expected_02 = _min_max_setup(operation="min", ndim=3, axis=1, comm=comm)
    assert len(np.setdiff1d(result_02, expected_02)) == 0

    result_03, expected_03 = _min_max_setup(operation="min", ndim=3, axis=(1,2), comm=comm)
    assert len(np.setdiff1d(result_03, expected_03)) == 0


@pytest.mark.mpi(min_size=3)
def test_python_min_array_mpi():
    comm = MPI.COMM_WORLD
    result_01, expected_01 = _min_max_setup(operation="min", ndim=2, comm=comm)
    assert np.allclose(result_01, expected_01)

    result_02, expected_02 = _min_max_setup(operation="min", ndim=3, comm=comm)
    assert np.allclose(result_02, expected_02)


def test_python_max_serial():
    vector = np.random.rand(10)
    assert _basic_max_via_python(vector) == np.max(vector)

    array = np.random.rand(3, 10)
    assert _basic_max_via_python(array) == np.max(array)

def test_python_min_serial():
    vector = np.random.rand(10)
    assert _basic_min_via_python(vector) == np.min(vector)

    array = np.random.rand(3, 10)
    assert _basic_min_via_python(array) == np.min(array)


if __name__ == "__main__":
    test_python_max_examples_mpi()
    test_python_min_examples_mpi()
    test_python_max_vector_mpi()
    test_python_min_vector_mpi()
    test_python_max_array_mpi()
    test_python_min_array_mpi()
    test_python_max_on_axis_mpi()
    test_python_min_on_axis_mpi()
    test_python_max_serial()
    test_python_min_serial()
