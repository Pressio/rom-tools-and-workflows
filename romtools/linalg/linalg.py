
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

from pathlib import Path

import builtins
import warnings
import numpy as np
from romtools.linalg.parallel_utils import assert_axis_is_none_or_within_rank

# ----------------------------------------------------

def _basic_max_via_python(a: np.ndarray, axis=None, comm=None):
    '''
    Return the maximum of a possibly distributed array or maximum along an axis.

    Parameters:
        a (np.ndarray): input data
        axis (None or int): the axis along which to compute the maximum.
            If None, computes the max of the flattened array. (default: None)
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        - if axis == None, returns a scalar
        - if axis is not None, returns an array of dimension a.ndim - 1

    Preconditions:
        - a is at most a rank-3 tensor
        - if a is a distributed 2-D array, it must be distributed along axis=0,
          and every rank must have the same a.shape[1]
        - if a is a distributed 3-D tensor, it must be distributed along axis=1,
          and every rank must have the same a.shape[0] and a.shape[2]
        - if axis != None, then it must be an int

    Postconditions:
        - a and comm are not modified

    Example 1:
    ^^^^^^^^^^

    Input data:

    .. code-block:: text

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    Suppose that we do:

    .. code-block:: python

        res = la.max(a, comm)

    Then all ranks will contain res = 51.

    Example 2:
    ^^^^^^^^^^

    .. code-block:: text

       rank 0  2.2  1.3  4.
               3.3  5.0  33.
      =======================
       rank 1  40.  -2.  -4.
               51.   4.   6.
               -24.  8.   9.
               45.  -3.  -4.
      =======================
       rank 2  -4.  8.   9.

    Suppose that we do:

    .. code-block:: python

       res = la.max(a, axis=0, comm)

    Then every rank will contain the same res which is an array = ([51., 8., 33])
    this is because the max is queried for the 0-th axis which is the
    axis along which the data array is distributed. So this operation must be a
    collective operation.

    Suppose that we do:

    .. code-block:: python

      res = la.max(a, axis=1, comm)

    Then res is now a rank-1 array as follows:

    .. code-block:: text

        rank 0  4.
                33.
        =======================
        rank 1  40.
                51.
                9.
                45.
        =======================
        rank 2  9.

    Because the axis queried for the max is NOT a distributed axis so this
    operation is purely local and the result has the same distribution as the
    original array.

    Example 3:
    ^^^^^^^^^^

    .. code-block:: text

           / 3.   4.   /  2.   8.   2.   1.   / 2.
          /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
         /  -7.   5. /    5.   0.   3.   1. /   3.
        |-----------|----------------------|--------
        | 2.   3.   |  4.   5.  -2.   4.   | -4.
        | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
        | 4.   3.   | -4.   6.   9.  -4.   |  9.

            r0                r1              r2

    Suppose that we do:

    .. code-block:: python

        res = la.max(a, axis=0, comm)

    Then res is now a rank-2 array as follows:

    .. code-block:: text

           /  6.  5.   /  5.   8.   3.   1.  /  3.
          / 4.   5.   / 4.   6.   9.   4.   /  9.
         /           /                     /
        /    r1     /         r2          /  r3

    Because the axis queried for the max is NOT a distributed axis and this is
    effectively a reduction over the 0-th axis so this operation is purely local
    and the result has the same distribution as the original array.

    Suppose that we do:

    .. code-block:: python

        res = la.max(a, axis=1, comm)

    Then this is effectively a reduction over axis=1, and every rank will
    contain the same res which is a rank-2 array as follows:

    .. code-block:: text

        5.  8.
        8.  6.
        9.  5.

    This is because the max is queried for the 0-th axis which is the axis along
    which the data array is distributed. So this operation must be a collective
    operation and we know that memory-wise it is feasible to hold because this
    is no larger than the local allocation on each rank.

    Suppose that we do:

    .. code-block:: python

        res = la.max(a, axis=2, comm)

    Then res is now a rank-2 array as follows:

    .. code-block:: text

          r0     ||          r1           ||  r2
                 ||                       ||
        3.   4.  ||   4.   8.   2.   4.   ||   2.
        6.   5.  ||  -2.   4.   8.  -3.   ||   8.
        4.   5.  ||   5.   6.   9.   1.   ||   9.
                 ||                       ||

    Because the axis queried for the max is NOT a distributed axis and this is
    effectively a reduction over the 2-th axis so this operation is purely local
    and the result has the same distribution as the original array.

    '''
    # Enforce preconditions
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"
    assert_axis_is_none_or_within_rank(a, axis)

    # Return np.max if running serial
    if comm is None or comm.Get_size() == 1:
        return np.max(a, axis=axis)

    # Otherwise, calculate distributed max
    from mpi4py import MPI

    # Get the max on the current process
    local_max = np.max(a, axis=axis)

    # Identify the axis along which the data is the distributed
    distributed_axis = 0 if a.ndim < 3 else 1

    # Return the max of the flattened array if no axis is given
    if axis is None:
        return comm.allreduce(local_max, op=MPI.MAX)

    # If queried axis is the same as distributed axis, perform collective operation
    if axis==distributed_axis:
        if a.ndim == 1:
            local_max = a
        global_max = np.zeros_like(local_max, dtype=local_max.dtype)
        comm.Allreduce(local_max, global_max, op=MPI.MAX)
        return global_max

    # Otherwise, return the local_max on the current process
    return local_max


# ----------------------------------------------------
def _basic_argmax_via_python(a: np.ndarray, comm=None):
    '''
    Return the index of an array's maximum value. If the array is distributed, also returns the
    value itself and the MPI rank on which it occurs.

    Parameters:
        a (np.ndarray): input data
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        - if comm == None, returns the index of the maximum value (identical to np.argmax)
        - if comm != None, returns a tuple containing (value, index, rank):
            - value: the global maximum
            - index: the local index of the global maximum
            - rank: the rank on which the global maximum resides

    Preconditions:
      - a is at most a rank-3 tensor
      - if a is a distributed 2-D array, it must be distributed along axis=0,
        and every rank must have the same a.shape[1]
      - if a is a distributed 3-D tensor, it must be distributed along axis=1,
        and every rank must have the same a.shape[0] and a.shape[2]

    Postconditions:
      - a and comm are not modified

    Example 1:
    ^^^^^^^^^^

    .. code-block:: text

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    Suppose that we do:

    .. code-block:: python

        res = la.argmax(a, comm)

    then ALL ranks will contain res = (1, 1).
    (The global maximum (51.) occurs at index 1 of the local array on Rank 1.)

    Example 2:
    ^^^^^^^^^^

    .. code-block:: text

       rank 0  2.2  1.3  4.
               3.3  5.0  33.
      =======================
       rank 1  40.  -2.  -4.
               51.   4.   6.
               -24.  8.   9.
               45.  -3.  -4.
      =======================
       rank 2  -4.  8.   9.

    Suppose that we do:

    .. code-block:: python

       res = la.argmax(a, comm)

    then ALL ranks will contain res = (3, 1)
    (The global maximum (51.) occurs at index 3 of the flattened local array on Rank 1.)

    Example 3:
    ^^^^^^^^^^

    .. code-block:: text

           / 3.   4.   /  2.   8.   2.   1.   / 2.
          /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
         /  -7.   5. /    5.   0.   3.   1. /   3.
        |-----------|----------------------|--------
        | 2.   3.   |  4.   5.  -2.   4.   | -4.
        | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
        | 4.   3.   | -4.   6.   9.  -4.   |  9.

            r0                r1              r2

    Suppose that we do:

    .. code-block:: python

        res = la.argmax(a, comm)

    then ALL ranks will contain res = (20, 1)
    (The global maximum (9.) occurs on both Rank 1 and Rank 2, but we automatically return the
    index on the lowest rank. In this case, that is index 20 of the flattened local array on Rank 1.)

    '''
    # Enforce preconditions
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"

    # Return "local" result if not running distributed
    if comm is None or comm.Get_size() == 1:
        return np.argmax(a)

    # Get local array argmax result
    local_max_index = np.argmax(a)
    local_max_val = a.ravel()[local_max_index]

    # Set up local solution
    tmp = np.zeros(3)
    tmp[0] = local_max_val
    tmp[1] = local_max_index
    tmp[2] = comm.Get_rank() if comm is not None else 0

    # Define custom MPI op to find distributed max index
    from mpi4py import MPI
    def mycomp(A_mem,B_mem,dt): # pylint: disable=unused-argument
        A = np.frombuffer(A_mem)
        B = np.frombuffer(B_mem)

        # Return the index of the max (or the max on the lowest rank, if multiple occurrences)
        if A[0] < B[0] or (A[0] == B[0] and A[2] > B[2]):
            result = B
        else:
            result = A

        # Copy result to B for next comparison
        B[:] = result

    # Perform operation
    result = np.zeros(3)
    myop = MPI.Op.Create(mycomp, commute=False)
    comm.Allreduce(tmp, result, op=myop)
    myop.Free()

    # Return index (int64), and rank (int)
    return np.int64(result[1]), int(result[2])


# # ----------------------------------------------------
def _basic_min_via_python(a: np.ndarray, axis=None, comm=None):
    '''
    Return the minimum of a possibly distributed array or minimum along an axis.

    Parameters:
        a (np.ndarray): input data
        axis (None or int): the axis along which to compute the minimum. If None, computes the min of the flattened array.
        comm (MPI_Comm): MPI communicator

    Returns:
        - if axis == None, returns a scalar
        - if axis is not None, returns an array of dimension a.ndim - 1

    Preconditions:
        - a is at most a rank-3 tensor
        - if a is a distributed 2-D array, it must be distributed along axis=0,
          and every rank must have the same a.shape[1]
        - if a is a distributed 3-D tensor, it must be distributed along axis=1,
          and every rank must have the same a.shape[0] and a.shape[2]
        - if axis != None, then it must be an int

    Postconditions:
        - a and comm are not modified

    Example 1:
    ^^^^^^^^^^

    .. code-block:: text

        rank 0  2.2
                3.3
        =======================
        rank 1  40.
                51.
                -24.
                45.
        =======================
        rank 2  -4.

    .. code-block:: python

        res = la.min(a, comm)

    then ALL ranks will contain res = -4.

    Example 2:
    ^^^^^^^^^^

    .. code-block:: text

        rank 0  2.2  1.3  4.
                3.3  5.0  33.
        =======================
        rank 1  40.  -2.  -4.
                51.   4.   6.
                -24.  8.   9.
                45.  -3.  -4.
        =======================
        rank 2  -4.  8.   9.

    Suppose that we do:

    .. code-block:: python

       res = la.min(a, axis=0, comm)

    then every rank will contain the same res which is an array = ([-24., -3., -4])
    this is because the min is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

    .. code-block:: python

      res = la.min(a, axis=1, comm)

    then res is now a rank-1 array as follows

    .. code-block:: text

        rank 0  1.3
                3.3
        =======================
        rank 1  -4.
                4.
                -24.
                -4.
        =======================
        rank 2  -4.

    because the axis queried for the min is NOT a distributed axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Example 3:
    ^^^^^^^^^^

    .. code-block:: text

           / 3.   4.   /  2.   8.   2.   1.   / 2.
          /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
         /  -7.   5. /    5.   0.   3.   1. /   3.
        |-----------|----------------------|--------
        | 2.   3.   |  4.   5.  -2.   4.   | -4.
        | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
        | 4.   3.   | -4.   6.   9.  -4.   |  9.

            r0                r1              r2

    Suppose that we do:

    .. code-block:: python

        res = la.max(a, axis=0, comm)

    then res is now a rank-2 array as follows:

    .. code-block:: text

           /  -7.  -1.  /  -2.   -1.   0.   -6.  /  0.
          / 1.    3.   / -4.    4.   -2.   -4.  /  -4.
         /            /                        /
        /     r1     /           r2           /   r3

    because the axis queried for the max is NOT a distributed axis
    and this is effectively a reduction over the 0-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

    .. code-block:: python

        res = la.max(a, axis=1, comm)

    then this is effectively a reduction over axis=1,
    and every rank will contain the same res which is a rank-2 array as follows

    .. code-block:: text

        -4.   1.
        -3.  -6.
        -4.  -7.

    this is because the max is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

    .. code-block:: python

        res = la.max(a, axis=2, comm)

    then res is now a rank-2 array as follows

    .. code-block:: text

             r0    ||          r1           ||  r2
                   ||                       ||
           2.  3.  ||   2.   5.  -2.   1.   ||  -4.
           1. -1.  ||  -2.  -1.   0.  -6.   ||   0.
          -7.  3.  ||  -4.   0.   3.  -4.   ||   3.
                   ||                       ||

    because the axis queried for the max is NOT a distributed axis
    and this is effectively a reduction over the 2-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    '''
    # Enforce preconditions
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"
    assert_axis_is_none_or_within_rank(a, axis)

    # Return np.min if running serial
    if comm is None or comm.Get_size() == 1:
        return np.min(a, axis=axis)

    # Otherwise, calculate distributed min
    from mpi4py import MPI

    # Get the min on the current process
    local_min = np.min(a, axis=axis)

    # Identify the axis along which the data is the distributed
    distributed_axis = 0 if a.ndim < 3 else 1

    # Return the min of the flattened array if no axis is given
    if axis is None:
        return comm.allreduce(local_min, op=MPI.MIN)

    # If queried axis is the same as distributed axis, perform collective operation
    if axis==distributed_axis:
        if a.ndim == 1:
            local_min = a
        global_min = np.zeros_like(local_min, dtype=local_min.dtype)
        comm.Allreduce(local_min, global_min, op=MPI.MIN)
        return global_min

    # Otherwise, return the local_min on the current process
    return local_min


# # ----------------------------------------------------
def _basic_mean_via_python(a: np.ndarray, dtype=None, axis=None, comm=None):
    '''
    Return the mean of a possibly distributed array over a given axis.

    Parameters:
        a (np.ndarray): input data
        dtype (data-type): Type to use in computing the mean
        axis (None or int): the axis along which to compute the mean. If None, computes the mean of the flattened array.
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        - if axis == None, returns a scalar
        - if axis is not None, returns an array of dimension a.ndim - 1

    Preconditions:
        - a is at most a rank-3 tensor
        - if a is a distributed 2-D array, it must be distributed along axis=0,
          and every rank must have the same a.shape[1]
        - if a is a distributed 3-D tensor, it must be distributed along axis=1,
          and every rank must have the same a.shape[0] and a.shape[2]
        - if axis != None, then it must be an int

    Postconditions:
        - a and comm are not modified

    Example 1:
    ^^^^^^^^^^

    .. code-block:: text

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    .. code-block:: python

        res = la.mean(a, comm)

    then ALL ranks will contain res = 16.21

    Example 2:
    ^^^^^^^^^^

    .. code-block:: text

        rank 0  2.2  1.3  4.
                3.3  5.0  33.
        =======================
        rank 1  40.  -2.  -4.
                51.   4.   6.
                -24.  8.   9.
                45.  -3.  -4.
        =======================
        rank 2  -4.   8.   9.

    Suppose that we do:

    .. code-block:: python

       res = la.mean(a, axis=0, comm)

    then every rank will contain the same res which is:

    .. code-block:: python

       res = ([16.21,  3.04,  7.57])

    this is because the mean is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

    .. code-block:: python

      res = la.mean(a, axis=1, comm)

    then res is now a rank-1 array as follows

    .. code-block:: text

        rank 0  2.5
                13.77
        =======================
        rank 1  11.33
                20.33
                -2.33
                12.67
        =======================
        rank 2  4.33

    because the axis queried for the mean is NOT a distributed axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Example 3:
    ^^^^^^^^^^

    .. code-block:: text

           / 3.   4.   /  2.   8.   2.   1.   / 2.
          /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
         /  -7.   5. /    5.   0.   3.   1. /   3.
        |-----------|----------------------|--------
        | 2.   3.   |  4.   5.  -2.   4.   | -4.
        | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
        | 4.   3.   | -4.   6.   9.  -4.   |  9.

            r0                r1              r2

    Suppose that we do:

    .. code-block:: python

        res = la.mean(a, axis=0, comm)

    then res is now a rank-2 array as follows:

    .. code-block:: text

           /   0.6667   2.6667  /    1.6667   2.3333   1.6667   -1.3333  /   1.6667
          / 2.3333  3.6667     / -0.6667.   5.       5.      -1.        /  4.3333
         /                    /                                        /
        /         r1         /                  r2                    /    r3

    because the axis queried for the mean is NOT a distributed axis
    and this is effectively a reduction over the 0-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

    .. code-block:: python

      res = la.mean(a, axis=1, comm)

    then this is effectively a reduction over axis=1,
    and every rank will contain the same res which is a rank-2 array as follows

    .. code-block:: text

        1.71428571  3.1428571
        3.         -0.5714285
        3.28571429  1.4285714

    this is because the mean is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

    .. code-block:: python

      res = la.mean(a, axis=2, comm)

    then res is now a rank-2 array as follows

    .. code-block:: text

           r0      ||          r1           ||  r2
                   ||                       ||
         2.5  3.5  ||   3.   6.5  0.   2.5  || -1.
         3.5  2.   ||  -2.   1.5  4.  -4.5  ||  4.
        -1.5  4.   ||   0.5  3.   6.  -1.5  ||  6.
                   ||                       ||

    because the axis queried for the mean is NOT a distributed axis
    and this is effectively a reduction over the 2-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    '''
    # Enforce preconditions
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"
    assert_axis_is_none_or_within_rank(a, axis)

    # Return np.mean if running serial
    if comm is None or comm.Get_size() == 1:
        return np.mean(a, dtype=dtype, axis=axis)

    # Otherwise calculate distributed mean
    from mpi4py import MPI

    # Get the size (mean = sum/size) -- num elements if axis is None, or num rows along given axis
    local_size = a.size if axis is None else a.shape[axis]
    global_size = comm.allreduce(local_size, op=MPI.SUM)

    # Warn if dividing by 0
    if global_size == 0:
        warnings.warn("Invalid value encountered in scalar divide (global_size = 0)")
        return np.nan

    # Identify the axis along which the input array is distributed
    distributed_axis = 0 if a.ndim < 3 else 1

    # Calculate mean of flattened array if no axis is given
    if axis is None:
        local_sum = np.sum(a)
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)
        return global_sum / global_size

    # Get mean along distributed axis and perform collective operation
    if axis == distributed_axis:
        local_sum = np.sum(a, axis=axis)
        global_sum = np.zeros_like(np.mean(a, axis=axis))
        comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        return global_sum / global_size

    # Return the local mean if queried axis is not the distributed axis
    return np.mean(a, dtype=dtype, axis=axis)

# ----------------------------------------------------
def _basic_std_via_python(a: np.ndarray, dtype=None, axis=None, comm=None):
    '''
    Return the standard deviation of a possibly distributed array over a given axis.

    Parameters:
        a (np.ndarray): input data
        dtype (data-type): Type to use in computing the mean
        axis (None or int): the axis along which to compute the mean. If None, computes the mean of the flattened array.
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        - if axis == None, returns a scalar
        - if axis is not None, returns an array of dimension a.ndim - 1

    Preconditions:
        - a is at most a rank-3 tensor
        - if a is a distributed 2-D array, it must be distributed along axis=0,
          and every rank must have the same a.shape[1]
        - if a is a distributed 3-D tensor, it must be distributed along axis=1,
          and every rank must have the same a.shape[0] and a.shape[2]
        - if axis != None, then it must be an int

    Postconditions:
        - a and comm are not modified

    Example 1:
    ^^^^^^^^^^

    .. code-block:: text

        rank 0  2.2
                3.3
        =======================
        rank 1  40.
                51.
                -24.
                45.
        =======================
        rank 2  -4.

    .. code-block:: python

        res = la.std(a, comm)

    then ALL ranks will contain res = 26.71

    Example 2:
    ^^^^^^^^^^

    .. code-block:: text

        rank 0  2.2  1.3  4.
                3.3  5.0  33.
        =======================
        rank 1  40.  -2.  -4.
                51.   4.   6.
                -24.  8.   9.
                45.  -3.  -4.
        =======================
        rank 2. -4.  8.   9.

    Suppose that we do:

    .. code-block:: python

        res = la.std(a, axis=0, comm)

    then every rank will contain the same res which is:

    .. code-block:: python

       res = ([26.71,  4.12 , 11.55])

    this is because the standard deviation is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

    .. code-block:: python

        res = la.std(a, axis=1, comm)

    then res is now a rank-1 array as follows

    .. code-block:: text

        rank 0  1.12
                13.62
        =======================
        rank 1  20.29
                21.70
                15.33
                22.87
        =======================
        rank 2  5.91

    because the axis queried for the standard deviation is NOT a distributed axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Example 3:
    ^^^^^^^^^^

    .. code-block:: text

           / 3.   4.   /  2.   8.   2.   1.   / 2.
          /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
         /  -7.   5. /    5.   0.   3.   1. /   3.
        |-----------|----------------------|--------
        | 2.   3.   |  4.   5.  -2.   4.   | -4.
        | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
        | 4.   3.   | -4.   6.   9.  -4.   |  9.

            r0                r1              r2

    Suppose that we do:

    .. code-block:: python

        res = la.std(a, axis=0, comm)

    then res is now a rank-2 array as follows:

    .. code-block:: text

           /   5.5578   2.6247   /    2.8674   4.0277   1.2472   3.2998   /   1.2472
          / 1.2472   0.9428     / 3.3993   0.8165   4.9666   3.5590      / 5.9067
         /                     /                                        /
        /          r1         /                  r2                    /     r3

    because the axis queried for the standard deviation is NOT a distributed axis
    and this is effectively a reduction over the 0-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

    .. code-block:: python

        res = la.std(a, axis=1, comm)

    then this is effectively a reduction over axis=1,
    and every rank will contain the same res which is a rank-2 array as follows

    .. code-block:: text

        3.14934396  2.16653584
        4.14039336  3.28881841
        5.06287004  3.84919817

    this is because the standard deviation is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

    .. code-block:: python

        res = la.std(a, axis=2, comm)

    then res is now a rank-2 array as follows

    .. code-block:: text

           r0      ||          r1           ||  r2
                   ||                       ||
         0.5  0.5  ||   1.   1.5  2.  1.5   ||   3.
         2.5  3.   ||   0.   2.5  4.  1.5   ||   4.
         5.5  1.   ||   4.5  3.   3.  2.5   ||   3.
                   ||                       ||

    because the axis queried for the standard deviation is NOT a distributed axis
    and this is effectively a reduction over the 2-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.
    '''
    # Enforce preconditions
    assert a.ndim <= 3, "a must be at most a rank-3 tensor"
    assert_axis_is_none_or_within_rank(a, axis)

    # Return np.std if running serial
    if comm is None or comm.Get_size() == 1:
        return np.std(a, dtype=dtype, axis=axis)

    # Otherwis, calculate distributed standard deviation
    from mpi4py import MPI

    # Determine the axis along which the data is distributed
    distributed_axis = 0 if a.ndim < 3 else 1

    # Calculate standard deviation of flattened array
    if axis is None:
        global_mean = _basic_mean_via_python(a, dtype=dtype, axis=axis, comm=comm)

        # Compute the sum of the squared differences from the mean
        local_sq_diff = np.sum(np.square(a - global_mean), axis=axis)
        local_size = a.size
        global_size = comm.allreduce(local_size, op=MPI.SUM)
        global_sq_diff = comm.allreduce(local_sq_diff, op=MPI.SUM)

        # Return the standard deviation
        global_std_dev = np.sqrt(global_sq_diff / (global_size))
        return global_std_dev

    # Calculate standard deviation along specified axis
    if axis == distributed_axis:
        global_mean = _basic_mean_via_python(a, dtype=dtype, axis=axis, comm=comm)

        # Compute the sum of the squared differences from the mean
        if distributed_axis == 0:
            local_sq_diff = np.sum(np.square(a - global_mean), axis=axis)
        else:
            # Must specify how to broadcast the global_mean to match dimensions of a
            local_sq_diff = np.sum(np.square(a - global_mean[:,np.newaxis,:]), axis=axis)

        # Get global squared differences
        local_size = a.shape[axis]
        global_size = comm.allreduce(local_size, op=MPI.SUM)
        global_sq_diff = np.zeros_like(local_sq_diff)
        comm.Allreduce(local_sq_diff, global_sq_diff, op=MPI.SUM)

        # Return the standard deviation
        global_std_dev = np.sqrt(global_sq_diff / (global_size))
        return global_std_dev

    # Return the local standard deviation if queried axis is not the distributed axis
    return np.std(a, dtype=dtype, axis=axis)

# ----------------------------------------------------
def _basic_product_via_python(flagA, flagB, alpha, A, B, beta, C, comm=None):
    '''
    Computes C = beta*C + alpha*op(A)*op(B), where A and B are row-distributed matrices.

    Parameters:
        flagA (str): Determines the orientation of A, "T" for transpose or "N" for non-transpose.
        flagB (str): Determines the orientation of B, "T" for transpose or "N" for non-transpose.
        alpha (float): Coefficient of AB.
        A (np.array): 2-D matrix
        B (np.array): 2-D matrix
        beta (float): Coefficient of C.
        C (np.array): 2-D matrix to be filled with the product
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        C (np.array): The specified product
    '''
    if flagA == "N":
        mat1 = A * alpha
    elif flagA == "T":
        mat1 = A.transpose() * alpha
    else:
        raise ValueError("flagA not recognized; use either 'N' or 'T'")

    if flagB == "N":
        mat2 = B
    elif flagB == "T":
        mat2 = B.transpose()
    else:
        raise ValueError("flagB not recognized; use either 'N' or 'T'")

    # CONSTRAINTS
    mat1_shape = np.shape(mat1)
    mat2_shape = np.shape(mat2)

    if (mat1.ndim == 2) and (mat2.ndim == 2):
        if np.shape(C) != (mat1_shape[0], mat2_shape[1]):
            raise ValueError(
                f"Size of output array C ({np.shape(C)}) is invalid. For A (m x n) and B (n x l), C has dimensions (m x l))."
            )

        if mat1_shape[1] != mat2_shape[0]:
            raise ValueError("Invalid input array size. For A (m x n), B must be (n x l).")

    if (mat1.ndim != 2) | (mat2.ndim != 2):
        raise ValueError("This operation currently supports rank-2 tensors.")

    local_product = np.dot(mat1, mat2)

    if comm is not None and comm.Get_size() > 1:

        from mpi4py import MPI

        global_product = np.zeros_like(C, dtype=local_product.dtype)
        comm.Allreduce(local_product, global_product, op=MPI.SUM)

        if beta == 0:
            np.copyto(C, global_product)
        else:
            new_C = beta * C + global_product
            np.copyto(C, new_C)

    else:
        if beta == 0:
            np.copyto(C, local_product)
        else:
            new_C = beta * C + local_product
            np.copyto(C, new_C)

# ----------------------------------------------------
def _transposed_pseudoinverse_via_python(A, comm=None):
    '''
    Computes the pseudoinverse of A and returns its *transpose*.
    Note that returning the transpose(A^+) is because of convenience.
    In fact, when A is row-distributed and comm is not None,
    then the result has the same distribution of A.
    If the matrix A is too large, this is the only feasble way
    to store the pseudoinverse since no single rank can fully store it.

    Parameters:
        - A (np.ndarray): input matrix
        - comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        - The transpose of A^+ computed as: (A^+)^T = A (A^T A)^(-1)^T

    Preconditions:
        - A must be a real, rank-2 matrix
        - A must have more rows than columns
        - A must have linearly independent columns
        - If A is distributed, it must be so along its rows

    Post-conditions:
        - A and comm are not modified
        - A^+ A = I
    '''
    # Check preconditions
    assert A.ndim == 2, "A must be a rank-2 matrix"
    assert np.issubdtype(A.dtype, np.floating)

    # (A^T A)
    C = np.zeros((A.shape[1], A.shape[1]))
    _basic_product_via_python("T", "N", 1, A, A, 0, C, comm)

    # (A^T A)^(-1)
    C_inv = np.linalg.inv(C)

    # A ((A^T A)^(-1))^T
    pinv_transpose = np.zeros((A.shape[0], C_inv.shape[0]))
    _basic_product_via_python("N", "T", 1, A, C_inv, 0, pinv_transpose)

    return pinv_transpose


# ----------------------------------------------------
def _thin_svd_via_method_of_snapshots(snapshots, comm=None):
    '''
    Performs SVD via method of snapshots.

    Args:
        snapshots (np.array): Distributed array of snapshots
        comm (MPI_Comm): MPI communicator (default: None)

    Returns:
        U (np.array): Phi, or modes; a numpy array where each column is a POD mode
        sigma (float): Energy; the energy associated with each mode (singular values)
    '''
    gram_matrix = np.zeros((np.shape(snapshots)[1], np.shape(snapshots)[1]))
    _basic_product_via_python("T", "N", 1, snapshots, snapshots, 0, gram_matrix, comm)
    eigenvalues,eigenvectors = np.linalg.eig(gram_matrix)
    sigma = np.sqrt(eigenvalues)
    modes = np.zeros(np.shape(snapshots))
    modes[:] = np.dot(snapshots, np.dot(eigenvectors, np.diag(1./sigma)))
    ## sort by singular values
    ordering = np.argsort(sigma)[::-1]
    print("function modes:", modes[:, ordering])
    return modes[:, ordering], sigma[ordering]

def _thin_svd_auto_select_algo(M, comm):
    # for now this is it, improve later
    return _thin_svd_via_method_of_snapshots(M, comm)

def _thin_svd(M, comm=None, method='auto'):
    '''
    Preconditions:
      - M is rank-2 tensor
      - if M is distributed, M is distributed over its 0-th axis (row distribution)
      - allowed choices for method are "auto", "method_of_snapshots"

    Returns:
      - left singular vectors and singular values

    Postconditions:
      - M is not modified
      - if M is distributed, the left singular vectors have the same distributions
    '''
    assert method in ['auto', 'method_of_snapshots'], \
        "thin_svd currently supports only method = 'auto' or 'method_of_snapshots'"

    # if user wants a specific algorithm, then call it
    if method == 'method_of_snapshots':
        return _thin_svd_via_method_of_snapshots(M, comm)

    # otherwise we have some freedom to decide
    if comm is not None and comm.Get_size() > 1:
        return _thin_svd_auto_select_algo(M, comm)

    return np.linalg.svd(M, full_matrices=False, compute_uv=True)


def move_distributed_linear_system_to_rank_zero(A_in: np.ndarray, b_in: np.ndarray, comm):
    '''
    Gathers a distributed linear system (A, b) from multiple MPI ranks to rank 0.

    Preconditions:
      - A_in is a rank-2 tensor (2D array) representing a portion of the global matrix A.
      - b_in is a rank-1 or rank-2 tensor (1D or 2D array) representing a portion of the global vector b.
      - A_in and b_in are distributed row-wise across MPI ranks.

    Returns:
      - A_g (numpy.ndarray): The global matrix A assembled on rank 0.
      - b_g (numpy.ndarray): The global vector b assembled on rank 0.

    Postconditions:
      - On rank 0, A_g and b_g contain the fully assembled matrix and vector, respectively.
      - On other ranks, A_g and b_g are dummy arrays with no meaningful content.
      - The input arrays A_in and b_in are not modified.

    Notes:
      - The function ensures that all data is gathered without additional copies or unnecessary data movement.
      - Only rank 0 ends up with the complete system; other ranks have placeholder arrays.
    '''
    from mpi4py import MPI

    root_rank  = 0
    my_rank = comm.Get_rank()

    # need to copy into C order because this is needed below when we
    # serialize to send/recv with mpi wihout additional copies and also
    # working correctly to store the data when received
    A = np.copy(A_in, order='C') if np.isfortran(A_in) else A_in
    b = np.copy(b_in, order='C') if np.isfortran(b_in) else b_in
    my_num_rows = 0 if A.size == 0 else A.shape[0]
    my_num_cols = 0 if A.size == 0 else A.shape[1]

    # for ranks where we have data, check that num of rows of A = rows of b
    # and that the dimensionality makes sense
    if A.size > 0:
        assert A.shape[0] == b.ravel().size
        assert A.ndim == 2
        assert b.ndim <= 2
        if b.ndim == 2:
            assert b.shape[1] == 1

    # count total num of rows across the whole communicator
    rows_per_rank = np.zeros(comm.Get_size(), dtype=int)
    comm.Gather(np.array([my_num_rows]), rows_per_rank)
    global_num_rows = np.sum(rows_per_rank)
    # at least one rank must have data
    if my_rank==root_rank:
        assert global_num_rows > 0

    # we need to figure out the num of columns using a collective
    # we assume row-distributed
    global_num_cols = np.array([0], dtype=int)
    comm.Reduce(np.array([my_num_cols], dtype=int), global_num_cols, op=MPI.MAX)
    # global_num_cols is only valid on rank root_rank
    global_num_cols = global_num_cols[0]

    # create the storage for the final assembled system
    # note that this only has meaningful shape on rank root_rank
    # all other ranks have a dummy A_g, b_g
    A_g = np.zeros((global_num_rows, global_num_cols), order='C')
    b_g = np.zeros(global_num_rows)

    # each rank != root_rank starts the send of its part of A and b
    my_reqs = []
    if my_rank > root_rank:
        if A.size > 0:
            tag_A = my_rank*2
            # we can ravel here because A is row-major  so this guarantees a view
            req = comm.Isend(np.ravel(A), 0, tag=tag_A)
            my_reqs.append(req)
            req = comm.Isend(np.ravel(b), 0, tag=tag_A+1)
            my_reqs.append(req)

    else:
        # rank0 first stores, if needed, its part
        if my_num_rows > 0:
            A_g[0:my_num_rows, :] = A
            b_g[0:my_num_rows] = b.ravel()

        # then posts recvs for all other messages from other ranks
        row_shift = my_num_rows
        for i_rank in range(1, comm.Get_size()):
            curr_rank_num_rows = rows_per_rank[i_rank]
            if curr_rank_num_rows > 0:
                tag_A = i_rank*2
                row_begin = row_shift
                row_end_exclusive = row_shift + curr_rank_num_rows
                req = comm.Irecv(np.ravel(A_g[row_begin:row_end_exclusive,:]), i_rank, tag=tag_A)
                my_reqs.append(req)
                req = comm.Irecv(b_g[row_shift:], i_rank, tag=tag_A+1)
                my_reqs.append(req)
                row_shift += curr_rank_num_rows

    for req in my_reqs:
        req.Wait()

    return A_g, b_g

# ----------------------------------------------------
def load_snapshot(dataset_dir: str, i: int):
    '''
    Load snapshot i from disk.

    Parameters:
        dataset_dir (str): directory containing snapshot files.
        i (int): snapshot index.

    Returns:
        - ndarray(shape=(N,)) - snapshot vector.
    '''
    path = Path(dataset_dir)
    return np.loadtxt(path / f"snapshot_{i}.txt")

def _snapshot_loader(dataset_dir: str, start: int, end: int):
    '''
    Load a contiguous range of snapshots from a dataset directory and stack them
    as columns in a single matrix.

    Parameters:
        dataset_dir (str): directory containing snapshot files.
        start (int): first snapshot index (inclusive).
        end (int): last snapshot index (exclusive).

    Return:
        - Xb, ndarray(shape=(N, end-start)) - block of snapshots stacked as columns.

    Exemple 1
    ^^^^^^^^^

    Load snapshots 0 through 9:

    .. code-block:: text

        >>> Xb = _snapshot_loader("data/snapshots", 0, 10)
        >>> Xb.shape
        (N, 10)

    Exemple 2
    ^^^^^^^^^

    Load a subset of snapshots for a training block:

    .. code-block:: text

        >>> Xb = _snapshot_loader("data/snapshots", 20, 25)
        >>> Xb.shape
        (N, 5)
    '''
    snapshots = []
    for i in range(start, end):
        Xbi = load_snapshot(dataset_dir, i)
        snapshots.append(Xbi)
    Xb = np.column_stack(snapshots)
    return Xb

def _streaming_pod(snapshot_loader, block_size: int, N: int, M: int, k: int, p: int):
    '''
    Compute an approximate POD/SVD decomposition of a snapshot matrix using a
    two-pass randomized streaming algorithm that processes snapshots in blocks.

    Parameters:
        snapshot_loader: capable of loading blocks of columns (of X).
        block_size (int): number of snapshots loaded at once.
        N (int): number of rows (of X).
        M (int): number of columns/snapshots (of X).
        k (int): target rank.
        p (int): oversampling parameter.

    Returns:
        - Uk, ndarray(shape=(N, k)) - approximate POD modes.
        - Sk, ndarray(shape=(k,)) - approximate POD singular values.
        - Vk, ndarray(shape=(k, M)) - approximate right singular vectors.

    Example 1
    ^^^^^^^^^

    Compute the first 3 POD modes from 8 snapshots loaded in blocks of 2:

    .. code-block:: text

        >>> path = "data/snapshots/rank3_5x8"
        >>> loader = lambda s, e: _snapshot_loader(path, s, e)
        >>> U, S, Vt = _streaming_pod(
                snapshot_loader=loader,
                block_size=2,
                N=5, M=8,
                k=3, p=1,
            )

        >>> U.shape
        (5, 3)

        >>> S.shape
        (3,)

        >>> Vt.shape
        (3, 8)

    Example 2
    ^^^^^^^^^

    Compute a rank-10 approximation of a larger dataset while limiting memory
    usage by loading snapshots in blocks of 50:

    .. code-block:: text

        >>> U, S, Vt = _streaming_pod(
                snapshot_loader=loader,
                block_size=50,
                N=1000, M=500,
                k=10, p=5,
            )

        >>> U.shape
        (1000, 10)

        >>> S.shape
        (10,)

        >>> Vt.shape
        (10, 500)
    '''

    assert block_size < M
    # NOTE: block_size = M -> no more streaming...
    # NOTE: block_size = 1 -> minimum memory, lot of I/O.

    assert k <= N and k <= M

    # sketch dimension
    l = k + p
    assert l <= N and l <= M

    # pass 1
    omega = np.random.randn(M, l)
    Y = np.zeros(shape=(N, l))
    for start in range(0, M, block_size):
        end = builtins.min(start + block_size, M)
        Xb = snapshot_loader(start, end)
        Ob = omega[start:end, :]
        Y += Xb @ Ob

    # compute orthonormal basis
    Uy, _, _ = np.linalg.svd(Y, full_matrices=False)
    Q = Uy[:, :l]

    # pass 2
    B = np.zeros(shape=(l, M))
    for start in range(0, M, block_size):
        end = builtins.min(start + block_size, M)
        Xb = snapshot_loader(start, end)
        Bb = Q.T @ Xb
        B[:, start:end] = Bb

    # compute approximate SVD
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    # results
    return (U[:, :k], S[:k], Vt[:k, :])

def _local_column_range(rank, size, M):
    '''
    Compute the range of matrix columns assigned to a given process.

    Parameters:
        rank (int): identifier of the current process
        size (int): total number of processes
        M (int): total number of columns to distribute

    Returns:
        - start (int) - index of the first column assigned to the process (inclusive).
        - end (int) - index of the last column boundary (exclusive).

    Postconditions:
        - Columns are distributed as evenly as possible among processes.
        - If M is not divisible by size, the first (M % size) processes receive one additional column.

    Example 1
    ^^^^^^^^^

    For size = 3 and M = 10:

    .. code-block:: text

        >>> start, end = _local_column_range(rank=0, size=3, M=10)
        (0, 4)

        >>> start, end = _local_column_range(rank=1, size, M)
        (4, 7)

        >>> start, end = _local_column_range(rank=1, size, M)
        (7, 10)

    Example 2
    ^^^^^^^^^

    For size = 4 and M = 8:

    .. code-block:: text

        >>> start, end = _local_column_range(rank=0, size=4, M=8)
        (0, 2)

        >>> start, end = _local_column_range(rank=1, size, M)
        (2, 4)

        >>> start, end = _local_column_range(rank=2, size, M)
        (4, 6)

        >>> start, end = _local_column_range(rank=3, size, M)
        (6, 8)
    '''
    q, r = divmod(M, size)

    start = rank * q + builtins.min(rank, r)
    end = start + q + (1 if rank < r else 0)

    return start, end

def _streaming_pod_mpi(snapshot_loader, N: int, M: int, k: int, p: int, comm=None):
    '''
    Distributed randomized streaming POD using MPI by partitioning snapshot
    columns across processes and combining local computations through collective
    communication.

    Parameters:
        snapshot_loader: callable(start, end) returning locally assigned
            snapshot columns with shape (N, end-start).
        N (int): number of rows of the snapshot matrix X.
        M (int): number of columns (snapshots) of X.
        k (int): target rank.
        p (int): oversampling parameter.
        comm: MPI communicator (defaults to MPI.COMM_WORLD).

    Returns:
        - Uk, ndarray(shape=(N, k)) - approximate POD modes.
        - Sk, ndarray(shape=(k,)) - approximate POD singular values.
        - Vk, ndarray(shape=(k, M)) - approximate right singular vectors transposed.

    Example 1
    ^^^^^^^^^

    Compute a rank-3 POD approximation using 4 MPI processes.

    .. code-block:: bash

        mpirun -n 4 python pod_mpi.py

    .. code-block:: text

        >>> comm = MPI.COMM_WORLD
        >>> loader = lambda s, e: _snapshot_loader("snapshots/rank3_5x8", s, e)
        >>> U, S, Vt = _streaming_pod_mpi(
                snapshot_loader=loader,
                N=5, M=8,
                k=1, p=1,
                comm=comm
            )

        >>> U.shape
        (5, 1)

        >>> S.shape
        (1,)

        >>> Vt.shape
        (1, 8)

    Example 2
    ^^^^^^^^^

    Compute a rank-10 POD approximation of a large snapshot matrix distributed
    across 8 MPI processes.

    .. code-block:: bash

        mpirun -n 8 python pod_mpi.py

    .. code-block:: text

        >>> U, S, Vt = _streaming_pod_mpi(
            snapshot_loader=loader,
            N=5000, M=1000,
            k=10, p=5,
            comm=comm
        )

        >>> U.shape
        (5000, 10)
    '''
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    l = k + p
    assert l <= N and l <= M

    # pass 1: distribute
    if rank == 0:
        omega = np.random.randn(M, l)
    else:
        omega = None

    omega = comm.bcast(omega, root=0)

    start, end = _local_column_range(rank, size, M)
    X_local = snapshot_loader(start, end)
    omega_local = omega[start:end, :]
    Y_local = X_local @ omega_local # shape=(N,l)
    Y = np.zeros(shape=(N,l))
    comm.Allreduce(Y_local, Y, op=MPI.SUM)

    # compute orthonormal basis: no distribute
    Uy, _, _ = np.linalg.svd(Y, full_matrices=False)
    Q = Uy[:, :l]

    # pass 2: distribute
    B_local = Q.T @ X_local # shape=(l, M_local)

    # gather
    B_parts = comm.gather(B_local, root=0)

    if rank == 0:
        B = np.hstack(B_parts)
        U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    else:
        U_tilde = None
        S = None
        Vt = None

    # broadcast
    U_tilde = comm.bcast(U_tilde, root=0)
    S = comm.bcast(S, root=0)
    Vt = comm.bcast(Vt, root=0)

    # compute approximate SVD
    U = Q @ U_tilde

    # results
    return (U[:, :k], S[:k], Vt[:k, :])

def _distributed_svd(a, comm=None, full_matrices=True, compute_uv=True,
                     hermitian=False):
    '''Compute the thin SVD of a matrix distributed by rows.

    This is a two-level Tall-Skinny QR (TSQR) algorithm. Each MPI rank
    factors its local rows, rank zero factors the vertically stacked local
    ``R`` factors, and only the resulting small factors are communicated.
    The global input matrix is never assembled on any rank.

    Parameters:
        a (np.ndarray): Local rows of a globally row-distributed 2-D matrix.
        comm (MPI_Comm): MPI communicator. If ``None``, NumPy is used directly.
        full_matrices (bool): ``False`` is supported when singular vectors are
            requested. As in NumPy, this option is ignored when
            ``compute_uv=False``.
        compute_uv (bool): If ``True``, return ``(U_local, s, Vh)``. Otherwise,
            return only ``s``.
        hermitian (bool): Only ``False`` is supported.

    Returns:
        - If ``compute_uv=True``, ``(U_local, s, Vh)``. ``U_local`` has the same
            row distribution as ``a`` while ``s`` and ``Vh`` are replicated.
        - If ``compute_uv=False``, only the replicated singular values are
            returned.
    '''
    # Convert the local input without copying it when it is already an ndarray,
    # and record validation failures instead of raising immediately. Delaying
    # the error lets every MPI rank participate in the same collective.
    local_error = None
    try:
        local_a = np.asarray(a)
        if local_a.ndim != 2:
            local_error = "a must be a two-dimensional array"
        elif not np.issubdtype(local_a.dtype, np.number):
            local_error = "a must have a real or complex numeric dtype"
    except Exception as exception:
        local_a = None
        local_error = f"a could not be converted to an array: {exception}"

    local_metadata = {
        "error": local_error,
        "shape": None if local_a is None or local_a.ndim != 2 else local_a.shape,
        "dtype": None if local_a is None else local_a.dtype.str,
        "options": (bool(full_matrices), bool(compute_uv), bool(hermitian)),
    }

    # Exchange only error, shape, dtype, and options metadata. This collectively
    # validates the row distribution before any rank enters QR.
    if comm is None:
        all_metadata = [local_metadata]
    else:
        all_metadata = comm.allgather(local_metadata)

    errors = [metadata["error"] for metadata in all_metadata
              if metadata["error"] is not None]
    if errors:
        raise ValueError("invalid distributed SVD input: " + "; ".join(errors))

    option_sets = {metadata["options"] for metadata in all_metadata}
    if len(option_sets) != 1:
        raise ValueError("all ranks must use the same SVD options")

    column_counts = {metadata["shape"][1] for metadata in all_metadata}
    if len(column_counts) != 1:
        raise ValueError(
            "all ranks must have the same number of matrix columns"
        )

    dtypes = {metadata["dtype"] for metadata in all_metadata}
    if len(dtypes) != 1:
        raise ValueError("all ranks must use the same matrix dtype")

    # Reject NumPy options for which this distributed implementation cannot
    # provide the documented NumPy result shape or algorithm semantics.
    if hermitian:
        raise NotImplementedError(
            "DistributedSvd does not support hermitian=True"
        )
    if compute_uv and full_matrices:
        raise NotImplementedError(
            "DistributedSvd supports only full_matrices=False when "
            "compute_uv=True"
        )

    # For a serial call, use NumPy directly after applying the same option
    # validation as the distributed path.
    if comm is None:
        return np.linalg.svd(
            local_a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=False,
        )

    rank = comm.Get_rank()
    root = 0

    # Compute a reduced QR factorization of the local rows. Empty local
    # partitions and partitions with fewer rows than columns are valid inputs.
    local_qr_error = None
    try:
        local_q, local_r = np.linalg.qr(local_a, mode="reduced")
    except Exception as exception:
        local_q = None
        local_r = None
        local_qr_error = str(exception)

    qr_errors = comm.allgather(local_qr_error)
    qr_errors = [error for error in qr_errors if error is not None]
    if qr_errors:
        raise np.linalg.LinAlgError(
            "local QR factorization failed: " + "; ".join(qr_errors)
        )

    # Gather only the reduced R factors on rank zero.
    gathered_r = comm.gather(local_r, root=root)

    singular_values = None
    right_singular_vectors = None
    local_left_transforms = None
    reduction_error = None

    if rank == root:
        try:
            # Finish the two-level TSQR factorization on rank zero by factoring
            # the vertical stack of local R factors.
            stacked_r = np.vstack(gathered_r)
            reduced_q, final_r = np.linalg.qr(stacked_r, mode="reduced")

            if compute_uv:
                # A: Compute the SVD only of the final reduced factor, then fold
                # its left vectors into the second-level TSQR Q.
                final_u, singular_values, right_singular_vectors = np.linalg.svd(
                    final_r,
                    full_matrices=False,
                    compute_uv=True,
                    hermitian=False,
                )
                stacked_left_transform = reduced_q @ final_u

                # Split the reduced left transformation according to each rank's
                # local R row count so it can be scattered below.
                local_left_transforms = []
                row_offset = 0
                for local_r_factor in gathered_r:
                    next_offset = row_offset + local_r_factor.shape[0]
                    local_left_transforms.append(
                        stacked_left_transform[row_offset:next_offset, :]
                    )
                    row_offset = next_offset
            else:
                # B: NumPy returns only singular values when left and right
                # singular vectors were not requested.
                singular_values = np.linalg.svd(
                    final_r,
                    full_matrices=full_matrices,
                    compute_uv=False,
                    hermitian=False,
                )
        except Exception as exception:
            reduction_error = str(exception)

    # Broadcast a root-side failure before entering any later collectives,
    # preventing other ranks from waiting indefinitely.
    reduction_error = comm.bcast(reduction_error, root=root)
    if reduction_error is not None:
        raise np.linalg.LinAlgError(
            "reduced QR or SVD factorization failed: " + reduction_error
        )

    # Replicate the singular values, matching NumPy's single-array return
    # convention when compute_uv=False.
    singular_values = comm.bcast(singular_values, root=root)
    if not compute_uv:
        return singular_values

    # Replicate Vh, scatter the reduced left transformations, and multiply by
    # the local first-level Q to recover each rank's rows of U.
    right_singular_vectors = comm.bcast(right_singular_vectors, root=root)
    local_left_transform = comm.scatter(local_left_transforms, root=root)
    local_left_singular_vectors = local_q @ local_left_transform

    return local_left_singular_vectors, singular_values, right_singular_vectors


class DistributedSvd:
    '''NumPy-compatible thin SVD callable for a row-distributed matrix.

    Bind an MPI communicator once, then pass this object anywhere a callable
    compatible with :func:`numpy.linalg.svd` is expected, including
    :class:`romtools.vector_space.VectorSpaceFromPOD`.

    Args:
        comm (MPI_Comm): Communicator over which matrix rows are distributed.
            If ``None``, supported calls delegate to NumPy in serial.

    Example:
        >>> distributed_svd = DistributedSvd(comm)
        >>> U_local, s, Vh = distributed_svd(
        ...     A_local, full_matrices=False, compute_uv=True, hermitian=False
        ... )
    '''

    def __init__(self, comm=None) -> None:
        self._comm = comm

    def __call__(self, a, full_matrices=True, compute_uv=True,
                 hermitian=False):
        '''Compute a thin SVD collectively over the bound communicator.

        Args:
            a (np.ndarray): The calling rank's local matrix rows.
            full_matrices (bool): Must be ``False`` when ``compute_uv=True``.
            compute_uv (bool): Return singular vectors when ``True``; otherwise
                return only singular values.
            hermitian (bool): Must be ``False``.

        Returns:
            ``(U_local, s, Vh)`` when ``compute_uv=True``; otherwise ``s``.
        '''
        # Use the internal implementation.
        return _distributed_svd(
            a,
            comm=self._comm,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
        )
# ----------------------------------------------------
# ----------------------------------------------------

# pylint: disable=redefined-builtin
# Define public facing API
max = _basic_max_via_python
argmax = _basic_argmax_via_python
min = _basic_min_via_python
mean = _basic_mean_via_python
std = _basic_std_via_python
product = _basic_product_via_python
pinv = _transposed_pseudoinverse_via_python
thin_svd = _thin_svd
snapshot_loader = _snapshot_loader
streaming_pod = _streaming_pod
local_column_range = _local_column_range
streaming_pod_mpi = _streaming_pod_mpi
