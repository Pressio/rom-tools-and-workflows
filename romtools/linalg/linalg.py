
'''
see this for why this file exists and is done this way
https://stackoverflow.com/questions/47599162/pybind11-how-to-package-c-and-python-code-into-a-single-package?rq=1
'''

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
        if axis == None, returns a scalar
        if axis is not None, returns an array of dimension a.ndim - 1

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
    **********

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    res = la.max(a, comm)
    then ALL ranks will contain res = 51.

    Example 2:
    **********

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

       res = la.max(a, axis=0, comm)

    then every rank will contain the same res which is an array = ([51., 8., 33])
    this is because the max is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

      res = la.max(a, axis=1, comm)

    then res is now a rank-1 array as follows

       rank 0  4.
               33.
      =======================
       rank 1  40.
               51.
               9.
               45.
      =======================
       rank 2  9.

    because the axis queried for the max is NOT a distributed axis
    so this operation is purely local and the result has the same distribution
    as the original array.


    Example 3:
    **********

       / 3.   4.   /  2.   8.   2.   1.   / 2.
      /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
     /  -7.   5. /    5.   0.   3.   1. /   3.
    |-----------|----------------------|--------
    | 2.   3.   |  4.   5.  -2.   4.   | -4.
    | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
    | 4.   3.   | -4.   6.   9.  -4.   |  9.

        r0                r1              r2

    Suppose that we do:

        res = la.max(a, axis=0, comm)

    then res is now a rank-2 array as follows:

       /  6.  5.   /  5.   8.   3.   1.  /  3.
      / 4.   5.   / 4.   6.   9.   4.   /  9.
     /           /                     /
    /    r1     /         r2          /  r3

    because the axis queried for the max is NOT a distributed axis
    and this is effectively a reduction over the 0-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

      res = la.max(a, axis=1, comm)

    then this is effectively a reduction over axis=1,
    and every rank will contain the same res which is a rank-2 array as follows

                  5.  8.
                  8.  6.
                  9.  5.

    this is because the max is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

      res = la.max(a, axis=2, comm)

    then res is now a rank-2 array as follows

            r0     ||          r1           ||  r2
                   ||                       ||
          3.   4.  ||   4.   8.   2.   4.   ||   2.
          6.   5.  ||  -2.   4.   8.  -3.   ||   8.
          4.   5.  ||   5.   6.   9.   1.   ||   9.
                   ||                       ||

    because the axis queried for the max is NOT a distributed axis
    and this is effectively a reduction over the 2-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

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
        if comm == None, returns the index of the maximum value (identical to np.argmax)
        if comm != None, returns a tuple containing (value, index, rank):
            value: the global maximum
            index: the local index of the global maximum
            rank:  the rank on which the global maximum resides

    Preconditions:
      - a is at most a rank-3 tensor
      - if a is a distributed 2-D array, it must be distributed along axis=0,
        and every rank must have the same a.shape[1]
      - if a is a distributed 3-D tensor, it must be distributed along axis=1,
        and every rank must have the same a.shape[0] and a.shape[2]

    Postconditions:
      - a and comm are not modified

    Example 1:
    **********

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

        res = la.argmax(a, comm)

    then ALL ranks will contain res = (1, 1).
    (The global maximum (51.) occurs at index 1 of the local array on Rank 1.)

    Example 2:
    **********

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

       res = la.argmax(a, comm)

    then ALL ranks will contain res = (3, 1)
    (The global maximum (51.) occurs at index 3 of the flattened local array on Rank 1.)

    Example 3:
    **********

       / 3.   4.   /  2.   8.   2.   1.   / 2.
      /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
     /  -7.   5. /    5.   0.   3.   1. /   3.
    |-----------|----------------------|--------
    | 2.   3.   |  4.   5.  -2.   4.   | -4.
    | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
    | 4.   3.   | -4.   6.   9.  -4.   |  9.

        r0                r1              r2

    Suppose that we do:

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

    # Find distributed max index
    from mpi4py import MPI

    # Define custom MPI op
    def mycomp(A_mem,B_mem):
        # Get matrices from memory buffers
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
        if axis == None, returns a scalar
        if axis is not None, returns an array of dimension a.ndim - 1

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
    **********

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    res = la.min(a, comm)
    then ALL ranks will contain res = -4.

    Example 2:
    **********

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

       res = la.min(a, axis=0, comm)

    then every rank will contain the same res which is an array = ([-24., -3., -4])
    this is because the min is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

      res = la.min(a, axis=1, comm)

    then res is now a rank-1 array as follows

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
    **********

       / 3.   4.   /  2.   8.   2.   1.   / 2.
      /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
     /  -7.   5. /    5.   0.   3.   1. /   3.
    |-----------|----------------------|--------
    | 2.   3.   |  4.   5.  -2.   4.   | -4.
    | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
    | 4.   3.   | -4.   6.   9.  -4.   |  9.

        r0                r1              r2

    Suppose that we do:

        res = la.max(a, axis=0, comm)

    then res is now a rank-2 array as follows:

       /  -7.  -1.  /  -2.   -1.   0.   -6.  /  0.
      / 1.    3.   / -4.    4.   -2.   -4.  /  -4.
     /            /                        /
    /     r1     /           r2           /   r3

    because the axis queried for the max is NOT a distributed axis
    and this is effectively a reduction over the 0-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

      res = la.max(a, axis=1, comm)

    then this is effectively a reduction over axis=1,
    and every rank will contain the same res which is a rank-2 array as follows

                    -4.   1.
                    -3.  -6.
                    -4.  -7.

    this is because the max is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

      res = la.max(a, axis=2, comm)

    then res is now a rank-2 array as follows

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
        if axis == None, returns a scalar
        if axis is not None, returns an array of dimension a.ndim - 1

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
    **********

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    res = la.mean(a, comm)
    then ALL ranks will contain res = 16.21


    Example 2:
    **********

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

       res = la.mean(a, axis=0, comm)

    then every rank will contain the same res which is:

       res  = ([16.21,  3.04,  7.57])

    this is because the mean is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

      res = la.mean(a, axis=1, comm)

    then res is now a rank-1 array as follows

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
    **********

       / 3.   4.   /  2.   8.   2.   1.   / 2.
      /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
     /  -7.   5. /    5.   0.   3.   1. /   3.
    |-----------|----------------------|--------
    | 2.   3.   |  4.   5.  -2.   4.   | -4.
    | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
    | 4.   3.   | -4.   6.   9.  -4.   |  9.

        r0                r1              r2

    Suppose that we do:

        res = la.mean(a, axis=0, comm)

    then res is now a rank-2 array as follows:

       /   0.6667   2.6667  /    1.6667   2.3333   1.6667   -1.3333  /   1.6667
      / 2.3333  3.6667     / -0.6667.   5.       5.      -1.        /  4.3333
     /                    /                                        /
    /         r1         /                  r2                    /    r3

    because the axis queried for the mean is NOT a distributed axis
    and this is effectively a reduction over the 0-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

      res = la.mean(a, axis=1, comm)

    then this is effectively a reduction over axis=1,
    and every rank will contain the same res which is a rank-2 array as follows

              1.71428571  3.1428571
              3.         -0.5714285
              3.28571429  1.4285714

    this is because the mean is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

      res = la.mean(a, axis=2, comm)

    then res is now a rank-2 array as follows

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
        if axis == None, returns a scalar
        if axis is not None, returns an array of dimension a.ndim - 1

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
    **********

       rank 0  2.2
               3.3
      =======================
       rank 1  40.
               51.
               -24.
               45.
      =======================
       rank 2  -4.

    res = la.std(a, comm)
    then ALL ranks will contain res = 26.71

    Example 2:
    **********

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

       res = la.std(a, axis=0, comm)

    then every rank will contain the same res which is:

       res  = ([26.71,  4.12 , 11.55])

    this is because the standard deviation is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation.

    Suppose that we do:

      res = la.std(a, axis=1, comm)

    then res is now a rank-1 array as follows

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
    **********

       / 3.   4.   /  2.   8.   2.   1.   / 2.
      /  6.  -1.  /  -2.  -1.   0.  -6.  /  0.    -> slice T(:,:,1)
     /  -7.   5. /    5.   0.   3.   1. /   3.
    |-----------|----------------------|--------
    | 2.   3.   |  4.   5.  -2.   4.   | -4.
    | 1.   5.   | -2.   4.   8.  -3.   |  8.    ->  slice T(:,:,0)
    | 4.   3.   | -4.   6.   9.  -4.   |  9.

        r0                r1              r2

    Suppose that we do:

        res = la.std(a, axis=0, comm)

    then res is now a rank-2 array as follows:

       /   5.5578   2.6247   /    2.8674   4.0277   1.2472   3.2998   /   1.2472
      / 1.2472   0.9428     / 3.3993   0.8165   4.9666   3.5590      / 5.9067
     /                     /                                        /
    /          r1         /                  r2                    /     r3

    because the axis queried for the standard deviation is NOT a distributed axis
    and this is effectively a reduction over the 0-th axis
    so this operation is purely local and the result has the same distribution
    as the original array.

    Suppose that we do:

      res = la.std(a, axis=1, comm)

    then this is effectively a reduction over axis=1,
    and every rank will contain the same res which is a rank-2 array as follows

              3.14934396  2.16653584
              4.14039336  3.28881841
              5.06287004  3.84919817

    this is because the standard deviation is queried for the 0-th axis which is the
    axis along which the data array is distributed.
    So this operation must be a collective operation and we know that
    memory-wise it is feasible to hold because this is no larger than the
    local allocation on each rank.

    Suppose that we do:

      res = la.std(a, axis=2, comm)

    then res is now a rank-2 array as follows

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
