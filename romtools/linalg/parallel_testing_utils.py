import numpy as np
try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


#####################################################
########             MPI Helpers             ########
#####################################################

def distribute_array_impl(global_array, comm, dist_axis=0):
    '''
    Splts an np.array and distributes to all available MPI processes as evenly as possible.

    For example, distributing an array with 6 rows over 3 processes will result in 2 rows
    per process.
    If the array has 7 rows, 2 processors will hold 2 rows, and the third will hold 3 rows.

    Inputs:
        global_array: The global np.array to be distributed.
        comm: The MPI communicator
        dist_axis: The axis along which to split the input array. By default, splits along the first axis (rows).

    Returns:
        local_array: The subset of global_array sent to the current MPI process.

    '''
    # Get comm info
    n_procs = comm.Get_size()
    rank = comm.Get_rank()

    # Handle null case
    if global_array.size == 0:
        return np.empty(0)

    # Split the global_array and send to corresponding MPI rank
    if rank == 0:
        splits = np.array_split(global_array, n_procs, axis=dist_axis)
        for proc in range(n_procs):
            if proc == 0:
                local_array = splits[proc]
            else:
                comm.send(splits[proc], dest=proc)
    else:
        local_array = comm.recv(source=0)

    return local_array

def generate_random_local_and_global_arrays_impl(shape, comm):
    '''
    Randomly generates a global array of the specified shape and distributes to all available
    MPI processes.

    Returns both the local and global arrays.
    '''
    # Get comm info
    rank = comm.Get_rank()

    # Create global_array (using optional dim<x> arguments)
    if shape == tuple():
        global_arr = np.empty(0)
    elif len(shape) <=3:
        global_arr = np.random.rand(*shape) if rank == 0 else np.empty(shape)
    else:
        raise ValueError(f"This function only supports arrays up to rank 3 (received rank {ndim})")

    # Broadcast global_array and create local_array
    comm.Bcast(global_arr, root=0)
    dist_axis = 0 if len(shape) < 3 else 1
    local_arr = distribute_array_impl(global_arr.copy(), comm, dist_axis)

    return local_arr, global_arr

def generate_local_and_global_arrays_from_example_impl(rank, slices, example: int):
    '''
    Generates and returns the local and global arrays built from the example tensors in the documentation.
    '''
    # Create arrays
    if example == 1:
        global_arr = np.array([2.2, 3.3, 40., 51., -24., 45., -4.])
        local_arr = global_arr[slices[rank][0]:slices[rank][1]]

    elif example == 2:
        global_arr = np.array([[2.2, 1.3, 4.],
                               [3.3, 5.0, 33.],
                               [40., -2., -4.],
                               [51., 4., 6.],
                               [-24., 8., 9.],
                               [45., -3., -4.],
                               [-4., 8., 9.]])
        local_arr = global_arr[slices[rank][0]:slices[rank][1],:]

    elif example == 3:
        global_arr = np.array([[[2.,3.],[3.,4.],[4.,2.],[5.,8.],[-2.,2.],[4.,1.],[-4.,2.]],
                               [[1.,6.],[5.,-1.],[-2.,-2.],[4.,-1.],[8.,0.],[-3.,-6.],[8.,0.]],
                               [[4.,-7.],[3.,5.],[-4.,5.],[6.,0.],[9.,3.],[-4.,1.],[9.,3.]]])
        local_arr = global_arr[:,slices[rank][0]:slices[rank][1],:]

    else:
        return None, None

    return local_arr, global_arr
