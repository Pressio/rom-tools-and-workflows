import pytest
import numpy as np
from romtools.vector_space.utils.shifter import *
from pressiolinalg import test_utils
try:
  import mpi4py
  from mpi4py import MPI
except ModuleNotFoundError:
  print("module 'mpi4py' is not installed")


@pytest.mark.mpi_skip
def test_noop_shifter():
  my_snapshots = np.random.normal(size=(3,10,2))
  original_snapshots = my_snapshots.copy()
  shifter = create_noop_shifter(my_snapshots)
  shifter.apply_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots))
  shifter.apply_inverse_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots))

@pytest.mark.mpi(min_size=3)
def test_noop_shifter_mpi():
  comm = MPI.COMM_WORLD
  global_shape = (3,10,2)
  local_snapshots, global_snapshots = test_utils.generate_random_local_and_global_arrays_impl(global_shape, comm=comm)
  shifter = create_noop_shifter(local_snapshots)
  shifter.apply_shift(local_snapshots)
  assert len(np.setdiff1d(local_snapshots, global_snapshots)) == 0
  shifter.apply_inverse_shift(local_snapshots)
  assert len(np.setdiff1d(local_snapshots, global_snapshots)) == 0

@pytest.mark.mpi_skip
def test_constant_shifter():
  # Test with np.ndarray
  shift_value = np.array([4,1,3],dtype='int')
  my_snapshots = np.random.normal(size=(3,10,2))
  original_snapshots = my_snapshots.copy()
  shifter = create_constant_shifter(shift_value, my_snapshots)
  shifter.apply_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots - shift_value[:,None,None]))
  shifter.apply_inverse_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots))

  # Test with number
  shift_value = 2
  my_snapshots = np.random.normal(size=(3,10,2))
  test_snapshots = my_snapshots.copy()
  shifter = create_constant_shifter(shift_value, my_snapshots)
  shifter.apply_shift(my_snapshots)
  test_snapshots -= shift_value
  assert(np.allclose(my_snapshots, test_snapshots))
  shifter.apply_inverse_shift(my_snapshots)
  test_snapshots += shift_value
  assert(np.allclose(my_snapshots, test_snapshots))

@pytest.mark.mpi(min_size=3)
def test_constant_shifter_mpi():
  comm = MPI.COMM_WORLD
  global_shape = (3,10,2)

  # Test with np.ndarray
  shift_value = np.array([4,1,3],dtype='int')
  original_local_snapshots, global_snapshots = test_utils.generate_random_local_and_global_arrays_impl(global_shape, comm=comm)
  local_snapshots = original_local_snapshots.copy()
  shifter = create_constant_shifter(shift_value, local_snapshots)
  shifter.apply_shift(local_snapshots)
  for i in range(global_snapshots.shape[0]):
    global_snapshots[i] -= shift_value[i]
  assert len(np.setdiff1d(local_snapshots, global_snapshots)) == 0
  shifter.apply_inverse_shift(local_snapshots)
  assert np.allclose(local_snapshots, original_local_snapshots)

  # Test with number
  shift_value = 2
  local_snapshots, global_snapshots = test_utils.generate_random_local_and_global_arrays_impl(global_shape, comm=comm)
  original_local_snapshots = local_snapshots.copy()
  shifted_local_snapshots = local_snapshots.copy()
  shifted_local_snapshots -= shift_value
  shifter = create_constant_shifter(shift_value, local_snapshots)
  shifter.apply_shift(local_snapshots)
  assert np.allclose(local_snapshots, shifted_local_snapshots)
  shifter.apply_inverse_shift(local_snapshots)
  assert np.allclose(local_snapshots, original_local_snapshots)

@pytest.mark.mpi_skip
def test_average_shifter():
  my_snapshots = np.random.normal(size=(3,10,5))
  mean_vec = np.mean(my_snapshots,axis=2)
  original_snapshots = my_snapshots.copy()
  shifter = create_average_shifter(my_snapshots)
  shifter.apply_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots - mean_vec[:,:,None]))
  assert(np.allclose(np.mean(my_snapshots, axis=2), 0))
  shifter.apply_inverse_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots))

@pytest.mark.mpi(min_size=3)
def test_average_shifter_mpi():
  comm=MPI.COMM_WORLD
  global_shape = (3,10,5)
  local_snapshots, global_snapshots = test_utils.generate_random_local_and_global_arrays_impl(global_shape, comm=comm)
  original_local_snapshots = local_snapshots.copy()
  mean_vec = np.mean(global_snapshots, axis=2)
  local_shifter = create_average_shifter(local_snapshots)
  local_shifter.apply_shift(local_snapshots)
  assert len(np.setdiff1d(local_snapshots, global_snapshots - mean_vec[:,:,None])) == 0
  local_shifter.apply_inverse_shift(local_snapshots)
  assert np.allclose(local_snapshots, original_local_snapshots)

@pytest.mark.mpi_skip
def test_first_vec_shifter():
  my_snapshots = np.random.normal(size=(3,10,5))
  original_snapshots = my_snapshots.copy()
  first_vec = original_snapshots[:,:,0]
  shifter = create_firstvec_shifter(my_snapshots)
  shifter.apply_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots - first_vec[:,:,None]))
  assert(my_snapshots.shape[2] == original_snapshots.shape[2])
  shifter.apply_inverse_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots))

@pytest.mark.mpi(min_size=3)
def test_first_vec_shifter():
  comm = MPI.COMM_WORLD
  global_shape = (3,10,5)
  local_snapshots, global_snapshots = test_utils.generate_random_local_and_global_arrays_impl(global_shape, comm=comm)
  original_local_snapshots = local_snapshots.copy()
  first_vec = global_snapshots[:,:,0]
  shifter = create_firstvec_shifter(local_snapshots)
  shifter.apply_shift(local_snapshots)
  assert(len(np.setdiff1d(local_snapshots, global_snapshots - first_vec[:,:,None])) == 0)
  assert(local_snapshots.shape[2] == global_snapshots.shape[2])
  shifter.apply_inverse_shift(local_snapshots)
  assert(np.allclose(local_snapshots, original_local_snapshots))

@pytest.mark.mpi_skip
def test_vector_shifter():
  shift_vec = np.random.normal(size=(3,10))
  my_snapshots = np.random.normal(size=(3,10,5))
  original_snapshots = my_snapshots.copy()
  shifter = create_vector_shifter(shift_vec)
  shifter.apply_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots - shift_vec[:,:,None]))
  assert(my_snapshots.shape[2] == original_snapshots.shape[2])
  shifter.apply_inverse_shift(my_snapshots)
  assert(np.allclose(my_snapshots, original_snapshots))

@pytest.mark.mpi(min_size=3)
def test_vector_shifter_mpi():
  comm = MPI.COMM_WORLD
  global_shape = (3,10,5)
  shift_vec = np.random.normal(size=(3,5))
  original_local_snapshots, global_snapshots = test_utils.generate_random_local_and_global_arrays_impl(global_shape, comm=comm)
  local_snapshots = original_local_snapshots.copy()
  shifter = create_vector_shifter(shift_vec)
  shifter.apply_shift(local_snapshots)
  global_snapshots -= shift_vec[:,None,:]
  assert len(np.setdiff1d(local_snapshots, global_snapshots)) == 0
  shifter.apply_inverse_shift(local_snapshots)
  assert np.allclose(local_snapshots, original_local_snapshots)


if __name__=="__main__":
  test_noop_shifter()
  test_noop_shifter_mpi()
  test_constant_shifter()
  test_constant_shifter_mpi
  test_average_shifter()
  test_average_shifter_mpi()
  test_first_vec_shifter()
  test_first_vec_shifter_mpi()
  test_vector_shifter()
  test_vector_shifter_mpi()
