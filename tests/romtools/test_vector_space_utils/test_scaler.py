import copy
import pytest
import numpy as np
import scipy.sparse
from romtools.vector_space.utils.scaler import *
import pressiolinalg.linalg as pla
from pressiolinalg import test_utils
try:
  import mpi4py
  from mpi4py import MPI
except ModuleNotFoundError:
  print("module 'mpi4py' is not installed")


@pytest.mark.mpi_skip
def test_noop_scaler():
  scaler = NoOpScaler()
  my_snapshots = np.random.normal(size=(3,10,2))
  my_scaled_snapshots = scaler.pre_scale(my_snapshots)
  my_unscaled_snapshots = scaler.post_scale(my_scaled_snapshots)
  assert(np.allclose(my_snapshots,my_scaled_snapshots))
  assert(np.allclose(my_scaled_snapshots,my_unscaled_snapshots))

@pytest.mark.mpi(min_size=3)
def test_noop_scaler_mpi():
  comm = MPI.COMM_WORLD
  scaler = NoOpScaler()
  local_snapshots, _ = test_utils.generate_random_local_and_global_arrays_impl((3,10,2), comm=comm)
  scaled_local_snapshots = scaler.pre_scale(local_snapshots)
  unscaled_local_snapshots = scaler.post_scale(scaled_local_snapshots)
  assert(np.allclose(local_snapshots, scaled_local_snapshots))
  assert(np.allclose(scaled_local_snapshots, unscaled_local_snapshots))

def scaling_op(scaling_type,arg):
  if scaling_type == 'max_abs':
    return pla.max(np.abs(arg))
  elif scaling_type == 'mean_abs':
    return pla.mean(np.abs(arg))
  elif scaling_type == 'variance':
    return pla.std(arg)

@pytest.mark.mpi_skip
def test_vector_scaler():
  n_var = 3
  nx = 5
  my_snapshots = np.random.normal(size=(n_var,nx,5))
  my_scaling_vector = np.abs(np.random.normal(size=nx))
  my_initial_snapshots = copy.deepcopy(my_snapshots)
  scaler = VectorScaler(my_scaling_vector)

  my_scaled_snapshots = scaler.pre_scale(my_snapshots)

  assert np.allclose(my_scaled_snapshots , 1./my_scaling_vector[None,:,None] * my_initial_snapshots)
  my_unscaled_snapshots = scaler.post_scale(my_scaled_snapshots)
  assert(np.allclose(my_initial_snapshots,my_unscaled_snapshots))

@pytest.mark.mpi(min_size=3)
def test_vector_scaler_mpi():
  comm = MPI.COMM_WORLD
  n_var = 3
  nx = 5
  local_snapshots, _ = test_utils.generate_random_local_and_global_arrays_impl((n_var,nx,5), comm=comm)
  my_scaling_vector = np.abs(np.random.normal(size=local_snapshots.shape[1]))
  initial_local_snapshots = copy.deepcopy(local_snapshots)
  scaler = VectorScaler(my_scaling_vector)

  scaled_local_snapshots = scaler.pre_scale(local_snapshots)

  assert np.allclose(scaled_local_snapshots , 1./my_scaling_vector[None,:,None] * initial_local_snapshots)
  unscaled_local_snapshots = scaler.post_scale(scaled_local_snapshots)
  assert(np.allclose(initial_local_snapshots, unscaled_local_snapshots))

@pytest.mark.mpi_skip
def test_variable_scaler():
 n_var = 3
 nx = 5
 def run_test(scaling_type):
  my_snapshots = np.random.normal(size=(n_var,nx,5))
  scales = np.zeros(n_var)
  for i in range(0,n_var):
    scales[i] = scaling_op(scaling_type,my_snapshots[i])

  my_initial_snapshots = copy.deepcopy(my_snapshots)
  scaler = VariableScaler(scaling_type)
  my_scaled_snapshots = scaler.pre_scale(my_snapshots)
  for i in range(0,n_var):
    assert np.allclose(my_scaled_snapshots[i] , 1./scales[i]*my_initial_snapshots[i])

  my_unscaled_snapshots = scaler.post_scale(my_scaled_snapshots)
  assert(np.allclose(scales,scaler.var_scales_))
  assert(np.allclose(my_initial_snapshots,my_unscaled_snapshots))
 run_test('max_abs')
 run_test('mean_abs')
 run_test('variance')

@pytest.mark.mpi(min_size=3)
def test_variable_scaler_mpi():
 comm = MPI.COMM_WORLD
 n_var = 3
 nx = 5
 def run_test(scaling_type):
  local_snapshots, _ = test_utils.generate_random_local_and_global_arrays_impl((n_var,nx,5), comm=comm)
  scales = np.zeros(local_snapshots.shape[0])
  for i in range(0,local_snapshots.shape[0]):
    scales[i] = scaling_op(scaling_type, local_snapshots[i])

  initial_local_snapshots = copy.deepcopy(local_snapshots)
  scaler = VariableScaler(scaling_type)
  scaled_local_snapshots = scaler.pre_scale(local_snapshots)
  for i in range(0,local_snapshots.shape[0]):
    assert np.allclose(scaled_local_snapshots[i] , 1./scales[i]*initial_local_snapshots[i])

  unscaled_local_snapshots = scaler.post_scale(scaled_local_snapshots)
  assert(np.allclose(scales,scaler.var_scales_))
  assert(np.allclose(initial_local_snapshots,unscaled_local_snapshots))
 run_test('max_abs')
 run_test('mean_abs')
 run_test('variance')

@pytest.mark.mpi_skip
def test_variable_and_vector_scaler():
 def run_test(scaling_type):
  n_var = 3
  nx = 5
  my_snapshots = np.random.normal(size=(n_var,nx,5))
  my_scaling_vector = np.abs(np.random.normal(size=nx))

  scales = np.zeros(n_var)
  for i in range(0,n_var):
    scales[i] = scaling_op(scaling_type,my_snapshots[i])

  my_initial_snapshots = copy.deepcopy(my_snapshots)
  scaler = VariableAndVectorScaler(my_scaling_vector,scaling_type)
  my_scaled_snapshots = scaler.pre_scale(my_snapshots)
  for i in range(0,n_var):
    assert np.allclose(my_scaled_snapshots[i] , 1./scales[i]*(1./my_scaling_vector[None,:,None]* my_initial_snapshots)[i])

  my_unscaled_snapshots = scaler.post_scale(my_scaled_snapshots)
  assert(np.allclose(my_initial_snapshots,my_unscaled_snapshots))

 run_test('max_abs')
 run_test('mean_abs')
 run_test('variance')

@pytest.mark.mpi(min_size=3)
def test_variable_and_vector_scaler_mpi():
 def run_test(scaling_type):
  comm = MPI.COMM_WORLD
  n_var = 3
  nx = 5
  local_snapshots, global_snapshots = test_utils.generate_random_local_and_global_arrays_impl((n_var,nx,5), comm=comm)
  my_scaling_vector = np.abs(np.random.normal(size=local_snapshots.shape[1]))

  scales = np.zeros(local_snapshots.shape[0])
  for i in range(0,local_snapshots.shape[0]):
    scales[i] = scaling_op(scaling_type, local_snapshots[i])

  initial_local_snapshots = copy.deepcopy(local_snapshots)
  scaler = VariableAndVectorScaler(my_scaling_vector,scaling_type)
  scaled_local_snapshots = scaler.pre_scale(local_snapshots)
  for i in range(0,local_snapshots.shape[0]):
    assert np.allclose(scaled_local_snapshots[i] , 1./scales[i]*(1./my_scaling_vector[None,:,None]* initial_local_snapshots)[i])

  unscaled_local_snapshots = scaler.post_scale(scaled_local_snapshots)
  assert(np.allclose(initial_local_snapshots, unscaled_local_snapshots))

 run_test('max_abs')
 run_test('mean_abs')
 run_test('variance')


if __name__=="__main__":
  test_noop_scaler()
  test_noop_scaler_mpi()
  test_vector_scaler()
  test_vector_scaler_mpi()
  test_variable_scaler()
  test_variable_scaler_mpi()
  test_variable_and_vector_scaler()
