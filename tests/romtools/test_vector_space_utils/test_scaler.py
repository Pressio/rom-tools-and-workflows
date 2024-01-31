import copy
import pytest
import numpy as np
import scipy.sparse
from romtools.vector_space.utils.scaler import *


@pytest.mark.mpi_skip
def test_noop_scaler():
  scaler = NoOpScaler()
  my_snapshots = np.random.normal(size=(3,10,2))
  my_scaled_snapshots = scaler.pre_scaling(my_snapshots)
  my_unscaled_snapshots = scaler.post_scaling(my_scaled_snapshots)
  assert(np.allclose(my_snapshots,my_scaled_snapshots))
  assert(np.allclose(my_scaled_snapshots,my_unscaled_snapshots))

def scaling_op(scaling_type,arg):
  if scaling_type == 'max_abs':
    return np.amax(np.abs(arg))
  elif scaling_type == 'mean_abs':
    return np.mean(np.abs(arg))
  elif scaling_type == 'variance':
    return np.std(arg)

@pytest.mark.mpi_skip
def test_vector_scaler():
  n_var = 3
  nx = 5
  my_snapshots = np.random.normal(size=(n_var,nx,5))
  my_scaling_vector = np.abs(np.random.normal(size=nx))
  my_initial_snapshots = copy.deepcopy(my_snapshots)
  scaler = VectorScaler(my_scaling_vector)

  my_scaled_snapshots = scaler.pre_scaling(my_snapshots)

  assert np.allclose(my_scaled_snapshots , 1./my_scaling_vector[None,:,None] * my_initial_snapshots)
  my_unscaled_snapshots = scaler.post_scaling(my_scaled_snapshots)
  assert(np.allclose(my_initial_snapshots,my_unscaled_snapshots))

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
  my_scaled_snapshots = scaler.pre_scaling(my_snapshots)
  for i in range(0,n_var):
    assert np.allclose(my_scaled_snapshots[i] , 1./scales[i]*my_initial_snapshots[i])

  my_unscaled_snapshots = scaler.post_scaling(my_scaled_snapshots)
  assert(np.allclose(scales,scaler.var_scales_))
  assert(np.allclose(my_initial_snapshots,my_unscaled_snapshots))
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
  my_scaled_snapshots = scaler.pre_scaling(my_snapshots)
  for i in range(0,n_var):
    assert np.allclose(my_scaled_snapshots[i] , 1./scales[i]*(1./my_scaling_vector[None,:,None]* my_initial_snapshots)[i])

  my_unscaled_snapshots = scaler.post_scaling(my_scaled_snapshots)
  assert(np.allclose(my_initial_snapshots,my_unscaled_snapshots))

 run_test('max_abs')
 run_test('mean_abs')
 run_test('variance')


if __name__=="__main__":
  test_noop_scaler()
  test_vector_scaler()
  test_variable_scaler()
  test_variable_and_vector_scaler()
