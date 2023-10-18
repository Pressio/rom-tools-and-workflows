import numpy as np
from romtools.trial_space_utils.scaler import *
import copy
import scipy.sparse
import pytest

@pytest.mark.mpi_skip
def test_noop_scaler():
  scaler = NoOpScaler()
  my_basis = np.random.normal(size=(10,2))
  my_scaled_basis = scaler.preScaling(my_basis)
  my_unscaled_basis = scaler.postScaling(my_scaled_basis)
  assert(np.allclose(my_basis,my_scaled_basis))
  assert(np.allclose(my_scaled_basis,my_unscaled_basis))

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
  variable_ordering = 'F'
  my_basis = np.random.normal(size=(int(n_var*nx),5))
  scales = np.zeros(n_var)
  my_scaling_vector = np.abs(np.random.normal(size=n_var*nx))
  my_initial_basis = copy.deepcopy(my_basis)
  scaler = VectorScaler(my_scaling_vector)

  my_scaled_basis = scaler.preScaling(my_basis)
  my_diag_scales = scipy.sparse.diags(my_scaling_vector)
  my_diag_scales_inv = scipy.sparse.diags(1./my_scaling_vector)

  assert np.allclose(my_scaled_basis , my_diag_scales_inv @ my_initial_basis)
  my_unscaled_basis = scaler.postScaling(my_scaled_basis)
  assert(np.allclose(my_initial_basis,my_unscaled_basis))

@pytest.mark.mpi_skip
def test_variable_scaler():
 def run_test(scaling_type):
  n_var = 3
  nx = 5
  variable_ordering = 'F'
  my_basis = np.random.normal(size=(int(n_var*nx),5))
  scales = np.zeros(n_var)
  for i in range(0,n_var):
    scales[i] = scaling_op(scaling_type,my_basis[i::n_var])

  my_initial_basis = copy.deepcopy(my_basis)
  scaler = VariableScaler(scaling_type,variable_ordering,n_var)
  my_scaled_basis = scaler.preScaling(my_basis)
  for i in range(0,n_var):
    assert np.allclose(my_scaled_basis[i::n_var] , 1./scales[i]*my_initial_basis[i::n_var])

  my_unscaled_basis = scaler.postScaling(my_scaled_basis)
  assert(np.allclose(scales,scaler.var_scales_))
  assert(np.allclose(my_initial_basis,my_unscaled_basis))

  ## Repeat for order C
  n_var = 3
  nx = 5
  variable_ordering = 'C'
  my_basis = np.random.normal(size=(int(n_var*nx),5))
  scales = np.zeros(n_var)
  for i in range(0,n_var):
    start = int(i*nx)
    end = int( (i+1)*nx )
    scales[i] = scaling_op(scaling_type,my_basis[start:end])

  my_initial_basis = copy.deepcopy(my_basis)
  scaler = VariableScaler(scaling_type,variable_ordering,n_var)
  my_scaled_basis = scaler.preScaling(my_basis)
  assert np.allclose(scales,scaler.var_scales_), print(scales,scaler.var_scales_)

  for i in range(0,n_var):
    start = int(i*nx)
    end = int( (i+1)*nx )
    assert np.allclose(my_scaled_basis[start:end] , 1./scales[i]*my_initial_basis[start:end])

  my_unscaled_basis = scaler.postScaling(my_scaled_basis)
  assert(np.allclose(my_initial_basis,my_unscaled_basis))
  assert(2 == 4)
 run_test('max_abs')
 run_test('mean_abs')
 run_test('variance')

@pytest.mark.mpi_skip
def test_variable_and_vector_scaler():
 def run_test(scaling_type):
  n_var = 3
  nx = 5
  variable_ordering = 'F'
  my_basis = np.random.normal(size=(int(n_var*nx),5))
  my_scaling_vector = np.abs(np.random.normal(size=n_var*nx))
  my_diag_scales = scipy.sparse.diags(my_scaling_vector)
  my_diag_scales_inv = scipy.sparse.diags(1./my_scaling_vector)

  scales = np.zeros(n_var)
  for i in range(0,n_var):
    scales[i] = scaling_op(scaling_type,my_basis[i::n_var])

  my_initial_basis = copy.deepcopy(my_basis)
  scaler = VariableAndVectorScaler(my_scaling_vector,scaling_type,variable_ordering,n_var)
  my_scaled_basis = scaler.preScaling(my_basis)
  for i in range(0,n_var):
    assert np.allclose(my_scaled_basis[i::n_var] , 1./scales[i]*(my_diag_scales_inv @ my_initial_basis)[i::n_var])

  my_unscaled_basis = scaler.postScaling(my_scaled_basis)
  assert(np.allclose(my_initial_basis,my_unscaled_basis))

  ## Repeat for order C
  n_var = 3
  nx = 5
  variable_ordering = 'C'
  my_basis = np.random.normal(size=(int(n_var*nx),5))
  scales = np.zeros(n_var)
  for i in range(0,n_var):
    start = int(i*nx)
    end = int( (i+1)*nx )
    scales[i] = scaling_op(scaling_type,my_basis[start:end])

  my_initial_basis = copy.deepcopy(my_basis)
  scaler = VariableAndVectorScaler(my_scaling_vector,scaling_type,variable_ordering,n_var)
  my_scaled_basis = scaler.preScaling(my_basis)

  for i in range(0,n_var):
    start = int(i*nx)
    end = int( (i+1)*nx )
    assert np.allclose(my_scaled_basis[start:end] , 1./scales[i]* (my_diag_scales_inv @ my_initial_basis)[start:end])

  my_unscaled_basis = scaler.postScaling(my_scaled_basis)
  assert(np.allclose(my_initial_basis,my_unscaled_basis))
 run_test('max_abs')
 run_test('mean_abs')
 run_test('variance')


if __name__=="__main__":
  test_noop_scaler()
  test_vector_scaler()
  test_variable_scaler()
  test_variable_and_vector_scaler()
