import pytest
import numpy as np
from romtools.vector_space.utils.truncater import *
from pressiolinalg import test_utils
try:
  import mpi4py
  from mpi4py import MPI
except ModuleNotFoundError:
  print("module 'mpi4py' is not installed")


@pytest.mark.mpi_skip
def test_noop_truncater():
  truncater = NoOpTruncater()
  my_basis = np.random.normal(size=(10,2))
  singular_vectors = np.ones(2)
  my_truncated_basis = truncater.truncate(my_basis,singular_vectors)
  assert(np.allclose(my_truncated_basis,my_basis))

@pytest.mark.mpi(min_size=3)
def test_noop_truncater_mpi():
  comm=MPI.COMM_WORLD
  truncater = NoOpTruncater()
  basis_shape=(10,2)
  local_basis, global_basis = test_utils.generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
  singular_vectors = np.ones(2)
  local_truncated_basis = truncater.truncate(local_basis, singular_vectors)
  assert(np.allclose(local_truncated_basis, local_basis))

@pytest.mark.mpi_skip
def test_basis_size_truncater():
  reduced_size = 4
  truncater = BasisSizeTruncater(reduced_size)
  my_basis = np.random.normal(size=(10,8))
  singular_values = np.ones(2)
  my_truncated_basis = truncater.truncate(my_basis,singular_values)
  assert(np.allclose(my_truncated_basis,my_basis[:,0:reduced_size]))
  assert(my_truncated_basis.shape[1] == 4)

@pytest.mark.mpi(min_size=3)
def test_basis_size_truncater_mpi():
  comm = MPI.COMM_WORLD
  reduced_size = 4
  truncater = BasisSizeTruncater(reduced_size)
  my_basis = np.random.normal(size=(10,8))
  basis_shape=(10,8)
  local_basis, global_basis = test_utils.generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
  singular_values = np.ones(2)
  truncated_local_basis = truncater.truncate(local_basis,singular_values)
  assert(np.allclose(truncated_local_basis, local_basis[:,0:reduced_size]))
  assert(truncated_local_basis.shape[1] == 4)

@pytest.mark.mpi_skip
def test_energy_truncater():
  np.random.seed(1)
  energy_threshold = 0.65
  singular_values = np.random.normal(size=10)**2
  np.sort( singular_values  )
  energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
  K = 1
  for i in range(0,energy.size):
    if energy[i] < energy_threshold:
      K += 1

  truncater = EnergyBasedTruncater(energy_threshold)
  my_basis = np.random.normal(size=(10,8))
  my_truncated_basis = truncater.truncate(my_basis,singular_values)
  assert(np.allclose(my_truncated_basis,my_basis[:,0:K]))
  assert(my_truncated_basis.shape[1] == K)

@pytest.mark.mpi(min_size=3)
def test_energy_truncater_mpi():
  comm = MPI.COMM_WORLD
  np.random.seed(1)
  energy_threshold = 0.65
  singular_values = np.random.normal(size=10)**2
  np.sort( singular_values  )
  energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
  K = 1
  for i in range(0,energy.size):
    if energy[i] < energy_threshold:
      K += 1

  truncater = EnergyBasedTruncater(energy_threshold)
  basis_shape=(10,8)
  local_basis, global_basis = test_utils.generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
  local_truncated_basis = truncater.truncate(local_basis,singular_values)
  assert(np.allclose(local_truncated_basis, local_basis[:,0:K]))
  assert(local_truncated_basis.shape[1] == K)


if __name__=="__main__":
  test_noop_truncater()
  test_noop_truncater_mpi()
  test_basis_size_truncater()
  test_basis_size_truncater_mpi()
  test_energy_truncater()
  test_energy_truncater_mpi()
