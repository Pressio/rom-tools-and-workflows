import numpy as np
from romtools.trial_space_utils.truncater import *
import pytest

@pytest.mark.mpi_skip
def test_noop_truncater():
  truncater = NoOpTruncater()
  my_basis = np.random.normal(size=(10,2))
  singular_vectors = np.ones(2)
  my_truncated_basis = truncater(my_basis,singular_vectors)
  assert(np.allclose(my_truncated_basis,my_basis))

@pytest.mark.mpi_skip
def test_basis_size_truncater():
  reduced_size = 4
  truncater = BasisSizeTruncater(reduced_size)
  my_basis = np.random.normal(size=(10,8))
  singular_values = np.ones(2)
  my_truncated_basis = truncater(my_basis,singular_values)
  assert(np.allclose(my_truncated_basis,my_basis[:,0:reduced_size]))
  assert(my_truncated_basis.shape[1] == 4)

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

  truncater = EnergyTruncater(energy_threshold)
  my_basis = np.random.normal(size=(10,8))
  my_truncated_basis = truncater(my_basis,singular_values)
  assert(np.allclose(my_truncated_basis,my_basis[:,0:K]))
  assert(my_truncated_basis.shape[1] == K)


if __name__=="__main__":
  test_noop_truncater()
  test_basis_size_truncater()
  test_energy_truncater()
