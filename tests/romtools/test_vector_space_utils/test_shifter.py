import pytest
import numpy as np
from romtools.vector_space.utils.shifter import *


@pytest.mark.mpi_skip
def test_noop_shifter():
  my_snapshots = np.random.normal(size=(3,10,2))
  original_snapshots = my_snapshots.copy()
  shifter = create_noop_shifter(my_snapshots)
  shifter.apply_shift()
  assert(np.allclose(my_snapshots, original_snapshots))
  shifter.apply_inverse_shift()
  assert(np.allclose(my_snapshots, original_snapshots))

@pytest.mark.mpi_skip
def test_constant_shifter():
  shift_value = np.array([4,1,3],dtype='int')
  my_snapshots = np.random.normal(size=(3,10,2))
  original_snapshots = my_snapshots.copy()
  shifter = create_constant_shifter(shift_value, my_snapshots)
  shifter.apply_shift()
  assert(np.allclose(my_snapshots, original_snapshots - shift_value[:,None,None]))
  shifter.apply_inverse_shift()
  assert(np.allclose(my_snapshots, original_snapshots))

@pytest.mark.mpi_skip
def test_average_shifter():
  my_snapshots = np.random.normal(size=(3,10,5))
  mean_vec = np.mean(my_snapshots,axis=2)
  original_snapshots = my_snapshots.copy()
  shifter = create_average_shifter(my_snapshots)
  shifter.apply_shift()
  assert(np.allclose(my_snapshots, original_snapshots - mean_vec[:,:,None]))
  assert(np.allclose(np.mean(my_snapshots, axis=2), 0))
  shifter.apply_inverse_shift()
  assert(np.allclose(my_snapshots, original_snapshots))

@pytest.mark.mpi_skip
def test_first_vec_shifter():
  my_snapshots = np.random.normal(size=(3,10,5))
  original_snapshots = my_snapshots.copy()
  first_vec = original_snapshots[:,:,0]
  shifter = create_firstvec_shifter(my_snapshots)
  shifter.apply_shift()
  assert(np.allclose(my_snapshots, original_snapshots - first_vec[:,:,None]))
  assert(my_snapshots.shape[2] == original_snapshots.shape[2])
  shifter.apply_inverse_shift()
  assert(np.allclose(my_snapshots, original_snapshots))

@pytest.mark.mpi_skip
def test_vector_shifter():
  shift_vec = np.random.normal(size=(3,10))
  my_snapshots = np.random.normal(size=(3,10,5))
  original_snapshots = my_snapshots.copy()
  shifter = create_vector_shifter(shift_vec, my_snapshots)
  shifter.apply_shift()
  assert(np.allclose(my_snapshots, original_snapshots - shift_vec[:,:,None]))
  assert(my_snapshots.shape[2] == original_snapshots.shape[2])
  shifter.apply_inverse_shift()
  assert(np.allclose(my_snapshots, original_snapshots))


if __name__=="__main__":
  test_noop_shifter()
  test_constant_shifter()
  test_average_shifter()
  test_first_vec_shifter()
  test_vector_shifter()
