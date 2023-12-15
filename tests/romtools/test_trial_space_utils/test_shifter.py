import pytest
import numpy as np
from romtools.trial_space_utils.shifter import *


@pytest.mark.mpi_skip
def test_noop_shifter():
  shifter = NoOpShifter()
  my_snapshots = np.random.normal(size=(3,10,2))
  my_snapshots_shifted,shift_vec = shifter(my_snapshots)
  assert(np.allclose(my_snapshots_shifted,my_snapshots))
  assert(np.allclose(shift_vec,0))
  assert(shift_vec.shape == my_snapshots.shape[0:2])

@pytest.mark.mpi_skip
def test_constant_shifter():
  shift_value = np.array([4,1,3],dtype='int')
  shifter = ConstantShifter(shift_value)
  my_snapshots = np.random.normal(size=(3,10,2))
  my_snapshots_shifted,shift_vec = shifter(my_snapshots)
  assert(np.allclose(my_snapshots_shifted,my_snapshots - shift_value[:,None,None]))
  assert(np.allclose(shift_vec,shift_value[:,None]))
  assert(shift_vec.shape == my_snapshots.shape[0:2])

@pytest.mark.mpi_skip
def test_average_shifter():
  shifter = AverageShifter()
  my_snapshots = np.random.normal(size=(3,10,5))
  mean_vec = np.mean(my_snapshots,axis=2)
  my_snapshots_shifted,shift_vec = shifter(my_snapshots)
  assert(np.allclose(my_snapshots_shifted,my_snapshots - mean_vec[:,:,None]))
  assert(np.allclose(shift_vec,mean_vec))
  assert(shift_vec.shape == my_snapshots.shape[0:2])
  assert(np.allclose(np.mean(my_snapshots_shifted,axis=2),0))

@pytest.mark.mpi_skip
def test_first_vec_shifter():
  shifter = FirstVecShifter()
  my_snapshots = np.random.normal(size=(3,10,5))
  first_vec = my_snapshots[:,:,0]
  my_snapshots_shifted,shift_vec = shifter(my_snapshots)
  assert(np.allclose(my_snapshots_shifted,my_snapshots[:,:,1::] - first_vec[:,:,None]))
  assert(np.allclose(shift_vec,first_vec))
  assert(shift_vec.shape[0] == my_snapshots.shape[0])
  assert(my_snapshots_shifted.shape[2] == my_snapshots.shape[2] - 1)

@pytest.mark.mpi_skip
def test_vector_shifter():
  shift_vec = np.random.normal(size=(3,10))
  shifter = VectorShifter(shift_vec)
  my_snapshots = np.random.normal(size=(3,10,5))
  my_snapshots_shifted,shift_vec2 = shifter(my_snapshots)
  assert(np.allclose(my_snapshots_shifted,my_snapshots - shift_vec[:,:,None]))
  assert(np.allclose(shift_vec,shift_vec2))
  assert(shift_vec.shape == my_snapshots.shape[0:2])
  assert(my_snapshots_shifted.shape[2] == my_snapshots.shape[2])


if __name__=="__main__":
  test_noop_shifter()
  test_constant_shifter()
  test_average_shifter()
  test_first_vec_shifter()
  test_vector_shifter()
