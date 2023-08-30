import numpy as np
from romtools.trial_space_utils.shifter import *
def test_noop_shifter():
  shifter = NoOpShifter()
  my_basis = np.random.normal(size=(10,2)) 
  my_basis_shifted,shift_vec = shifter(my_basis)
  assert(np.allclose(my_basis_shifted,my_basis))
  assert(np.allclose(shift_vec,0))
  assert(shift_vec.shape[0] == my_basis.shape[0])


def test_constant_shifter():
  shift_value = 4
  shifter = ConstantShifter(shift_value)
  my_basis = np.random.normal(size=(10,2)) 
  my_basis_shifted,shift_vec = shifter(my_basis)
  assert(np.allclose(my_basis_shifted,my_basis - shift_value))
  assert(np.allclose(shift_vec,shift_value))
  assert(shift_vec.shape[0] == my_basis.shape[0])

def test_average_shifter():
  shifter = AverageShifter()
  my_basis = np.random.normal(size=(10,5)) 
  mean_vec = np.mean(my_basis,axis=1)
  my_basis_shifted,shift_vec = shifter(my_basis)
  assert(np.allclose(my_basis_shifted,my_basis - mean_vec[:,None]))
  assert(np.allclose(shift_vec,mean_vec))
  assert(shift_vec.shape[0] == my_basis.shape[0])
  assert(np.allclose(np.mean(my_basis_shifted,axis=1),0))

def test_first_vec_shifter():
  shifter = FirstVecShifter()
  my_basis = np.random.normal(size=(10,5)) 
  first_vec = my_basis[:,0] 
  my_basis_shifted,shift_vec = shifter(my_basis)
  assert(np.allclose(my_basis_shifted,my_basis[:,1::] - first_vec[:,None]))
  assert(np.allclose(shift_vec,first_vec))
  assert(shift_vec.shape[0] == my_basis.shape[0])
  assert(my_basis_shifted.shape[1] == my_basis.shape[1] - 1)

def test_vector_shifter():
  shift_vec = np.random.normal(size=10)
  shifter = VectorShifter(shift_vec)
  my_basis = np.random.normal(size=(10,5)) 
  my_basis_shifted,shift_vec2 = shifter(my_basis)
  assert(np.allclose(my_basis_shifted,my_basis - shift_vec[:,None]))
  assert(np.allclose(shift_vec,shift_vec2))
  assert(shift_vec.shape[0] == my_basis.shape[0])
  assert(my_basis_shifted.shape[1] == my_basis.shape[1])


if __name__=="__main__":
  test_noop_shifter()
  test_constant_shifter()
  test_average_shifter()
  test_first_vec_shifter()
