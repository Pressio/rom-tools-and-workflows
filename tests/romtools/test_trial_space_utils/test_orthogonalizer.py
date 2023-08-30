import numpy as np
from romtools.trial_space_utils.orthogonalizer import *
import scipy.sparse
def test_noop_orthogonalizer():
  orthogonalizer = NoOpOrthogonalizer()
  my_basis = np.random.normal(size=(10,2)) 
  my_orthogonalized_basis = orthogonalizer(my_basis)
  assert(np.allclose(my_orthogonalized_basis,my_basis))

def test_euclidean_l2_orthogonalizer():
  orthogonalizer = EuclideanL2Orthogonalizer()
  my_basis = np.random.normal(size=(10,2)) 
  my_orthogonalized_basis = orthogonalizer(my_basis)
  should_be_eye = my_orthogonalized_basis.transpose() @ my_orthogonalized_basis 
  assert(np.allclose(should_be_eye,np.eye(2)))


def test_euclidean_vector_weighted_l2_orthogonalizer():
  np.random.seed(1)
  vec_to_orthogonalize_against = np.abs(np.random.normal(size=10))
  orthogonalizer = EuclideanVectorWeightedL2Orthogonalizer(vec_to_orthogonalize_against)
  my_basis = np.random.normal(size=(10,2)) 
  my_orthogonalized_basis = orthogonalizer(my_basis)
  should_be_eye = my_orthogonalized_basis.transpose() @ ( scipy.sparse.diags(vec_to_orthogonalize_against) @ my_orthogonalized_basis ) 
  assert np.allclose(should_be_eye,np.eye(2)), should_be_eye


if __name__=="__main__":
  test_noop_orthogonalizer()
  test_euclidean_l2_orthogonalizer()
  test_euclidean_vector_weighted_l2_orthogonalizer()
