import pytest
import numpy as np
import scipy.sparse
from romtools.vector_space.utils.orthogonalizer import *
from pressiolinalg import test_utils
try:
  import mpi4py
  from mpi4py import MPI
except ModuleNotFoundError:
  print("module 'mpi4py' is not installed")


@pytest.mark.mpi_skip
def test_noop_orthogonalizer():
  orthogonalizer = NoOpOrthogonalizer()
  my_basis = np.random.normal(size=(10,2))
  my_orthogonalized_basis = orthogonalizer.orthogonalize(my_basis)
  assert(np.allclose(my_orthogonalized_basis,my_basis))

@pytest.mark.mpi(min_size=3)
def test_noop_orthogonalizer_mpi():
  comm = MPI.COMM_WORLD
  orthogonalizer = NoOpOrthogonalizer()
  basis_shape = (10,2)
  local_basis, global_basis = test_utils.generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
  local_orthogonalized_basis = orthogonalizer.orthogonalize(local_basis)
  assert(np.allclose(local_orthogonalized_basis, local_basis))

@pytest.mark.mpi_skip
def test_euclidean_l2_orthogonalizer():
  orthogonalizer = EuclideanL2Orthogonalizer()
  my_basis = np.random.normal(size=(10,2))
  my_orthogonalized_basis = orthogonalizer.orthogonalize(my_basis)
  should_be_eye = my_orthogonalized_basis.transpose() @ my_orthogonalized_basis
  assert(np.allclose(should_be_eye,np.eye(2)))

@pytest.mark.mpi(min_size=3)
def test_euclidean_l2_orthogonalizer_mpi():
  comm = MPI.COMM_WORLD
  orthogonalizer = EuclideanL2Orthogonalizer()
  basis_shape = (10,2)
  local_basis, global_basis = test_utils.generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
  local_orthogonalized_basis = orthogonalizer.orthogonalize(local_basis)
  should_be_eye = local_orthogonalized_basis.transpose() @ local_orthogonalized_basis
  assert(np.allclose(should_be_eye,np.eye(2)))

@pytest.mark.mpi_skip
def test_euclidean_vector_weighted_l2_orthogonalizer():
  np.random.seed(1)
  vec_to_orthogonalize_against = np.abs(np.random.normal(size=10))
  orthogonalizer = EuclideanVectorWeightedL2Orthogonalizer(vec_to_orthogonalize_against)
  my_basis = np.random.normal(size=(10,2))
  my_orthogonalized_basis = orthogonalizer.orthogonalize(my_basis)
  should_be_eye = my_orthogonalized_basis.transpose() @ ( scipy.sparse.diags(vec_to_orthogonalize_against) @ my_orthogonalized_basis )
  assert np.allclose(should_be_eye,np.eye(2)), should_be_eye

@pytest.mark.mpi(min_size=3)
def test_euclidean_vector_weighted_l2_orthogonalizer_mpi():
  comm = MPI.COMM_WORLD
  basis_shape = (10,2)
  local_basis, global_basis = test_utils.generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
  np.random.seed(1)
  vec_to_orthogonalize_against = np.abs(np.random.normal(size=local_basis.shape[0]))
  orthogonalizer = EuclideanVectorWeightedL2Orthogonalizer(vec_to_orthogonalize_against)
  local_orthogonalized_basis = orthogonalizer.orthogonalize(local_basis)
  should_be_eye = local_orthogonalized_basis.transpose() @ ( scipy.sparse.diags(vec_to_orthogonalize_against) @ local_orthogonalized_basis )
  assert np.allclose(should_be_eye,np.eye(2)), should_be_eye

if __name__=="__main__":
  test_noop_orthogonalizer()
  test_noop_orthogonalizer_mpi()
  test_euclidean_l2_orthogonalizer()
  test_euclidean_l2_orthogonalizer_mpi()
  test_euclidean_vector_weighted_l2_orthogonalizer()
  test_euclidean_vector_weighted_l2_orthogonalizer_mpi()
