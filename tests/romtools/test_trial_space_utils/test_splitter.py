import pytest
import numpy as np
from romtools.trial_space.utils.splitter import *


@pytest.mark.mpi_skip
def test_noop_splitter():
  splitter = NoOpSplitter()
  my_basis = np.random.normal(size=(10,2))
  my_split_basis = splitter(my_basis)
  assert(np.allclose(my_basis,my_split_basis))

@pytest.mark.mpi_skip
def test_block_splitter_order_f():
  n_var = 3
  nx = 5
  n = int(n_var*nx)
  my_basis = np.random.normal(size=(n,2))

  blocks = [[0,1,2]]
  variable_ordering = 'F'
  splitter = BlockSplitter(blocks,n_var,variable_ordering)
  my_split_basis = splitter(my_basis)
  assert np.allclose(my_split_basis,my_basis)

  blocks = [[0],[1],[2]]
  variable_ordering = 'F'
  splitter = BlockSplitter(blocks,n_var,variable_ordering)
  my_split_basis = splitter(my_basis)
  assert(my_split_basis.shape[1] == 6)
  assert(np.allclose( np.linalg.norm(my_split_basis), np.linalg.norm(my_basis)))
  assert(np.allclose(my_split_basis[0::n_var,0:2],my_basis[0::n_var,0:2]))
  assert(np.allclose(my_split_basis[1::n_var,0:2],0))
  assert(np.allclose(my_split_basis[2::n_var,0:2],0))
  assert(np.allclose(my_split_basis[0::n_var,2:4],0))
  assert(np.allclose(my_split_basis[1::n_var,2:4],my_basis[1::n_var,0:2]))
  assert(np.allclose(my_split_basis[2::n_var,2:4],0))
  assert(np.allclose(my_split_basis[0::n_var,4:6],0))
  assert(np.allclose(my_split_basis[2::n_var,4:6],my_basis[2::n_var,0:2]))
  assert(np.allclose(my_split_basis[1::n_var,4:6],0))

  blocks = [[0],[2],[1]]
  variable_ordering = 'F'
  splitter = BlockSplitter(blocks,n_var,variable_ordering)
  my_split_basis = splitter(my_basis)

  assert(my_split_basis.shape[1] == 6)
  assert(np.allclose(np.linalg.norm(my_split_basis),np.linalg.norm(my_basis)))
  assert(np.allclose(my_split_basis[0::n_var,0:2],my_basis[0::n_var,0:2]))
  assert(np.allclose(my_split_basis[1::n_var,0:2],0))
  assert(np.allclose(my_split_basis[2::n_var,0:2],0))
  assert(np.allclose(my_split_basis[0::n_var,4:6],0))
  assert(np.allclose(my_split_basis[1::n_var,4:6],my_basis[1::n_var,0:2]))
  assert(np.allclose(my_split_basis[2::n_var,4:6],0))
  assert(np.allclose(my_split_basis[0::n_var,2:4],0))
  assert(np.allclose(my_split_basis[2::n_var,2:4],my_basis[2::n_var,0:2]))
  assert(np.allclose(my_split_basis[1::n_var,2:4],0))

  blocks = [[0],[2,1]]
  variable_ordering = 'F'
  splitter = BlockSplitter(blocks,n_var,variable_ordering)
  my_split_basis = splitter(my_basis)

  assert(my_split_basis.shape[1] == 4)
  assert(np.allclose(np.linalg.norm(my_split_basis), np.linalg.norm(my_basis)))
  assert(np.allclose(my_split_basis[0::n_var,0:2],my_basis[0::n_var,0:2]))
  assert(np.allclose(my_split_basis[1::n_var,0:2],0))
  assert(np.allclose(my_split_basis[2::n_var,0:2],0))
  assert(np.allclose(my_split_basis[0::n_var,2:4],0))
  assert(np.allclose(my_split_basis[1::n_var,2:4],my_basis[1::n_var,0:2]))
  assert(np.allclose(my_split_basis[2::n_var,2:4],my_basis[2::n_var,0:2]))

@pytest.mark.mpi_skip
def test_block_splitter_order_c():
  n_var = 3
  n = 5
  N = int(n_var*n)
  my_basis = np.random.normal(size=(N,2))

  blocks = [[0,1,2]]
  variable_ordering = 'C'
  splitter = BlockSplitter(blocks,n_var,variable_ordering)
  my_split_basis = splitter(my_basis)
  assert np.allclose(my_split_basis,my_basis)

  blocks = [[0],[1],[2]]
  variable_ordering = 'C'
  splitter = BlockSplitter(blocks,n_var,variable_ordering)
  my_split_basis = splitter(my_basis)
  assert(my_split_basis.shape[1] == 6)
  assert(np.allclose(np.linalg.norm(my_split_basis),np.linalg.norm(my_basis)))
  assert(np.allclose(my_split_basis[0:n,0:2],my_basis[0:n,0:2]))
  assert(np.allclose(my_split_basis[n:2*n,0:2],0))
  assert(np.allclose(my_split_basis[2*n:3*n,0:2],0))
  assert(np.allclose(my_split_basis[0:n,2:4],0))
  assert(np.allclose(my_split_basis[n:2*n,2:4],my_basis[n:2*n,0:2]))
  assert(np.allclose(my_split_basis[2*n:3*n,2:4],0))
  assert(np.allclose(my_split_basis[0:n,4:6],0))
  assert(np.allclose(my_split_basis[n:2*n,4:6],0))
  assert(np.allclose(my_split_basis[2*n:3*n,4:6],my_basis[2*n:3*n,0:2]))

  blocks = [[0],[2],[1]]
  variable_ordering = 'C'
  splitter = BlockSplitter(blocks,n_var,variable_ordering)
  my_split_basis = splitter(my_basis)

  assert(my_split_basis.shape[1] == 6)
  assert(np.allclose(np.linalg.norm(my_split_basis), np.linalg.norm(my_basis)))
  assert(np.allclose(my_split_basis[0:n,0:2],my_basis[0:n,0:2]))
  assert(np.allclose(my_split_basis[n:2*n,0:2],0))
  assert(np.allclose(my_split_basis[2*n:3*n,0:2],0))
  assert(np.allclose(my_split_basis[0:n,4:6],0))
  assert(np.allclose(my_split_basis[n:2*n,4:6],my_basis[n:2*n,0:2]))
  assert(np.allclose(my_split_basis[2*n:3*n,4:6],0))
  assert(np.allclose(my_split_basis[0:n,2:4],0))
  assert(np.allclose(my_split_basis[2*n:3*n,2:4],my_basis[2*n:3*n,0:2]))
  assert(np.allclose(my_split_basis[n:2*n,2:4],0))

  blocks = [[0],[2,1]]
  variable_ordering = 'C'
  splitter = BlockSplitter(blocks,n_var,variable_ordering)
  my_split_basis = splitter(my_basis)

  assert(my_split_basis.shape[1] == 4)
  assert(np.allclose(np.linalg.norm(my_split_basis),np.linalg.norm(my_basis)))
  assert(np.allclose(my_split_basis[0:n,0:2],my_basis[0:n,0:2]))
  assert(np.allclose(my_split_basis[n:2*n,0:2],0))
  assert(np.allclose(my_split_basis[2*n:3*n,0:2],0))
  assert(np.allclose(my_split_basis[0:n,2:4],0))
  assert(np.allclose(my_split_basis[n:2*n,2:4],my_basis[n:2*n,0:2]))
  assert(np.allclose(my_split_basis[2*n:3*n,2:4],my_basis[2*n:3*n,0:2]))




if __name__=="__main__":
  test_noop_splitter()
  test_block_splitter_order_f()
  test_block_splitter_order_c()
