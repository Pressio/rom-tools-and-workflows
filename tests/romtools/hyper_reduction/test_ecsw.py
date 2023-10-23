import numpy as np
from romtools.hyper_reduction.ecsw import *


def test_ecsw_nnls():
  ## test NNLS routine for small random matrix
  G = np.random.normal(size=(5,10))
  b = np.sum(G,axis=1)

  nnls = ECSWsolverNNLS()
  inds,xi = nnls(G,b,1e-20)

  ## test weights from NNLS
  assert(np.allclose(G@xi,b))

  ## test indices from NNLS
  G_red = G[:,inds]
  xi_red = xi[inds]

  assert(np.allclose(G@xi,G_red@xi_red))

def test_ecsw_matrix():

  ## test matrix construction for scalar case
  Res = np.random.normal(size=(10,5))
  Psi = np.random.normal(size=(10,3))

  G,b=ConstructLinearSystemForECSWFixedTestBasis(Res, Psi, 1, 'C')

  ## Check left-hand-side
  assert(np.allclose((np.sum(G,axis=1)).reshape((3,5),order='F'),(Psi.T)@Res))

  ## Check right-hand-side 
  assert(np.allclose(b.reshape((3,5),order='F'),(Psi.T)@Res))

def test_full_ecsw():
  ## test ECSW
  nnls = ECSWsolverNNLS()
  Res = np.random.normal(size=(10,5))
  Psi = np.random.normal(size=(10,3))

  inds,xi = ECSWFixedTestBasis(nnls, Res, Psi, 1, 'C', 1e-2)

  ## Check full approximation
  exact = (Psi.T)@Res
  approx = (Psi.T)@(np.diag(xi)@Res)

  assert(np.allclose(exact,approx))

  ## Check indices
  Psi_sm = Psi[inds,:]
  Res_sm = Res[inds,:]
  xi_sm = xi[inds]

  approx_sm = (Psi_sm.T)@(np.diag(xi_sm)@Res_sm)

  assert(np.allclose(approx_sm,approx))


  
  

if __name__=="__main__":
    test_full_ecsw()
    test_ecsw_nnls()
    test_ecsw_matrix()
