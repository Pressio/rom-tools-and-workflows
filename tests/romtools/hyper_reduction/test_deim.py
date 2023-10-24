import numpy as np
from romtools.hyper_reduction.deim import *


def test_deim_approximation():
  U = np.random.normal(size=(10,5))
  indices = np.arange(0,5)
  np.random.shuffle(indices)
  Uhat = deimGetApproximationMatrix(U,indices)
  ## Check exact reconstruction for a function in our basis
  U_approx = Uhat @ U[indices,0]
  assert(np.allclose(U_approx,U[:,0])) 
 
  ## Check projections are correct
  Phi = np.random.normal(size=(10,3))
  deimPhi = deimGetTestBasis(Phi,U,indices)  
  assert(np.allclose(deimPhi,(Phi.transpose() @ Uhat).transpose()))


def test_deim_basis():
  U = np.random.normal(size=(10,5))
  Phi = np.random.normal(size=(10,3))

  indices = deimGetIndices(U)
  ## Test over sampling
  for i in range(0,10):
    if i in indices:
      pass
    else:
      indices = np.append(indices,i)
      break
  assert(indices.size==U.shape[1]+1)
  deimPhi = deimGetTestBasis(Phi,U,indices)  
  assert(deimPhi.shape[0] == indices.size)
  assert(deimPhi.shape[1] == Phi.shape[1])

def test_full_deim():
  U = np.random.normal(size=(5,5))
  indices = deimGetIndices(U)
  ## Confirm that we get all indices
  assert(indices.size == 5)
  ## Confirm test index
  assert(indices[0] == np.argmax(np.abs(U[:,0])))
  assert(np.allclose(np.sort(indices),np.arange(0,5))) 
  
  

if __name__=="__main__":
    test_full_deim()
    test_deim_basis()
    test_deim_approximation()
