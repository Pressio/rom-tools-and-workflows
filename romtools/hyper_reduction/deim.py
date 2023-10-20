import numpy as np

'''
Implementation of DEIM technique for hyper-reduction
'''

def __getDataMatrixForIthVar(i,n_var,data_matrix,variable_ordering):
  """helper function to split data"""
  if variable_ordering == 'F':
    return data_matrix[i::n_var]
  elif variable_ordering == 'C':
    n = int( data_matrix.shape[0] / n_var)
    start_index = i*n
    end_index = (i+1)*n
    return data_matrix[start_index:end_index]

def deimGetApproximationMatrix(functionBasis,sampleIndices):
  '''
  Given a function basis $\\mathbf{U}$ and sample indices defining $\\mathbf{P}$, we compute
  $$ \\mathbf{U} \\mathrm{pinv}( \\mathbf{P}^T \\mathbf{U})$$
  which comprises the matrix need for the DEIM approximation to $\\mathbf{f}$  '''
  sampledFunctionBasis = functionBasis[sampleIndices]
  PU_pinv = np.linalg.pinv(sampledFunctionBasis)
  result =  functionBasis @ PU_pinv  
  return result


def deimGetTestBasis(testBasis,functionBasis,sampleIndices):
  '''
  Given a test basis $\\mathbf{\\Phi}$, a function basis $\\mathbf{U}$, and sample indices defining $\\mathbf{P}$, we compute
  $$[ \\mathbf{\Phi}^T \\mathbf{U} \\mathrm{pinv}( \\mathbf{P}^T \\mathbf{U}) ]^T$$
  which comprises the "test basis" for the DEIM approximation fo $\\mathbf{\Phi}^T \\mathbf{f}$  '''
  sampledFunctionBasis = functionBasis[sampleIndices]
  PU_pinv = np.linalg.pinv(sampledFunctionBasis)
  result = (testBasis.transpose() @ functionBasis) @ PU_pinv  
  return result.transpose() 

def vectorDeimGetIndices(U,n_var,variable_ordering='F'):
  '''
  Version of DEIM for multi-state systems.
  We perform DEIM on each state variable, and 
  then return the union of all indices.
  Repeated indices are removed.
  '''
  all_indices = np.zeros(0,dtype=int)
  for i in range(0,n_var):
    dataMatrix = getDataMatrixForIthVar(i,n_var,U,variable_ordering)
    indices = deim(dataMatrix)
    all_indices = np.unique(np.append(all_indices,indices))
  return all_indices


def deimGetIndices(U):
  '''
  Implementation of the discrete empirical method as described in Algorithm 1 of 
  S. Chaturantabut and D. C. Sorensen, "Discrete Empirical Interpolation for nonlinear model reduction," doi: 10.1109/CDC.2009.5400045.  

  Inputs: $\\mathbf{U} \\in \\mathbb{R}^{m \\times n}$, where m is the number of DOFs and n the number of samples

  Outputs: $\\mathrm{indices} \\in \\mathbb{I}^{n}$ 
  '''

  m = np.shape(U)[1]
  n = np.shape(U)[0]
  first_index = np.argmax(np.abs(U[:,0]))
  indices = first_index
  for l in range(1,m):
    LHS = U[indices,0:l]
    RHS = U[indices,l]
    if l == 1:
      LHS = np.ones((1,1))*LHS
      RHS = np.ones(1)*RHS
    C = np.linalg.solve(LHS,RHS)

    residual = U[:,l] - U[:,0:l] @ C
    index_to_add = np.argmax(np.abs(residual))
    indices = np.append(indices,index_to_add)
  return indices 

