import numpy as np
try:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
except ModuleNotFoundError:
  print("module 'mpi4py' is not installed")
  mpi_rank = 0
  num_processes = 1




class svdMethodOfSnapshots:
  def __init__(self,comm):
    self.comm = comm

  def __call__(self, snapshots: np.ndarray, full_matrices=False, compute_uv=False,hermitian=False):
    U,s = svdMethodOfSnapshotsImpl(snapshots,self.comm)
    return U,s,'not_computed_in_method_of_snapshots'


def svdMethodOfSnapshotsImpl(snapshots,comm):
  #
  # outputs:
  # modes, Phi: numpy array where each column is a POD mode
  # energy, sigma: energy associated with each mode (singular values)
  
  STS = __A_transpose_dot_b(snapshots,snapshots,comm)
  Lam,E = np.linalg.eig(STS)
  sigma = np.sqrt(Lam) 
  U = np.zeros(np.shape(snapshots) )
  U[:] = np.dot(snapshots, np.dot(E , np.diag(1./sigma)))
  ## sort by singular values
  ordering = np.argsort(sigma)[::-1]
  return U[:,ordering],sigma[ordering]



def __globalAbsSum(r):
  if (num_processes == 1):
    return np.sum(r)
  else:
    data = comm.gather(np.sum(np.abs(r)),root = 0)
    rn_glob = np.zeros(1)
    if (mpi_rank == 0):
      for j in range(0,num_processes):
        rn_glob[:] += data[j]
      for j in range(1,num_processes):
        comm.Send(rn_glob, dest=j)
    else:
      comm.Recv(rn_glob,source=0)
    return rn_glob[0]

## Helper functions will be moved to python mpi library at some point

def __A_transpose_dot_b(A,b,comm):
  '''
  Compute A^T A when A's columns are distributed
  '''
  mpi_rank = comm.Get_rank()
  num_processes = comm.Get_size()

  if (num_processes == 1):
    return np.dot(A.transpose(),b)
  else:
    tmp = np.dot(A.transpose(),b)

    data = comm.gather(tmp.flatten(),root = 0)

    ATb_glob = np.zeros(np.size(tmp))
    if (mpi_rank == 0):
      for j in range(0,num_processes):
        ATb_glob[:] += data[j]
      for j in range(1,num_processes):
        comm.Send(ATb_glob, dest=j)

    else:
        comm.Recv(ATb_glob, source=0)
    return np.reshape( ATb_glob , np.shape(tmp) )

