import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import copy
axis_font = {'size':'20'}
import matplotlib.pyplot as plt 
axis_font = {'size':16,'family':'serif'}
class MySparseMatrixBuilder:
  '''
  This class contains information for iteratively constructing a 
  sparse matrix
  '''
  def __init__(self,N,M):
    # constructor for an NxM matrix 
    self.N = N
    self.M = M

  #create containers for non-zero row indices,
  #column indices, and their values
  row_indices = np.zeros(0,dtype='int')
  col_indices = np.zeros(0,dtype='int')
  values = np.zeros(0)

  #Function to add a new entry to the matrix
  def addEntry(self,row_index,col_index,value):
    self.row_indices = np.append(self.row_indices,row_index)
    self.col_indices = np.append(self.col_indices,col_index)
    self.values = np.append(self.values,value)

  #Assemble the sparse matrix
  def assemble(self):
    SparseMatrix = scipy.sparse.csr_matrix((self.values,(self.row_indices,self.col_indices)),(self.N,self.M))
    return SparseMatrix

def BuildAdvectionMatrices(Nx,Ny,dx,dy):
  '''
  This function builds in the system matrix for 
  advection via a second order upwind (backward) difference

  For 2D, we run through x points left to right,
  and then y points up and down
  '''
  Ax_builder = MySparseMatrixBuilder(Nx*Ny,Nx*Ny)
  Ay_builder = MySparseMatrixBuilder(Nx*Ny,Nx*Ny)

  #first compute du/dx and du/dy on interior 
  for j in range(1,Ny):
    for i in range(1,Nx):
      indx = i + j*Nx
      indx_im1 = indx - 1
      indx_im2 = indx - 2
      indx_jm1 = indx - Nx
      indx_jm2 = indx - Nx*2

      Ax_builder.addEntry(indx,indx,3./2./dx)
      Ax_builder.addEntry(indx,indx_im1,-2./dx)
      if (i > 1):
        Ax_builder.addEntry(indx,indx_im2,0.5/dx) #if i=1, the value is zero from the BCs
      
      Ay_builder.addEntry(indx,indx,3./2./dy)
      Ay_builder.addEntry(indx,indx_jm1,-2./dy)
      if (j > 1):
        Ay_builder.addEntry(indx,indx_jm2,0.5/dy)

  # now compute interior x at first y
  for j in range(0,1):
    for i in range(1,Nx):
      indx = i + j*Nx
      indx_im1 = indx - 1
      indx_im2 = indx - 2
      indx_jm1 = indx - Nx
      indx_jm2 = indx - Nx*2

      Ax_builder.addEntry(indx,indx,3./2./dx)
      Ax_builder.addEntry(indx,indx_im1,-2./dx)
      if (i > 1):
        Ax_builder.addEntry(indx,indx_im2,0.5/dx) #if i=1, the value is zero from the BCs

      Ay_builder.addEntry(indx,indx,1./dy)

  # now compute interior y at first x
  for j in range(1,Ny):
    for i in range(0,1):
      indx = i + j*Nx
      indx_im1 = indx - 1
      indx_im2 = indx - 2
      indx_jm1 = indx - Nx
      indx_jm2 = indx - Nx*2

      Ax_builder.addEntry(indx,indx,1./dx)

      Ay_builder.addEntry(indx,indx,3./2./dy)
      Ay_builder.addEntry(indx,indx_jm1,-2./dy)
      if (j > 1):
        Ay_builder.addEntry(indx,indx_jm2,0.5/dy)

  # now compute first x and y 
  Ax_builder.addEntry(0,0,1./dx)
  Ay_builder.addEntry(0,0,1./dy)

  Ax = Ax_builder.assemble()
  Ay = Ay_builder.assemble()

  return Ax,Ay

def BuildDiffusionMatrix(Nx,Ny,dx,dy):
  '''
  This function builds in the system matrix for 
  diffusion via a second order central difference
  '''
  A_builder = MySparseMatrixBuilder(Nx*Ny,Nx*Ny)
  #first compute du/dx and du/dy on interior 
  for j in range(0,Ny):
    for i in range(0,Nx):
      indx = i + j*Nx
      indx_im1 = indx - 1
      indx_ip1 = indx + 1
      indx_jm1 = indx - Nx
      indx_jp1 = indx + Nx

      A_builder.addEntry(indx,indx,-2./dx**2 - 2./dy**2)
      if (i > 0):
        A_builder.addEntry(indx,indx_im1,1./dx**2)
      if (i < Nx-1):
        A_builder.addEntry(indx,indx_ip1,1./dx**2)
      if (j > 0):
        A_builder.addEntry(indx,indx_jm1,1./dx**2)
      if (j < Ny-1):
        A_builder.addEntry(indx,indx_jp1,1./dx**2)
  A = A_builder.assemble()
  return A

def BuildObjectiveFunctionVector(Nx,Ny,dx,dy):
    '''
    This function builds a vector which can be used to compute
    the integral of du/dx along the right boundary (x=Lx). 
    The inner product of this vector with the state will compute
    the objective. The vector itself is also the right-hand side
    of the adjoint equation
    '''
    C = np.zeros(Nx*Ny)

    for j in range(0,Ny):
      indx = Nx-1 + j*Nx
      indx_im1 = indx - 1
      indx_jm1 = indx - Nx

      #C[indx] += 3./2./dx * dy this term is zero since diriclet BC sets u=0
      C[indx] += -2./dx * dy
      C[indx_im1] += 0.5/dx * dy
     
    return C


class AdvectionDiffusionSystem:
  '''
  This class contains information for an advection diffusion system
  '''
  def __init__(self,Nx,Ny):
    self.Lx = 1.
    self.Ly = 1.
    self.N  = Nx*Ny
    self.Nx =  Nx
    self.Ny =  Ny
    self.dx = float(self.Lx) / float(self.Nx + 1)
    self.dy = float(self.Ly) / float(self.Ny + 1)
    self.g = np.ones(self.N)
    self.x = np.linspace(self.dx,self.Lx-self.dx,self.Nx)
    self.y = np.linspace(self.dy,self.Ly-self.dy,self.Ny)

    self.A_diffusion = BuildDiffusionMatrix(self.Nx,self.Ny,self.dx,self.dy) 
    self.A_advection_x,self.A_advection_y = BuildAdvectionMatrices(self.Nx,self.Ny,self.dx,self.dy) 
    self.I = scipy.sparse.csr_matrix( np.eye(Nx*Ny) ) 

    # create objective function matrix
    self.C = BuildObjectiveFunctionVector(self.Nx,self.Ny,self.dx,self.dy)

class AdjointAdvectionDiffusionSystem:
  '''
  This class contains information for the adjoint of an advection diffusion system
  '''
  def __init__(self,system):
    self.Lx = system.Lx
    self.Ly = system.Ly
    self.N  = system.N
    self.Nx = system.Nx
    self.Ny = system.Ny
    self.dx = system.dx
    self.dy = system.dy
    self.g  = system.g
    self.x  = system.x
    self.y  = system.y
 
    self.A_diffusion = system.A_diffusion
    self.A_advection_x = system.A_advection_x
    self.A_advection_y = system.A_advection_y


    self.A_diffusion_adj = system.A_diffusion.transpose()
    self.A_advection_x_adj = system.A_advection_x.transpose()
    self.A_advection_y_adj = system.A_advection_y.transpose()
    self.I = system.I

    self.C = system.C


def solveFom(system,b,nu,sigma):
  '''
  Given an AdvectionDiffusionSystem class, this function 
  solves the steady linear advection diffusion equation
  '''
  LHS = system.A_diffusion*nu - b[0]*system.A_advection_x - b[1]*system.A_advection_y - sigma*system.I
  RHS = -system.g
  return scipy.sparse.linalg.spsolve(LHS,RHS)

def residual(system,u,b,nu,sigma):
  '''
  Given an AdvectionDiffusionSystem class and a solution,
  this function computes the residual
  '''
  A = system.A_diffusion*nu - b[0]*system.A_advection_x - b[1]*system.A_advection_y - sigma*system.I
  return A.dot(u) + system.g

def solveFomTangent(system,u,b,nu,sigma):
  '''
  Given an AdvectionDiffusionSystem class, this function 
  solves the steady linearized advection diffusion equation at state u
  '''
  LHS = system.A_diffusion*nu - b[0]*system.A_advection_x - b[1]*system.A_advection_y - sigma*system.I
  RHS = -system.A_diffusion.dot(u) # dF/dnu
  return scipy.sparse.linalg.spsolve(LHS,RHS) 

def solveFomAdjoint(system,b,nu,sigma):
  '''
  Given an AdjointAdvectionDiffusionSystem class, this function 
  solves the steady adjoint advection diffusion equation for a steady primal solution u 
  '''
  LHS = system.A_diffusion_adj*nu - b[0]*system.A_advection_x_adj - b[1]*system.A_advection_y_adj - sigma*system.I  
  RHS = -system.C
  return scipy.sparse.linalg.spsolve(LHS,RHS)

def computeGradientTangent(system,u,v):
  '''
  Given an AdvectionDiffusionSystem class, steady primal, and steady tangent solutions
  this function computes the gradient of the objective with respect to nu 
  '''
  return np.dot(system.C,v)

def computeGradientAdjoint(system,u,phi):
  '''
  Given an AdjointAdvectionDiffusionSystem class, steady primal, and steady adjoint solutions
  this function computes the gradient of the objective with respect to nu 
  '''
  dF = system.A_diffusion.dot(u) # dF/dnu
  return np.dot(dF,phi) 

if __name__ == '__main__':
  # Main driver script
  Nx = 31
  Ny = Nx
  system = AdvectionDiffusionSystem(Nx,Ny)
  
  b = np.zeros(2)
  bmag = 0.5
  angle = np.pi/3.
  b[0] = bmag*np.cos(angle)
  b[1] = bmag*np.sin(angle)
  nu    = 1e-3
  sigma = 1.0
  
  # solve CDR equation
  u = solveFom(system,b,nu,sigma) 
  
  #TODO pad with zeros prior to postprocessing
  u_full = np.zeros((Nx+2,Ny+2))
  u_full[1:-1,1:-1] = u.reshape(Nx,Ny)
  
  Lx = system.Lx
  Ly = system.Ly
  
  x = np.linspace(0.0,Lx,Nx+2)
  y = np.linspace(0.0,Ly,Ny+2)
  
  print(np.dot(system.C,u[:,-1]))
  
  # compute tangent for sensitivity analysis
  v = solveFomTangent(system,u[:,-1],b,nu,sigma)
  
  # computeadjoint
  adjointSystem = AdjointAdvectionDiffusionSystem(system)
  
  phi = solveFomAdjoint(adjointSystem,b,nu,sigma)
  
  # compute gradient
  print(computeGradientTangent(system,u[:,-1],v))
  print(computeGradientAdjoint(adjointSystem,u[:,-1],phi))
 
  plt.figure()
  
  plt.contourf(x,y,u_full,81)
  plt.colorbar()

  plt.axis('equal')
  
  
  #plt.figure()
  #
  #plt.contourf(x[1:-1],y[1:-1],v.reshape(Nx,Ny),81)
  #
  #plt.axis('equal')
  #plt.colorbar()
  
  
  plt.figure()
  
  plt.contourf(x[1:-1],y[1:-1],phi.reshape(Nx,Ny),81)
  
  plt.axis('equal')
  plt.colorbar()
  plt.show()
