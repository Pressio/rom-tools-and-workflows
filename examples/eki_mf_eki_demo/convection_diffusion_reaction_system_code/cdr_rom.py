from cdr import *


class primalGalerkinROM:
  '''
  This class contains information for a POD-Galerkin ROM of the advection diffusion system
  '''
  def __init__(self,system,basis):
    self.Lx = system.Lx
    self.Ly = system.Ly
    self.N  = system.N
    self.Nx = system.Nx
    self.Ny = system.Ny
    self.dx = system.dx
    self.dy = system.dy
    self.x  = system.x
    self.y  = system.y
    
    self.basis = basis
    Nfom, Nrom = basis.shape
    assert(Nfom == self.N)
    self.Nrom = Nrom
   
    # project operators
    self.g             = np.matmul(self.basis.T, system.g)
    self.A_diffusion   = np.matmul(self.basis.T,system.A_diffusion.dot(self.basis))
    self.A_advection_x = np.matmul(self.basis.T,system.A_advection_x.dot(self.basis))
    self.A_advection_y = np.matmul(self.basis.T,system.A_advection_y.dot(self.basis))
    self.C             = np.matmul(self.basis.T, system.C)
    self.I             = np.eye(Nrom) 


class adjointGalerkinROM:
  '''
  This class contains information for a POD-Galerkin ROM of the adjoint advection diffusion system
  '''
  def __init__(self,system,basis):
    self.Lx = system.Lx
    self.Ly = system.Ly
    self.N  = system.N
    self.Nx = system.Nx
    self.Ny = system.Ny
    self.dx = system.dx
    self.dy = system.dy
    self.x  = system.x
    self.y  = system.y
    
    self.basis = basis
    Nfom, Nrom = basis.shape
    assert(Nfom == self.N)
    self.Nrom = Nrom
    
    # project operators
    self.g                 = np.matmul(self.basis.T, system.g)
    self.A_diffusion_adj   = np.matmul(self.basis.T,system.A_diffusion_adj.dot(self.basis))
    self.A_advection_x_adj = np.matmul(self.basis.T,system.A_advection_x_adj.dot(self.basis))
    self.A_advection_y_adj = np.matmul(self.basis.T,system.A_advection_y_adj.dot(self.basis))
    self.C                 = np.matmul(self.basis.T, system.C)
    self.I                 = np.eye(Nrom) 


def solveRom(system,b,nu,sigma):
  '''
  Given an primalGalerkinROM class, this function 
  solves the Galerkin project of a steady advection diffusion equation
  '''

  LHS = system.A_diffusion*nu - b[0]*system.A_advection_x - b[1]*system.A_advection_y - sigma*system.I
  RHS = -system.g
  return np.linalg.solve(LHS,RHS) 

def solveRomAdjoint(system,b,nu,sigma):
  '''
  Given an primalGalerkinROM class, this function 
  solves the Galerkin projection of thr adjoint steady advection diffusion equation
  '''

  LHS = system.A_diffusion_adj*nu - b[0]*system.A_advection_x_adj - b[1]*system.A_advection_y_adj - sigma*system.I  
  RHS = -system.C
  return np.linalg.solve(LHS,RHS)

