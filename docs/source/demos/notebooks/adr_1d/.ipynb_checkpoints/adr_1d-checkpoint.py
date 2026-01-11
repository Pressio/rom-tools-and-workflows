import numpy as np

'''
Here, we will interface around a very basic model for solving the 1D poisson equation.
c u_x - nu * u_xx = 1
'''
# We will start with defining an advection diffusion problem. This class does NOT meet any interface related to romtools
class advectionDiffusionProblem:
    def __init__(self , nx):
        nx_ = nx
        self.x_ = np.linspace(0,1,nx)
        dx = 1. / (nx - 1)
        self.Ad_ = np.zeros((nx,nx))
        for i in range(1,nx_ -1):
            self.Ad_[i,i] = -2./dx**2
            self.Ad_[i,i-1] = 1./dx**2
            self.Ad_[i,i+1] = 1./dx**2
            
        self.Ac_ = np.zeros((nx,nx))
        for i in range(1,nx_ -1 ):
            self.Ac_[i,i] = 1/dx
            self.Ac_[i,i-1] = -1./dx
    
        self.f_ = np.ones(nx)

    def assemble_system(self,c,nu):
        self.A_ = c*self.Ac_ - nu*self.Ad_
        self.A_[0,0] = 1.
        self.A_[-1,-1] = 1.
        self.f_[0] = 0.
        self.f_[-1] = 0.

    def solve(self,c,nu):
        self.assemble_system(c,nu)
        solution = np.linalg.solve(self.A_,self.f_)
        return solution


if __name__ == "__main__":
    adr_problem = advectionDiffusionProblem(nx=33)
    params = np.genfromtxt('params.dat')
    u = adr_problem.solve(params[0],params[1])
    np.savez('solution',u=u,x=adr_problem.x_) 
