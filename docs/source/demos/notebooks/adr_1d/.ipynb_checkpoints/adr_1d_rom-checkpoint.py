import numpy as np
import argparse

class adrRom:
    def __init__(self,offline_directory):
        # We will assume that precomputed matrices for the diffusion and convection operators,
        # along with the ROM basis, exist in offline_directory
        # The rom_builder will populate this
        rom_data = np.load(offline_directory + '/rom_data.npz')
        self.Ad_r_ = rom_data['Adr']
        self.Ac_r_ = rom_data['Acr']
        self.f_r_ = rom_data['fr']
        self.basis_ = rom_data['basis']
        nx = self.basis_.shape[0]
        self.x_ = np.linspace(0,1,nx)

    def assemble_system(self,c,nu):
        self.A_r_ = c*self.Ac_r_ - nu*self.Ad_r_

    def solve(self,c,nu):
        self.assemble_system(c,nu)
        solution = np.linalg.solve(self.A_r_,self.f_r_)
        u = self.basis_ @ solution
        return u

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-offline_data_dir')
    args = parser.parse_args()
    adr_problem = adrRom(args.offline_data_dir)
    params = np.genfromtxt('params.dat')
    u = adr_problem.solve(params[0],params[1])
    np.savez('solution',u=u,x=adr_problem.x_) 
