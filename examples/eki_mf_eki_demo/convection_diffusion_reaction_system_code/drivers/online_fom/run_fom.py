import numpy as np
import sys
import os
import scipy.sparse
import scipy.sparse.linalg
sys.path.append('../../')
import cdr as cdr

import copy
axis_font = {'size':'20'}
import matplotlib.pyplot as plt 
axis_font = {'size':16,'family':'serif'}
import yaml

if __name__ == '__main__':
  # Main driver script
  with open('input_fom.yaml') as f:
      fom_yaml = yaml.safe_load(f)

  Nx = fom_yaml['Nx'] 
  Ny = Nx
  system = cdr.AdvectionDiffusionSystem(Nx,Ny)
  
  output_path = fom_yaml['output_path']
  out_dir = ''
  for directory in output_path.split('/'):
    out_dir += directory
    if os.path.isdir(out_dir):
      pass
    else:
      try:
        os.mkdir(out_dir)
      except:
        print('Cant make working directory')
    out_dir += '/'

  output_file = fom_yaml['output_file'] 
    
  c = np.zeros(2)
  b = fom_yaml['b']
  angle = fom_yaml['angle'] 
  c[0] = b*np.cos(angle)
  c[1] = b*np.sin(angle)
  nu    = fom_yaml['nu']
  sigma = fom_yaml['sigma']
  
  # solve CDR equation
  u = cdr.solveFom(system,c,nu,sigma) 
  np.savez(out_dir + output_file,u=u,angle=angle,nu=nu,sigma=sigma) 
  np.savetxt(out_dir + 'qoi.dat',np.array([np.linalg.norm(u)])) #norm uhat = norm u when basis are orthonormal 
