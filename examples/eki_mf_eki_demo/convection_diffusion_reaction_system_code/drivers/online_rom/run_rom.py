import numpy as np
import sys
import os
import scipy.sparse
import scipy.sparse.linalg
sys.path.append('../../')
import cdr as cdr
import cdr_rom as cdr_rom
import copy
axis_font = {'size':'20'}
import matplotlib.pyplot as plt 
axis_font = {'size':16,'family':'serif'}
import yaml

if __name__ == '__main__':
  # Main driver script
  with open('input_rom.yaml') as f:
      rom_yaml = yaml.safe_load(f)

  Nx = rom_yaml['Nx'] 
  Ny = Nx
  if rom_yaml['rom_construction_type'] ==  'load-from-file':
    rom = np.load(rom_yaml['rom_file_name'] + '.npy',allow_pickle=True)[0]
  elif rom_yaml['rom_construction_type'] == 'compute-from-fom':
    reducedBasis = np.load(rom_yaml['basis_file_name'])['reducedBasis']
    romDimension = rom_yaml['rom_dimension'] 
    reducedBasis = reducedBasis[:,0:romDimension]
    fom = cdr.AdvectionDiffusionSystem(Nx,Ny)
    rom = cdr_rom.primalGalerkinROM(fom,reducedBasis)
  else:
    print('ROM construction type not understood') 

  output_path = rom_yaml['output_path']
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
 
  output_file = rom_yaml['output_file'] 
  c = np.zeros(2)
  b = rom_yaml['b']
  angle = rom_yaml['angle'] 
  c[0] = b*np.cos(angle)
  c[1] = b*np.sin(angle)
  nu    = rom_yaml['nu']
  sigma = rom_yaml['sigma']
  
  # solve CDR equation
  uhat = cdr_rom.solveRom(rom,c,nu,sigma) 
  if rom_yaml['save_full_state']:
    u = rom.basis @ uhat
    np.savez(out_dir + output_file,uhat=uhat,u=u,angle=angle,nu=nu,sigma=sigma) 
  else:
    np.savez(out_dir + output_file,uhat=uhat,angle=angle,nu=nu,sigma=sigma) 
  np.savetxt(out_dir + 'qoi.dat',np.array([np.linalg.norm(uhat)])) #norm uhat = norm u when basis are orthonormal 
