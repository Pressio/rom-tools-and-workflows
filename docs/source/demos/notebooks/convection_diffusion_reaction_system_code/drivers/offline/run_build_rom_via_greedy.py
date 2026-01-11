import numpy as np
import csv
import scipy.stats
from scipy.stats import qmc
import sys
sys.path.append('../../')
import cdr as cdr
import cdr_rom as cdr_rom
import copy
import time
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import os
axis_font = {'size':24,'family':'serif'}
axis_font2 = {'size':16,'family':'serif'}
import yaml

if __name__ == '__main__':
  # Main driver script

  with open('offline_yaml.yaml') as f:
    offline_yaml = yaml.safe_load(f)

  Nx = offline_yaml['Nx'] 
  Ny = Nx

  b_min = offline_yaml['b_min']
  b_max = offline_yaml['b_max']
  theta_min = offline_yaml['theta_min']
  theta_max = offline_yaml['theta_max']
  nu_min = offline_yaml['nu_min']
  nu_max = offline_yaml['nu_max']
  sigma_min = offline_yaml['sigma_min']
  sigma_max = offline_yaml['sigma_max']

  fom = cdr.AdvectionDiffusionSystem(Nx,Ny)
  

  bBounds = np.array([b_min,b_max])
  angleBounds = np.array([theta_min,theta_max])
  nuBounds = np.array([nu_min,nu_max])
  sigmaBounds = np.array([sigma_min,sigma_max])
  lowerBounds = np.array([bBounds[0],angleBounds[0],nuBounds[0],sigmaBounds[0]])
  upperBounds = np.array([bBounds[1],angleBounds[1],nuBounds[1],sigmaBounds[1]])

  nSamples = offline_yaml['sample_size']
  sampler = qmc.LatinHypercube(d=lowerBounds.size)
  samples = sampler.random(n=nSamples)
  samples = qmc.scale(samples,lowerBounds,upperBounds)
  sampleIndices = np.array(range(0,nSamples))

  ## Solve FOM for first sample
  sampleNo = 0
  bVec = np.zeros(2)
  bmag = samples[sampleNo,0]
  angle = samples[sampleNo,1] 
  bVec[0] = bmag*np.cos(angle)
  bVec[1] = bmag*np.sin(angle)
  nu    = samples[sampleNo,2] 
  sigma = samples[sampleNo,3]
  
  # solve CDR equation
  u = cdr.solveFom(fom,bVec,nu,sigma) 
  tol = offline_yaml['rb_tolerance']*np.linalg.norm(u)

  reducedBasis = copy.deepcopy(u[:,None])
  reducedBasis,_ = np.linalg.qr(reducedBasis,mode='reduced')
  #Begin greedy loop
  converged = False
  coveredSamples = 1
  sampleIndicesLeft = np.delete(sampleIndices,0)
  nGreedySamples = 1
  residualVsGreedySample = np.zeros(0)
  while converged == False:
    rom = cdr_rom.primalGalerkinROM(fom,reducedBasis)
    residualNorms = np.zeros(nSamples)
    for sampleNo in sampleIndicesLeft:
      bVec = np.zeros(2)
      bmag = samples[sampleNo,0]
      angle = samples[sampleNo,1] 
      bVec[0] = bmag*np.cos(angle)
      bVec[1] = bmag*np.sin(angle)
      nu    = samples[sampleNo,2] 
      sigma = samples[sampleNo,3]
      uhat = cdr_rom.solveRom(rom,bVec,nu,sigma)
      uApprox = reducedBasis @ uhat
      residualNorms[sampleNo] = np.linalg.norm( cdr.residual(fom,uApprox,bVec,nu,sigma) )

    maxResidualNorm = np.amax(residualNorms)
    if (maxResidualNorm < tol):
      converged = True
    else:
      print('=========================')
      print('Greedy sample no = ' + str(nGreedySamples))
      print('Max residual = ' + str(maxResidualNorm))
      print('=========================')
      residualVsGreedySample = np.append(residualVsGreedySample,maxResidualNorm)
      sampleNo = np.argmax(residualNorms)
      ## find sample index
      sampleNoIndex = np.argmin( np.abs(sampleNo - sampleIndicesLeft))
      sampleIndicesLeft = np.delete(sampleIndicesLeft,sampleNoIndex)
      bVec = np.zeros(2)
      bmag = samples[sampleNo,0]
      angle = samples[sampleNo,1] 
      bVec[0] = bmag*np.cos(angle)
      bVec[1] = bmag*np.sin(angle)
      nu    = samples[sampleNo,2] 
      sigma = samples[sampleNo,3]
      # solve CDR equation
      u = cdr.solveFom(fom,bVec,nu,sigma)
      reducedBasis = np.append(reducedBasis,u[:,None],axis=1)
      reducedBasis,_ = np.linalg.qr(reducedBasis,mode='reduced')
      nGreedySamples += 1      



  rom = cdr_rom.primalGalerkinROM(fom,reducedBasis)

  output_path = offline_yaml['output_path']
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


  np.save(out_dir + offline_yaml['rom_file_name'],[rom])
  np.savez(out_dir + offline_yaml['basis_file_name'],reducedBasis=reducedBasis)
  if offline_yaml['test_greedy_rom']:
    ### test for new samples   
    nTestSamples = 150
    samples = sampler.random(n=nTestSamples)
    samples = qmc.scale(samples,lowerBounds,upperBounds)
    sampleIndices = np.array(range(0,nTestSamples))
    errorTesting = np.zeros(nTestSamples)
    residualNormsTesting = np.zeros(nTestSamples)
    for sampleNo in sampleIndices:
      bVec = np.zeros(2)
      bmag = samples[sampleNo,0]
      angle = samples[sampleNo,1] 
      bVec[0] = bmag*np.cos(angle)
      bVec[1] = bmag*np.sin(angle)
      nu    = samples[sampleNo,2] 
      sigma = samples[sampleNo,3]
      
      # solve CDR equation
      t0 = time.time()
      u = cdr.solveFom(fom,bVec,nu,sigma) 
      fom_time = time.time() - t0
      t0 = time.time()
      uhat = cdr_rom.solveRom(rom,bVec,nu,sigma)
      uApprox = reducedBasis @ uhat
      rom_time = time.time() - t0
      errorTesting[sampleNo] = np.linalg.norm(u - uApprox)
      residualNormsTesting[sampleNo] = np.linalg.norm( cdr.residual(fom,uApprox,bVec,nu,sigma) )
  
    plt.plot(residualVsGreedySample,color='black')
    plt.xlabel(r'Greedy sample number',**axis_font)
    plt.ylabel(r'Max residual norm',**axis_font)
    plt.grid()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(out_dir + 'residualVsGreedySample.pdf')
  
    plt.figure(2)
    plt.plot(errorTesting,residualNormsTesting,'o',mfc='none')
    plt.xlabel(r'ROM solution error',**axis_font)
    plt.ylabel(r'ROM solution residual',**axis_font)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir + 'errors.pdf')
  
  
    plt.figure(3)
    dx = 1./Nx
    diagVals = np.sqrt( 2.*(np.linspace(dx/2.,1-dx/2.,Nx))**2 )
    uFom = np.reshape(u,(Nx,Ny))
    uRom = np.reshape(uApprox,(Nx,Ny))
    plt.plot(diagVals,np.diag(uFom),color='black',label='FOM')
    plt.plot(diagVals,np.diag(uRom),'o',mfc='none',color='red',label='ROM')
    plt.xlabel(r'$\sqrt{x^2 + y^2} : x=y$',**axis_font)
    plt.ylabel(r'$u(x,y)$',**axis_font)
    plt.legend(loc=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir + 'solution.pdf')
    plt.show() 
