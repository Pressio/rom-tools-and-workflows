from romtools.workflows.greedy import *
from romtools.workflows.parameter_spaces import *
import numpy as np
import pytest

class ConcreteGreedyCoupler(GreedyCouplerBase):
  def __init__(self,template_directory, \
               template_fom_file, \
               template_rom_file, \
               workDir= None):

    GreedyCouplerBase.__init__(self,template_directory,template_fom_file,template_rom_file, work_directory = workDir)

    self.myParameterSpace = UniformParameterSpace(['u','v','w'],np.array([0,1,2]),np.array([1,2,3]))
    self.counter_ = 0
    self.template_fom_file = template_fom_file

    my_error_estimates = np.array([1.,2.,3.,1.5,4.]) # First iteration, should identify 5th entry as the sample to run
    my_error_estimates_iteration_2 = np.array([0.9,0.4,0.6])
    my_error_estimates_iteration_3 = np.array([0.09,0.1,0.06])
    my_error_estimates_iteration_4 = np.array([1e-7,1e-6,1e-5])
    my_error_estimates = np.append(my_error_estimates,my_error_estimates_iteration_2)
    my_error_estimates = np.append(my_error_estimates,my_error_estimates_iteration_3)
    self.my_error_estimates_ = np.append(my_error_estimates,my_error_estimates_iteration_4)
    self.error_estimate_counter_ = 0

    self.my_errors_ = np.array([1.,1.,0.4,0.09,0.01,1e-6])
    self.my_errors_counter_ = 0

  def setParametersInRomInput(self,filename,parameter_sample):
    pass

  def setParametersInFomInput(self,filename,parameter_sample):
    pass

  def runModel(self):
    pass

  def computeError(self,arg1,arg2):
    error = self.my_errors_[self.my_errors_counter_]
    self.my_errors_counter_ += 1
    return error

  def computeErrorIndicator(self):
    error_estimate = self.my_error_estimates_[self.error_estimate_counter_]
    self.error_estimate_counter_ += 1
    return error_estimate

  def computeQoi(self):
    return 0

  def runRom(self,filename,parameter_values):
    pass

  def runFom(self,filename,parameter_values):
    np.savetxt('fom_succesful.dat',parameter_values)

  def createTrialSpace(self,training_dirs):
    pass

  def getParameterSpace(self):
    return self.myParameterSpace

@pytest.mark.mpi_skip
def test_greedy_coupler_builder():
  my_dir = os.path.realpath(os.path.dirname(__file__))
  myGreedyCoupler = ConcreteGreedyCoupler(my_dir + '/templates/','test_template.dat','test_template.dat')

@pytest.mark.mpi_skip
def test_greedy(tmp_path):
  #see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info about tmp_path
  wdir = str(tmp_path) # does not like posixpaths
  print('\n', wdir)

  my_dir = os.path.realpath(os.path.dirname(__file__))
  myGreedyCoupler = ConcreteGreedyCoupler(my_dir + '/templates/','test_template.dat','test_template.dat', workDir=wdir)
  runGreedy(myGreedyCoupler,1e-5,5)
  ## First greedy pass
  base_path = myGreedyCoupler.getBaseDirectory() + '/work/' + myGreedyCoupler.getFomDirectoryBaseName()
  foms_samples_run = [0,1,4,2,5]
  foms_samples_not_run = [3,6,7]

  for sample in foms_samples_run:
    assert os.path.isfile(base_path + '_' + str(sample) + '/fom_succesful.dat'), sample

  for sample in foms_samples_not_run:
    assert os.path.isfile(base_path + '_' + str(sample) + '/fom_succesful.dat') == False, sample

  greedy_output = np.load('greedy_stats.npz')
  assert np.allclose(greedy_output['max_error_indicators'],np.array([4.,0.9,0.1]))
  assert np.allclose(greedy_output['training_samples'],np.array([0,1,4,2,5]))
  assert np.allclose(greedy_output['qoi_errors'],np.array([0.4,0.09,0.01]))

if __name__=="__main__":
  test_greedy_coupler_builder()
  test_greedy()
