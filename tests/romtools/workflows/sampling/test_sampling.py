from romtools.workflows.sampling.sampling_coupler_base import *
from romtools.workflows.sampling.sampling import *
from romtools.workflows.parameter_spaces import *
import numpy as np
import pytest

class ConcreteSampler(SamplingCouplerBase):
  def __init__(self,template_directory,template_file,sol_directory_base_name = 'run'):
    SamplingCouplerBase.__init__(self,template_directory,template_file,sol_directory_base_name = 'run')
    self.myParameterSpace = UniformParameterSpace(['u','v','w'],np.array([0,1,2]),np.array([1,2,3]))
    self.counter_ = 0
    self.template_file = template_file

  def setParametersInInput(self,filename,parameter_sample):
    file = np.genfromtxt(self.template_file)
    np.savetxt(self.template_file,[self.counter_])
    self.counter_ += 1

  def runModel(self,input_filename,parameter_values):
    return 0

  def getParameterSpace(self):
    return self.myParameterSpace

@pytest.mark.mpi_skip
def test_sampler_builder():
  my_dir = os.path.realpath(os.path.dirname(__file__))
  mySampler = ConcreteSampler(my_dir + '/templates/','test_template.dat')

@pytest.mark.mpi_skip
def test_sampler():
  my_dir = os.path.realpath(os.path.dirname(__file__))
  mySampler = ConcreteSampler(my_dir + '/templates/','test_template.dat')
  runSampling(mySampler,10)
  for i in range(0,10):
    assert(os.path.isdir('work/run' + str(i)))
    data = int(np.genfromtxt('work/run' + str(i) + '/test_template.dat'))
    assert(data == i)

if __name__=="__main__":
  test_sampler_builder()
  test_sampler()
