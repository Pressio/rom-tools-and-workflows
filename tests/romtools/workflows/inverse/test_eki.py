import romtools
import numpy as np
import sys
import os
import numpy as np
import pytest
import romtools.workflows




class RosenbrockModel:
    '''
    Protocol for a model that has a return_qoi implementation
    '''
    def __init__(self,a,b) -> None:
        '''
        Initialize coupler
        '''
        self.a_ = a
        self.b_ = b

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        pass

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        pass

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        x = parameter_sample['x']
        y = parameter_sample['y']
        f = (self.a_ - x)**2 + self.b_*(y - x**2)**2
        return np.array([f])

class RosenbrockParameterSpace(romtools.workflows.ParameterSpace):
    def __init__(self):
        self.parameter_mins = np.array([0.0,0.0])
        self.parameter_maxes = np.array([2.0,2.0])
        self.dimension = 2

    def get_names(self):
        '''
        Returns a list of parameter names
        # e.g., ['sigma','beta',...]
        '''
        names = ['x','y']
        return names

    def get_dimensionality(self) -> int:
        '''
        Returns an integer for the size
        of the parameter domain
        '''
        return self.dimension

    def generate_samples(self, number_of_samples: int, seed=None) -> np.array:
        '''
        Generates samples from the parameter space

        Returns np.array of shape
        (number_of_samples, self.get_dimensionality())
        '''
        samples = np.random.uniform(low=self.parameter_mins,high=self.parameter_maxes,size=(number_of_samples,self.dimension))
        return samples

@pytest.mark.mpi_skip
def test_rosenbrock(tmp_path):
    # Construct the fom model
    model = RosenbrockModel(a=1,b=100)

    my_parameter_space = RosenbrockParameterSpace()

    # Run sampling algorithm
    ensemble_size = 5


    obs = np.array([0.0])
    obs_cov = np.eye(obs.size)*0.0

    params,qois = romtools.workflows.run_eki(model = model,
                 parameter_space = my_parameter_space,
                 observations = obs,
                 observations_covariance = obs_cov,
                 parameter_mins = None,
                 parameter_maxes = None,
                 ensemble_size=ensemble_size,
                 absolute_eki_directory=tmp_path,
                 regularization_parameter= 0.0,
                 relaxation_parameter = 1.05,
                 delta_params_tolerance = 1e-50,
                 error_norm_tolerance = 1e-9,
                 evaluation_concurrency = 1,
                 max_iterations = 10000)
    params_mean = np.mean(params,axis=0)
    assert( np.allclose(params_mean,np.array([1.0,1.0]),atol=1e-5))

if __name__=='__main__':
    test_rosenbrock(os.getcwd() + "/work")
