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
        np.random.seed(seed)
        samples = np.random.uniform(low=self.parameter_mins,high=self.parameter_maxes,size=(number_of_samples,self.dimension))
        return samples

@pytest.mark.mpi_skip
def test_rosenbrock(tmp_path):
    # Construct the fom model
    model = RosenbrockModel(a=1,b=100)

    my_parameter_space = RosenbrockParameterSpace()



    obs = np.array([0.0])
    obs_cov = np.eye(obs.size)*0.0

    parameter_sample_min, obj_min, qoi_min = romtools.workflows.run_ego(model = model,
                 parameter_space = my_parameter_space,
                 observations = obs,
                 number_of_iterations = 60,
                 parameter_mins = my_parameter_space.parameter_mins,
                 parameter_maxes = my_parameter_space.parameter_maxes,
                 absolute_ego_directory=tmp_path,
                 number_initial_samples = 20,
                 evaluation_concurrency = 1,
                 random_seed=74,
                 use_relative_error=False)

    assert( qoi_min < 1 )
    assert( np.linalg.norm(parameter_sample_min-np.array([1.0,1.0])) < 0.1 )

if __name__=='__main__':
    test_rosenbrock(os.getcwd() + "/work")