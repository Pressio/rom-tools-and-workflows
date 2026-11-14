import romtools
import numpy as np
import sys
import os
import numpy as np
import pytest
import romtools.workflows
from romtools.rom.qoi_surrogates import GaussianProcessQoiModel
from romtools.workflows.inverse._inverse_utils import *
from romtools.workflows.inverse.ego_optimization_methods import *



class QuadraticModel:
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
        f = self.a_*(1.0 - x)**2 + self.b_*(1.0 - y)**2
        return np.array([f])

class QuadraticParameterSpace(romtools.workflows.ParameterSpace):
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
def test_quadratic(tmp_path):
    # Construct the fom model
    model = QuadraticModel(a=1,b=4)

    my_parameter_space = QuadraticParameterSpace()



    obs = np.array([0.0])
    obs_cov = np.eye(obs.size)*0.0

    parameter_sample_min, obj_min, qoi_min = romtools.workflows.run_ego(model = model,
                 parameter_space = my_parameter_space,
                 observations = obs,
                 number_of_iterations = 40,
                 parameter_mins = my_parameter_space.parameter_mins,
                 parameter_maxes = my_parameter_space.parameter_maxes,
                 absolute_work_dir=tmp_path,
                 number_initial_samples = 20,
                 random_seed=0,
                 use_relative_error=False)
    print(parameter_sample_min,qoi_min,np.linalg.norm(parameter_sample_min-np.array([1.0,1.0])))
    assert( qoi_min < 0.1 )
    assert( np.linalg.norm(parameter_sample_min-np.array([1.0,1.0])) < 0.25 )

@pytest.mark.mpi_skip
def test_quadratic_gp(tmp_path):
    # Construct the fom model
    model = QuadraticModel(a=1,b=4)

    my_parameter_space = QuadraticParameterSpace()
    parameter_samples = my_parameter_space.generate_samples(20, seed=0)
    parameter_names = my_parameter_space.get_names()
    obs = np.array([0.0])

    # run samples
    qois = []
    errors = []
    objs = []
    for sample in range(20):
        run_directory = f'{tmp_path}/run_{sample}'
        qoi, error, _ = prepare_and_run(model, obs, run_directory, parameter_names, parameter_samples[sample])
        obj = objective_function(qoi,obs,relative=False)
        qois.append(qoi)
        errors.append(error)
        objs.append(obj)

    gp_regressor = GaussianProcessQoiModel(parameter_samples,objs,tune_hyperparameters=True)

    # check variance against truth
    test_parameter_samples = my_parameter_space.generate_samples(4, seed=1)

    qoi_means_truth = [0.25223383,4.45371342,10.69529884,2.4877645]
    qoi_stds_truth  = [0.18561455,2.91041433,1.43428366,2.99915945]

    qoi_means = []
    qoi_stds = []
    for sample,qoi_mean_true,qoi_std_true in zip(test_parameter_samples,qoi_means_truth,qoi_stds_truth):
        qoi_mean,qoi_std = gp_regressor.compute_qoi_and_var("",sample)

        assert(np.isclose(qoi_mean,qoi_mean_true))
        assert(np.isclose(qoi_std,qoi_std_true))



@pytest.mark.mpi_skip
def test_quadratic_batch(tmp_path):
    # Construct the fom model
    model = QuadraticModel(a=1,b=4)

    my_parameter_space = QuadraticParameterSpace()

    obs = np.array([0.0])
    obs_cov = np.eye(obs.size)*0.0

    parameter_sample_min, obj_min, qoi_min = romtools.workflows.run_batch_ego(model = model,
                 parameter_space = my_parameter_space,
                 observations = obs,
                 number_of_iterations = 15,
                 batch_size = 4,
                 parameter_mins = my_parameter_space.parameter_mins,
                 parameter_maxes = my_parameter_space.parameter_maxes,
                 absolute_work_dir=tmp_path,
                 number_initial_samples = 20,
                 random_seed=0,
                 evaluation_concurrency=4,
                 use_relative_error=False)
    print(parameter_sample_min,qoi_min,np.linalg.norm(parameter_sample_min-np.array([1.0,1.0])))
    assert( qoi_min < 0.1 )
    assert( np.linalg.norm(parameter_sample_min-np.array([1.0,1.0])) < 0.25 )

@pytest.mark.mpi_skip
@pytest.mark.parametrize("batch_size", [None, 2], ids=["sequential-acquisition", "batch-acquisition"])
def test_ego_accepts_nonblocking_model(tmp_path, batch_size, deferred_model):
    model = deferred_model(QuadraticModel(a=1, b=4))
    space = QuadraticParameterSpace()
    options = dict(
        model=model, parameter_space=space, observations=np.array([0.0]),
        number_of_iterations=2, parameter_mins=space.parameter_mins,
        parameter_maxes=space.parameter_maxes, absolute_work_dir=tmp_path,
        number_initial_samples=6, evaluation_concurrency=2, random_seed=0,
        use_relative_error=False,
    )
    if batch_size is None:
        sample, _, qoi = romtools.workflows.run_ego(**options)
    else:
        sample, _, qoi = romtools.workflows.run_batch_ego(batch_size=batch_size, **options)

    assert model.peak_pending == 2
    assert not model.pending
    assert len(model.completed) == 6 + (batch_size or 1)
    np.testing.assert_allclose(qoi, model.model.compute_qoi("", dict(zip(space.get_names(), sample))))


if __name__=='__main__':
    test_quadratic(os.getcwd() + "/work")
    test_quadratic_gp(os.getcwd() + "/work_gp")
    test_quadratic_batch(os.getcwd() + "/work_batch")