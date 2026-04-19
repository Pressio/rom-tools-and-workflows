"""Optimization methods and configs used by inverse EGO workflows."""

from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.rom.qoi_surrogates import GaussianProcessQoiModel
from romtools.workflows.inverse._inverse_utils import bound_samples
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import copy

def objective_function(qoi: np.ndarray, observations: np.ndarray, relative=True):
    scale = 1.0
    if relative:
        scale = 1 / np.sum((observations.flatten())**2) 
    return np.sum((qoi.flatten() - observations.flatten())**2) * scale

def _expected_improvement(gp_regressor: GaussianProcessQoiModel, 
                          obj_min: float, 
                          parameter_sample: np.ndarray):

    mu,sigma = gp_regressor.compute_qoi_and_var("",parameter_sample)

    Z = (mu - obj_min) / sigma

    norm = stats.norm
    EI = (obj_min - mu) * norm.cdf(Z)+ sigma * norm.pdf(Z)
    mask = sigma < 1e-12
    EI[mask] = 0.0
    return EI

def argmax_expected_improvement(gp_regressor: GaussianProcessQoiModel, 
                                 obj_min: float,
                                 parameter_space: ParameterSpace,
                                 parameter_mins: np.ndarray = None,
                                 parameter_maxes: np.ndarray = None,
                                 num_restarts: int=25,
                                 random_seed: int = None):
    
    # determine bounds (if any)
    if parameter_mins is not None:
        parameter_bounds = ()
        for lb,ub in zip(parameter_mins,parameter_maxes):
            parameter_bounds += ((lb,ub),)
    else:
        parameter_bounds = None

    def objective_fcn(parameter_sample):
        parameter_sample = np.asarray(parameter_sample, float)
        return -_expected_improvement(gp_regressor, obj_min, parameter_sample[None, :])

    best_parameter_sample = None
    best_obj = np.inf

    # initial conditions
    init_parameter_samples = parameter_space.generate_samples(num_restarts, seed=random_seed)
    init_parameter_samples = bound_samples(init_parameter_samples,parameter_mins,parameter_maxes)

    for init_parameter_sample in init_parameter_samples:
        result = minimize(objective_fcn, x0=init_parameter_sample, method="L-BFGS-B", bounds=parameter_bounds, options={"maxiter": 200})
        if result.fun < best_obj:
            best_obj = result.fun
            best_parameter_sample = result.x

    return best_parameter_sample

def q_point_expected_improvement_constant_liar(gp_regressor: GaussianProcessQoiModel, 
                                                obj_min: float,
                                                parameter_samples: np.ndarray,
                                                objective_function_samples: np.ndarray,
                                                batch_size: int,
                                                parameter_space: ParameterSpace,
                                                parameter_mins: np.ndarray = None,
                                                parameter_maxes: np.ndarray = None,
                                                num_restarts: int=25,
                                                random_seed: int = None):
    
    number_of_samples = parameter_samples.shape[0]
    my_parameter_samples = parameter_samples.copy()
    my_objective_function_samples = objective_function_samples.copy()
    my_gp_regressor = copy.deepcopy(gp_regressor)
    my_kernel = copy.deepcopy(my_gp_regressor.kernel)
    
    # generate objective function evaluation "Lie". 
    # Since we are minimizing the objective function, using the maximum objective 
    # function is the pesimistic case in which exploration is weighed over exploitation.
    # Using the minimum objective function is the optimistic case which emphasizes exploitation.
    # Using the mean objective function is a compromise between the two former choices
    L = np.max(my_objective_function_samples)

    for i in range(batch_size):
        best_parameter_sample = argmax_expected_improvement(my_gp_regressor,
                                                            obj_min,
                                                            parameter_space,
                                                            parameter_mins,
                                                            parameter_maxes,
                                                            num_restarts,
                                                            random_seed)
        my_parameter_samples = np.vstack([my_parameter_samples,best_parameter_sample])
        my_objective_function_samples = np.concatenate([my_objective_function_samples,L])
        # Refit GP here!
        my_gp_regressor = GaussianProcessQoiModel(my_parameter_samples,
                                                  my_objective_function_samples,
                                                  kernel=my_kernel)

    # only return new samples
    return my_parameter_samples[number_of_samples:]
