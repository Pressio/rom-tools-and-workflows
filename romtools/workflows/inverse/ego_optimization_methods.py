"""Optimization methods and configs used by inverse EGO workflows."""

from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.rom.qoi_surrogates import GaussianProcessRegressorLite
from romtools.workflows.inverse._inverse_utils import bound_samples
import numpy as np
from scipy.stats.norm import norm
from scipy import minimize

def objective_function(qoi: np.ndarray, observations: np.ndarray):
    return np.sum((qoi.flatten() - observations.flatten())**2) / np.sum((observations.flatten())**2)

def _expected_improvement(gp_regressor: GaussianProcessRegressorLite, 
                          obj_min: float, 
                          parameter_samples: np.ndarray):

    mu,sigma = gp_regressor.predict_mean_and_std(parameter_samples)
    Z = (parameter_samples - mu[:,None]) / sigma

    EI = (obj_min[:,None] - mu) * norm.cdf(Z)+ sigma * norm.pdf(Z)
    mask = sigma < 1e-12
    EI[mask] = 0.0
    return EI

def argmax_expected_improvement(gp_regressor: GaussianProcessRegressorLite, 
                                 obj_min: float,
                                 parameter_space: ParameterSpace,
                                 parameter_mins: np.ndarray = None,
                                 parameter_maxes: np.ndarray = None,
                                 num_restarts: int=25,
                                 random_seed: int = 1):
    
    # determine bounds (if any)
    if parameter_mins is not None:
        parameter_bounds = ()
        for lb,ub in zip(parameter_mins,parameter_maxes):
            parameter_bounds += ((lb,ub),)

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

