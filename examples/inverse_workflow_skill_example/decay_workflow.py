import numpy as np

import romtools.workflows

from examples.inverse_workflow_skill_example.DecayModel import DecayModel
from examples.inverse_workflow_skill_example.DecayParameterSpace import DecayParameterSpace

if __name__ == '__main__':

    output_dir_name = "decay_inverse"

    model = DecayModel()
    params = DecayParameterSpace()

    observations = 2.0 * np.exp(-0.7 * model.times)
    observations_covariance = 0.02**2 * np.eye(observations.size)

    parameter_samples, qois = romtools.workflows.run_eki(
        model=model,
        parameter_space=params,
        observations=observations,
        observations_covariance=observations_covariance,
        parameter_mins=params.lower_bounds,
        parameter_maxes=params.upper_bounds,
        absolute_eki_directory=output_dir_name,
        ensemble_size=16,
        initial_step_size=0.5,
        max_iterations=20,
        max_step_size_decrease_trys=3,
        error_norm_tolerance=1e-3,
        random_seed=1,
        evaluation_concurrency=1,
    )

    print("Reference parameters [amplitude, rate]:", [2.0, 0.7])
    print("Ensemble mean parameters:", parameter_samples.mean(axis=0))
