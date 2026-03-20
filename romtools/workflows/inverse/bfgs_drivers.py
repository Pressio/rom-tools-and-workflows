import numpy as np
import os
import time
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.inverse._inverse_utils import *

def _normalize_finite_difference_method(finite_difference_method: str):
    method = finite_difference_method.strip().lower()
    aliases = {
        "first_order": "first_order",
        "forward_first_order": "first_order",
        "forward": "first_order",
        "second_order": "second_order",
        "central_second_order": "second_order",
        "central": "second_order",
        "fourth_order": "fourth_order",
        "central_fourth_order": "fourth_order",
    }
    if method not in aliases:
        raise ValueError(
            f"Unsupported finite_difference_method '{finite_difference_method}'. "
            "Supported methods are 'first_order', 'second_order', and 'fourth_order'."
        )
    return aliases[method]


def _project_parameter_values(parameter_values: np.ndarray,
                              parameter_mins: np.ndarray = None,
                              parameter_maxes: np.ndarray = None):
    projected_values = parameter_values.copy()
    if parameter_maxes is not None:
        projected_values = np.fmin(projected_values, parameter_maxes)
    if parameter_mins is not None:
        projected_values = np.fmax(projected_values, parameter_mins)
    return projected_values


def _compute_fractional_fd_steps(parameter_values: np.ndarray,
                                 parameter_step_fraction,
                                 minimum_parameter_step: float,
                                 parameter_mins: np.ndarray = None,
                                 parameter_maxes: np.ndarray = None):
    dimensionality = parameter_values.size
    if np.isscalar(parameter_step_fraction):
        fractions = np.full(dimensionality, float(parameter_step_fraction), dtype=float)
    else:
        fractions = np.asarray(parameter_step_fraction, dtype=float)
        if fractions.shape != (dimensionality,):
            raise ValueError(
                f"parameter_step_fraction shape {fractions.shape} does not match "
                f"the parameter dimensionality ({dimensionality},)."
            )
    if np.any(fractions <= 0.0):
        raise ValueError("parameter_step_fraction must be strictly positive.")

    scales = np.abs(parameter_values)
    if parameter_mins is not None and parameter_maxes is not None:
        bounded_scales = np.maximum(np.abs(parameter_maxes - parameter_mins), 1.0)
        scales = np.where(scales > 0.0, scales, bounded_scales)
    else:
        scales = np.where(scales > 0.0, scales, 1.0)
    step_sizes = fractions * scales
    return np.maximum(step_sizes, minimum_parameter_step)


def _compute_weighted_error_objective(error: np.ndarray,
                                      observations_covariance: np.ndarray,
                                      covariance_regularization: float):
    if observations_covariance is None:
        return 0.5 * float(error @ error)
    regularized_covariance = observations_covariance + covariance_regularization * np.eye(error.size)
    try:
        weighted_error = np.linalg.solve(regularized_covariance, error)
    except np.linalg.LinAlgError:
        weighted_error = np.linalg.pinv(regularized_covariance) @ error
    return 0.5 * float(error @ weighted_error)


def _perturb_parameter(parameter_values: np.ndarray,
                       parameter_index: int,
                       perturbation: float,
                       parameter_mins: np.ndarray = None,
                       parameter_maxes: np.ndarray = None):
    perturbed_values = parameter_values.copy()
    perturbed_values[parameter_index] += perturbation
    if parameter_maxes is not None:
        perturbed_values[parameter_index] = min(perturbed_values[parameter_index], parameter_maxes[parameter_index])
    if parameter_mins is not None:
        perturbed_values[parameter_index] = max(perturbed_values[parameter_index], parameter_mins[parameter_index])
    actual_perturbation = perturbed_values[parameter_index] - parameter_values[parameter_index]
    return perturbed_values, actual_perturbation


def _compute_finite_difference_gradient(parameter_values: np.ndarray,
                                        objective_at_parameters: float,
                                        finite_difference_method: str,
                                        step_sizes: np.ndarray,
                                        parameter_mins: np.ndarray,
                                        parameter_maxes: np.ndarray,
                                        evaluate_objective):
    gradient = np.zeros_like(parameter_values, dtype=float)
    dimensionality = parameter_values.size

    for parameter_index in range(dimensionality):
        step_size = step_sizes[parameter_index]

        plus_h_parameters, plus_h_delta = _perturb_parameter(
            parameter_values,
            parameter_index,
            step_size,
            parameter_mins,
            parameter_maxes,
        )
        minus_h_parameters, minus_h_delta = _perturb_parameter(
            parameter_values,
            parameter_index,
            -step_size,
            parameter_mins,
            parameter_maxes,
        )
        plus_h_objective = None
        minus_h_objective = None
        if np.abs(plus_h_delta) > 0.0:
            plus_h_objective = evaluate_objective(
                plus_h_parameters,
                f'fd_{parameter_index}_plus_h',
            )
        if np.abs(minus_h_delta) > 0.0:
            minus_h_objective = evaluate_objective(
                minus_h_parameters,
                f'fd_{parameter_index}_minus_h',
            )

        if finite_difference_method == 'first_order':
            if plus_h_objective is not None:
                gradient[parameter_index] = (plus_h_objective - objective_at_parameters) / plus_h_delta
            elif minus_h_objective is not None:
                gradient[parameter_index] = (minus_h_objective - objective_at_parameters) / minus_h_delta
            else:
                gradient[parameter_index] = 0.0
            continue

        if finite_difference_method == 'second_order':
            if plus_h_objective is not None and minus_h_objective is not None:
                gradient[parameter_index] = (plus_h_objective - minus_h_objective) / (plus_h_delta - minus_h_delta)
            elif plus_h_objective is not None:
                gradient[parameter_index] = (plus_h_objective - objective_at_parameters) / plus_h_delta
            elif minus_h_objective is not None:
                gradient[parameter_index] = (minus_h_objective - objective_at_parameters) / minus_h_delta
            else:
                gradient[parameter_index] = 0.0
            continue

        plus_2h_parameters, plus_2h_delta = _perturb_parameter(
            parameter_values,
            parameter_index,
            2.0 * step_size,
            parameter_mins,
            parameter_maxes,
        )
        minus_2h_parameters, minus_2h_delta = _perturb_parameter(
            parameter_values,
            parameter_index,
            -2.0 * step_size,
            parameter_mins,
            parameter_maxes,
        )
        can_use_fourth_order = (
            plus_h_objective is not None
            and minus_h_objective is not None
            and np.abs(plus_2h_delta) > 0.0
            and np.abs(minus_2h_delta) > 0.0
            and np.isclose(plus_h_delta, step_size)
            and np.isclose(minus_h_delta, -step_size)
            and np.isclose(plus_2h_delta, 2.0 * step_size)
            and np.isclose(minus_2h_delta, -2.0 * step_size)
        )
        if can_use_fourth_order:
            plus_2h_objective = evaluate_objective(
                plus_2h_parameters,
                f'fd_{parameter_index}_plus_2h',
            )
            minus_2h_objective = evaluate_objective(
                minus_2h_parameters,
                f'fd_{parameter_index}_minus_2h',
            )
            gradient[parameter_index] = (
                -plus_2h_objective
                + 8.0 * plus_h_objective
                - 8.0 * minus_h_objective
                + minus_2h_objective
            ) / (12.0 * step_size)
        elif plus_h_objective is not None and minus_h_objective is not None:
            gradient[parameter_index] = (plus_h_objective - minus_h_objective) / (plus_h_delta - minus_h_delta)
        elif plus_h_objective is not None:
            gradient[parameter_index] = (plus_h_objective - objective_at_parameters) / plus_h_delta
        elif minus_h_objective is not None:
            gradient[parameter_index] = (minus_h_objective - objective_at_parameters) / minus_h_delta
        else:
            gradient[parameter_index] = 0.0

    return gradient


def run_bfgs(model: QoiModel,
             parameter_space: ParameterSpace,
             observations: np.ndarray,
             observations_covariance: np.ndarray,
             parameter_mins: np.ndarray = None,
             parameter_maxes: np.ndarray = None,
             absolute_bfgs_directory: str = os.getcwd() + "/work/",
             initial_parameters: np.ndarray = None,
             finite_difference_method: str = 'second_order',
             parameter_step_fraction=1e-2,
             minimum_parameter_step: float = 1e-8,
             initial_line_search_step_size: float = 1.0,
             line_search_decay_factor: float = 2.0,
             armijo_sufficient_decrease: float = 1e-4,
             covariance_regularization: float = 1e-12,
             bfgs_curvature_tolerance: float = 1e-12,
             gradient_norm_tolerance: float = 1e-6,
             delta_params_tolerance: float = 1e-8,
             objective_tolerance: float = 1e-12,
             max_line_search_iterations: int = 20,
             max_iterations: int = 100,
             random_seed: int = 1,
             restart_file: str = None):
    '''
    Run a finite-difference BFGS optimizer for inverse problems.

    The objective is a weighted observation misfit:
    0.5 * (observations - qoi)^T * C^{-1} * (observations - qoi),
    where C is observations_covariance (with optional regularization).

    Args:
        model: QoiModel to evaluate.
        parameter_space: ParameterSpace used for initialization if
            initial_parameters is not provided.
        observations: Observed QoI vector.
        observations_covariance: Observation covariance matrix used in objective.
        parameter_mins: Optional lower bounds.
        parameter_maxes: Optional upper bounds.
        absolute_bfgs_directory: Absolute path for optimizer run directories.
        initial_parameters: Optional initial parameter vector.
        finite_difference_method: One of 'first_order', 'second_order',
            or 'fourth_order'.
        parameter_step_fraction: Scalar or per-parameter array defining finite
            difference step size as a fraction of each parameter value.
        minimum_parameter_step: Lower bound on finite difference step size.
        initial_line_search_step_size: Initial step size for line search.
        line_search_decay_factor: Backtracking divisor (> 1).
        armijo_sufficient_decrease: Armijo sufficient decrease constant.
        covariance_regularization: Diagonal regularization for covariance inversion.
        bfgs_curvature_tolerance: Minimum s^T y for BFGS Hessian update.
        gradient_norm_tolerance: Convergence tolerance for gradient norm.
        delta_params_tolerance: Convergence tolerance for parameter update norm.
        objective_tolerance: Convergence tolerance for objective value.
        max_line_search_iterations: Maximum backtracking iterations.
        max_iterations: Maximum BFGS iterations.
        random_seed: RNG seed for initialization.
        restart_file: Optional restart file created by this routine.

    Returns:
        Tuple of (parameter_values, qoi) at termination.
    '''
    start_time = time.time()
    assert os.path.isabs(absolute_bfgs_directory), (
        f"bfgs_directory is not an absolute path ({absolute_bfgs_directory})"
    )
    if line_search_decay_factor <= 1.0:
        raise ValueError("line_search_decay_factor must be greater than 1.0")
    if initial_line_search_step_size <= 0.0:
        raise ValueError("initial_line_search_step_size must be positive.")
    if minimum_parameter_step <= 0.0:
        raise ValueError("minimum_parameter_step must be positive.")
    if covariance_regularization < 0.0:
        raise ValueError("covariance_regularization cannot be negative.")
    if max_line_search_iterations <= 0:
        raise ValueError("max_line_search_iterations must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    finite_difference_method = _normalize_finite_difference_method(finite_difference_method)

    dimensionality = parameter_space.get_dimensionality()
    parameter_names = parameter_space.get_names()
    if parameter_mins is not None and np.size(parameter_mins) != dimensionality:
        raise ValueError(
            f"parameter_mins of size {np.size(parameter_mins)} is inconsistent "
            f"with parameter dimensionality ({dimensionality})."
        )
    if parameter_maxes is not None and np.size(parameter_maxes) != dimensionality:
        raise ValueError(
            f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent "
            f"with parameter dimensionality ({dimensionality})."
        )

    np.random.seed(random_seed)
    if restart_file is None:
        if initial_parameters is None:
            parameter_values = parameter_space.generate_samples(1)[0]
        else:
            parameter_values = np.asarray(initial_parameters, dtype=float)
            if parameter_values.shape != (dimensionality,):
                raise ValueError(
                    f"initial_parameters shape {parameter_values.shape} does not match "
                    f"required shape ({dimensionality},)."
                )
        parameter_values = _project_parameter_values(
            parameter_values,
            parameter_mins,
            parameter_maxes,
        )
        inverse_hessian = np.eye(dimensionality)
        iteration = 0
    else:
        restart_data = np.load(restart_file)
        parameter_values = restart_data['parameter_values']
        if parameter_values.shape != (dimensionality,):
            raise ValueError(
                f"Restart parameter_values shape {parameter_values.shape} does not match "
                f"required shape ({dimensionality},)."
            )
        parameter_values = _project_parameter_values(
            parameter_values,
            parameter_mins,
            parameter_maxes,
        )
        inverse_hessian = (
            restart_data['inverse_hessian']
            if 'inverse_hessian' in restart_data
            else np.eye(dimensionality)
        )
        iteration = int(restart_data['iteration']) if 'iteration' in restart_data else 0

    evaluation_cache = {}
    evaluation_counter = 0

    def _evaluate_at(candidate_parameters: np.ndarray, iteration_index: int, label: str):
        nonlocal evaluation_counter
        candidate_parameters = _project_parameter_values(
            np.asarray(candidate_parameters, dtype=float),
            parameter_mins,
            parameter_maxes,
        )
        cache_key = tuple(np.round(candidate_parameters, 15))
        if cache_key in evaluation_cache:
            return evaluation_cache[cache_key]
        run_directory = (
            f'{absolute_bfgs_directory}/iteration_{iteration_index}/'
            f'eval_{evaluation_counter}_{label}'
        )
        qoi, error, _ = prepare_and_run(
            model,
            observations,
            run_directory,
            parameter_names,
            candidate_parameters,
        )
        objective = _compute_weighted_error_objective(
            error,
            observations_covariance,
            covariance_regularization,
        )
        result = {
            'parameter_values': candidate_parameters.copy(),
            'qoi': qoi,
            'error': error,
            'objective': objective,
        }
        evaluation_cache[cache_key] = result
        evaluation_counter += 1
        return result

    current_state = _evaluate_at(parameter_values, iteration, 'base')
    parameter_values = current_state['parameter_values']
    qoi = current_state['qoi']
    error = current_state['error']
    objective = current_state['objective']

    fd_step_sizes = _compute_fractional_fd_steps(
        parameter_values,
        parameter_step_fraction,
        minimum_parameter_step,
        parameter_mins,
        parameter_maxes,
    )
    gradient = _compute_finite_difference_gradient(
        parameter_values,
        objective,
        finite_difference_method,
        fd_step_sizes,
        parameter_mins,
        parameter_maxes,
        lambda params, label: _evaluate_at(params, iteration, label)['objective'],
    )
    gradient_norm = np.linalg.norm(gradient)
    parameter_update_norm = np.inf

    wall_time = time.time() - start_time
    print(
        f'Iteration: {iteration}, Objective: {objective:.6e}, '
        f'Gradient 2-norm: {gradient_norm:.6e}, Delta p: {parameter_update_norm:.6e}, '
        f'Wall time: {wall_time:.6f}'
    )
    np.savez(
        f'{absolute_bfgs_directory}/iteration_{iteration}/restart.npz',
        parameter_values=parameter_values,
        qoi=qoi,
        error=error,
        objective=objective,
        gradient=gradient,
        inverse_hessian=inverse_hessian,
        iteration=iteration,
    )

    line_search_failed = False
    while (
        iteration < max_iterations
        and objective > objective_tolerance
        and gradient_norm > gradient_norm_tolerance
        and parameter_update_norm > delta_params_tolerance
    ):
        search_direction = -inverse_hessian @ gradient
        directional_derivative = float(gradient @ search_direction)
        if directional_derivative >= 0.0:
            inverse_hessian = np.eye(dimensionality)
            search_direction = -gradient.copy()
            directional_derivative = float(gradient @ search_direction)

        trial_step_size = initial_line_search_step_size
        accepted_trial_state = None
        accepted_parameter_update = None
        best_trial_state = None
        best_parameter_update = None
        best_trial_objective = np.inf
        for line_search_iteration in range(max_line_search_iterations):
            trial_parameters = _project_parameter_values(
                parameter_values + trial_step_size * search_direction,
                parameter_mins,
                parameter_maxes,
            )
            trial_update = trial_parameters - parameter_values
            if np.linalg.norm(trial_update) <= 0.0:
                trial_step_size /= line_search_decay_factor
                continue
            trial_state = _evaluate_at(
                trial_parameters,
                iteration + 1,
                f'line_search_{line_search_iteration}',
            )
            if trial_state['objective'] < best_trial_objective:
                best_trial_objective = trial_state['objective']
                best_trial_state = trial_state
                best_parameter_update = trial_update
            if trial_state['objective'] <= (
                objective + armijo_sufficient_decrease * trial_step_size * directional_derivative
            ):
                accepted_trial_state = trial_state
                accepted_parameter_update = trial_update
                break
            trial_step_size /= line_search_decay_factor

        if accepted_trial_state is None:
            if best_trial_state is not None and best_trial_state['objective'] < objective:
                accepted_trial_state = best_trial_state
                accepted_parameter_update = best_parameter_update
                print(
                    f'  Warning, Armijo condition failed after {max_line_search_iterations} '
                    f'tries at iteration {iteration}; accepting best objective-reducing trial step.'
                )
            else:
                inverse_hessian = np.eye(dimensionality)
                search_direction = -gradient.copy()
                directional_derivative = float(gradient @ search_direction)
                trial_step_size = initial_line_search_step_size
                fallback_best_state = None
                fallback_best_update = None
                fallback_best_objective = np.inf
                for line_search_iteration in range(max_line_search_iterations):
                    trial_parameters = _project_parameter_values(
                        parameter_values + trial_step_size * search_direction,
                        parameter_mins,
                        parameter_maxes,
                    )
                    trial_update = trial_parameters - parameter_values
                    if np.linalg.norm(trial_update) <= 0.0:
                        trial_step_size /= line_search_decay_factor
                        continue
                    trial_state = _evaluate_at(
                        trial_parameters,
                        iteration + 1,
                        f'fallback_line_search_{line_search_iteration}',
                    )
                    if trial_state['objective'] < fallback_best_objective:
                        fallback_best_objective = trial_state['objective']
                        fallback_best_state = trial_state
                        fallback_best_update = trial_update
                    if trial_state['objective'] <= (
                        objective + armijo_sufficient_decrease * trial_step_size * directional_derivative
                    ):
                        accepted_trial_state = trial_state
                        accepted_parameter_update = trial_update
                        break
                    trial_step_size /= line_search_decay_factor

                if accepted_trial_state is None and fallback_best_state is not None and fallback_best_state['objective'] < objective:
                    accepted_trial_state = fallback_best_state
                    accepted_parameter_update = fallback_best_update
                    print(
                        f'  Warning, fallback Armijo condition failed after {max_line_search_iterations} '
                        f'tries at iteration {iteration}; accepting best objective-reducing trial step.'
                    )

                if accepted_trial_state is None:
                    print(
                        f'  Warning, failed line search after {max_line_search_iterations} '
                        f'tries at iteration {iteration}.'
                    )
                    line_search_failed = True
                    break

        previous_parameters = parameter_values
        previous_gradient = gradient
        parameter_values = accepted_trial_state['parameter_values']
        qoi = accepted_trial_state['qoi']
        error = accepted_trial_state['error']
        objective = accepted_trial_state['objective']
        parameter_update_norm = np.linalg.norm(accepted_parameter_update)

        fd_step_sizes = _compute_fractional_fd_steps(
            parameter_values,
            parameter_step_fraction,
            minimum_parameter_step,
            parameter_mins,
            parameter_maxes,
        )
        gradient = _compute_finite_difference_gradient(
            parameter_values,
            objective,
            finite_difference_method,
            fd_step_sizes,
            parameter_mins,
            parameter_maxes,
            lambda params, label: _evaluate_at(params, iteration + 1, label)['objective'],
        )
        gradient_norm = np.linalg.norm(gradient)

        s_vector = parameter_values - previous_parameters
        y_vector = gradient - previous_gradient
        ys_dot = float(y_vector @ s_vector)
        if ys_dot > bfgs_curvature_tolerance:
            rho = 1.0 / ys_dot
            identity = np.eye(dimensionality)
            sy_outer = np.outer(s_vector, y_vector)
            ys_outer = np.outer(y_vector, s_vector)
            ss_outer = np.outer(s_vector, s_vector)
            inverse_hessian = (
                (identity - rho * sy_outer)
                @ inverse_hessian
                @ (identity - rho * ys_outer)
                + rho * ss_outer
            )
        else:
            inverse_hessian = np.eye(dimensionality)

        iteration += 1
        wall_time = time.time() - start_time
        print(
            f'Iteration: {iteration}, Objective: {objective:.6e}, '
            f'Gradient 2-norm: {gradient_norm:.6e}, Delta p: {parameter_update_norm:.6e}, '
            f'Wall time: {wall_time:.6f}'
        )
        np.savez(
            f'{absolute_bfgs_directory}/iteration_{iteration}/restart.npz',
            parameter_values=parameter_values,
            qoi=qoi,
            error=error,
            objective=objective,
            gradient=gradient,
            inverse_hessian=inverse_hessian,
            iteration=iteration,
        )

    if line_search_failed:
        print('Terminating due to line-search failure.')
    elif iteration >= max_iterations:
        print('Max iterations reached, terminating.')
    elif objective <= objective_tolerance:
        print('Objective dropped below tolerance.')
    elif gradient_norm <= gradient_norm_tolerance:
        print('Gradient norm dropped below tolerance.')
    elif parameter_update_norm <= delta_params_tolerance:
        print('Parameter update dropped below tolerance.')
    return parameter_values, qoi


def run_steepest_descent(model: QoiModel,
                         parameter_space: ParameterSpace,
                         observations: np.ndarray,
                         observations_covariance: np.ndarray,
                         parameter_mins: np.ndarray = None,
                         parameter_maxes: np.ndarray = None,
                         absolute_steepest_descent_directory: str = os.getcwd() + "/work/",
                         initial_parameters: np.ndarray = None,
                         finite_difference_method: str = 'second_order',
                         parameter_step_fraction=1e-2,
                         minimum_parameter_step: float = 1e-8,
                         initial_line_search_step_size: float = 1.0,
                         line_search_decay_factor: float = 2.0,
                         armijo_sufficient_decrease: float = 1e-4,
                         covariance_regularization: float = 1e-12,
                         gradient_norm_tolerance: float = 1e-6,
                         delta_params_tolerance: float = 1e-8,
                         objective_tolerance: float = 1e-12,
                         max_line_search_iterations: int = 20,
                         max_iterations: int = 100,
                         random_seed: int = 1,
                         restart_file: str = None):
    '''
    Run a finite-difference steepest-descent optimizer for inverse problems.

    The objective is a weighted observation misfit:
    0.5 * (observations - qoi)^T * C^{-1} * (observations - qoi),
    where C is observations_covariance (with optional regularization).

    Args:
        model: QoiModel to evaluate.
        parameter_space: ParameterSpace used for initialization if
            initial_parameters is not provided.
        observations: Observed QoI vector.
        observations_covariance: Observation covariance matrix used in objective.
        parameter_mins: Optional lower bounds.
        parameter_maxes: Optional upper bounds.
        absolute_steepest_descent_directory: Absolute path for optimizer runs.
        initial_parameters: Optional initial parameter vector.
        finite_difference_method: One of 'first_order', 'second_order',
            or 'fourth_order'.
        parameter_step_fraction: Scalar or per-parameter array defining finite
            difference step size as a fraction of each parameter value.
        minimum_parameter_step: Lower bound on finite difference step size.
        initial_line_search_step_size: Initial step size for line search.
        line_search_decay_factor: Backtracking divisor (> 1).
        armijo_sufficient_decrease: Armijo sufficient decrease constant.
        covariance_regularization: Diagonal regularization for covariance inversion.
        gradient_norm_tolerance: Convergence tolerance for gradient norm.
        delta_params_tolerance: Convergence tolerance for parameter update norm.
        objective_tolerance: Convergence tolerance for objective value.
        max_line_search_iterations: Maximum backtracking iterations.
        max_iterations: Maximum steepest-descent iterations.
        random_seed: RNG seed for initialization.
        restart_file: Optional restart file created by this routine.

    Returns:
        Tuple of (parameter_values, qoi) at termination.
    '''
    start_time = time.time()
    assert os.path.isabs(absolute_steepest_descent_directory), (
        "steepest_descent_directory is not an absolute path "
        f"({absolute_steepest_descent_directory})"
    )
    if line_search_decay_factor <= 1.0:
        raise ValueError("line_search_decay_factor must be greater than 1.0")
    if initial_line_search_step_size <= 0.0:
        raise ValueError("initial_line_search_step_size must be positive.")
    if minimum_parameter_step <= 0.0:
        raise ValueError("minimum_parameter_step must be positive.")
    if covariance_regularization < 0.0:
        raise ValueError("covariance_regularization cannot be negative.")
    if max_line_search_iterations <= 0:
        raise ValueError("max_line_search_iterations must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    finite_difference_method = _normalize_finite_difference_method(finite_difference_method)

    dimensionality = parameter_space.get_dimensionality()
    parameter_names = parameter_space.get_names()
    if parameter_mins is not None and np.size(parameter_mins) != dimensionality:
        raise ValueError(
            f"parameter_mins of size {np.size(parameter_mins)} is inconsistent "
            f"with parameter dimensionality ({dimensionality})."
        )
    if parameter_maxes is not None and np.size(parameter_maxes) != dimensionality:
        raise ValueError(
            f"parameter_maxes of size {np.size(parameter_maxes)} is inconsistent "
            f"with parameter dimensionality ({dimensionality})."
        )

    np.random.seed(random_seed)
    if restart_file is None:
        if initial_parameters is None:
            parameter_values = parameter_space.generate_samples(1)[0]
        else:
            parameter_values = np.asarray(initial_parameters, dtype=float)
            if parameter_values.shape != (dimensionality,):
                raise ValueError(
                    f"initial_parameters shape {parameter_values.shape} does not match "
                    f"required shape ({dimensionality},)."
                )
        parameter_values = _project_parameter_values(
            parameter_values,
            parameter_mins,
            parameter_maxes,
        )
        iteration = 0
    else:
        restart_data = np.load(restart_file)
        parameter_values = restart_data['parameter_values']
        if parameter_values.shape != (dimensionality,):
            raise ValueError(
                f"Restart parameter_values shape {parameter_values.shape} does not match "
                f"required shape ({dimensionality},)."
            )
        parameter_values = _project_parameter_values(
            parameter_values,
            parameter_mins,
            parameter_maxes,
        )
        iteration = int(restart_data['iteration']) if 'iteration' in restart_data else 0

    evaluation_cache = {}
    evaluation_counter = 0

    def _evaluate_at(candidate_parameters: np.ndarray, iteration_index: int, label: str):
        nonlocal evaluation_counter
        candidate_parameters = _project_parameter_values(
            np.asarray(candidate_parameters, dtype=float),
            parameter_mins,
            parameter_maxes,
        )
        cache_key = tuple(np.round(candidate_parameters, 15))
        if cache_key in evaluation_cache:
            return evaluation_cache[cache_key]
        run_directory = (
            f'{absolute_steepest_descent_directory}/iteration_{iteration_index}/'
            f'eval_{evaluation_counter}_{label}'
        )
        qoi, error, _ = prepare_and_run(
            model,
            observations,
            run_directory,
            parameter_names,
            candidate_parameters,
        )
        objective = _compute_weighted_error_objective(
            error,
            observations_covariance,
            covariance_regularization,
        )
        result = {
            'parameter_values': candidate_parameters.copy(),
            'qoi': qoi,
            'error': error,
            'objective': objective,
        }
        evaluation_cache[cache_key] = result
        evaluation_counter += 1
        return result

    current_state = _evaluate_at(parameter_values, iteration, 'base')
    parameter_values = current_state['parameter_values']
    qoi = current_state['qoi']
    error = current_state['error']
    objective = current_state['objective']

    fd_step_sizes = _compute_fractional_fd_steps(
        parameter_values,
        parameter_step_fraction,
        minimum_parameter_step,
        parameter_mins,
        parameter_maxes,
    )
    gradient = _compute_finite_difference_gradient(
        parameter_values,
        objective,
        finite_difference_method,
        fd_step_sizes,
        parameter_mins,
        parameter_maxes,
        lambda params, label: _evaluate_at(params, iteration, label)['objective'],
    )
    gradient_norm = np.linalg.norm(gradient)
    parameter_update_norm = np.inf

    wall_time = time.time() - start_time
    print(
        f'Iteration: {iteration}, Objective: {objective:.6e}, '
        f'Gradient 2-norm: {gradient_norm:.6e}, Delta p: {parameter_update_norm:.6e}, '
        f'Wall time: {wall_time:.6f}'
    )
    np.savez(
        f'{absolute_steepest_descent_directory}/iteration_{iteration}/restart.npz',
        parameter_values=parameter_values,
        qoi=qoi,
        error=error,
        objective=objective,
        gradient=gradient,
        iteration=iteration,
    )

    line_search_failed = False
    while (
        iteration < max_iterations
        and objective > objective_tolerance
        and gradient_norm > gradient_norm_tolerance
        and parameter_update_norm > delta_params_tolerance
    ):
        search_direction = -gradient.copy()
        directional_derivative = float(gradient @ search_direction)

        trial_step_size = initial_line_search_step_size
        accepted_trial_state = None
        accepted_parameter_update = None
        best_trial_state = None
        best_parameter_update = None
        best_trial_objective = np.inf
        for line_search_iteration in range(max_line_search_iterations):
            trial_parameters = _project_parameter_values(
                parameter_values + trial_step_size * search_direction,
                parameter_mins,
                parameter_maxes,
            )
            trial_update = trial_parameters - parameter_values
            if np.linalg.norm(trial_update) <= 0.0:
                trial_step_size /= line_search_decay_factor
                continue
            trial_state = _evaluate_at(
                trial_parameters,
                iteration + 1,
                f'line_search_{line_search_iteration}',
            )
            if trial_state['objective'] < best_trial_objective:
                best_trial_objective = trial_state['objective']
                best_trial_state = trial_state
                best_parameter_update = trial_update
            if trial_state['objective'] <= (
                objective + armijo_sufficient_decrease * trial_step_size * directional_derivative
            ):
                accepted_trial_state = trial_state
                accepted_parameter_update = trial_update
                break
            trial_step_size /= line_search_decay_factor

        if accepted_trial_state is None:
            if best_trial_state is not None and best_trial_state['objective'] < objective:
                accepted_trial_state = best_trial_state
                accepted_parameter_update = best_parameter_update
                print(
                    f'  Warning, Armijo condition failed after {max_line_search_iterations} '
                    f'tries at iteration {iteration}; accepting best objective-reducing trial step.'
                )
            else:
                print(
                    f'  Warning, failed line search after {max_line_search_iterations} '
                    f'tries at iteration {iteration}.'
                )
                line_search_failed = True
                break

        parameter_values = accepted_trial_state['parameter_values']
        qoi = accepted_trial_state['qoi']
        error = accepted_trial_state['error']
        objective = accepted_trial_state['objective']
        parameter_update_norm = np.linalg.norm(accepted_parameter_update)

        fd_step_sizes = _compute_fractional_fd_steps(
            parameter_values,
            parameter_step_fraction,
            minimum_parameter_step,
            parameter_mins,
            parameter_maxes,
        )
        gradient = _compute_finite_difference_gradient(
            parameter_values,
            objective,
            finite_difference_method,
            fd_step_sizes,
            parameter_mins,
            parameter_maxes,
            lambda params, label: _evaluate_at(params, iteration + 1, label)['objective'],
        )
        gradient_norm = np.linalg.norm(gradient)

        iteration += 1
        wall_time = time.time() - start_time
        print(
            f'Iteration: {iteration}, Objective: {objective:.6e}, '
            f'Gradient 2-norm: {gradient_norm:.6e}, Delta p: {parameter_update_norm:.6e}, '
            f'Wall time: {wall_time:.6f}'
        )
        np.savez(
            f'{absolute_steepest_descent_directory}/iteration_{iteration}/restart.npz',
            parameter_values=parameter_values,
            qoi=qoi,
            error=error,
            objective=objective,
            gradient=gradient,
            iteration=iteration,
        )

    if line_search_failed:
        print('Terminating due to line-search failure.')
    elif iteration >= max_iterations:
        print('Max iterations reached, terminating.')
    elif objective <= objective_tolerance:
        print('Objective dropped below tolerance.')
    elif gradient_norm <= gradient_norm_tolerance:
        print('Gradient norm dropped below tolerance.')
    elif parameter_update_norm <= delta_params_tolerance:
        print('Parameter update dropped below tolerance.')
    return parameter_values, qoi


def run_steepest_decent(*args, **kwargs):
    """Compatibility alias for the common misspelling 'decent'."""
    return run_steepest_descent(*args, **kwargs)
