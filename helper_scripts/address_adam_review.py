from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_all(text, old, new, expected, label):
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


# --- Adam solver: public diagnostics + restart serialization/validation ---
path = Path("romtools/workflows/inverse/vi_optimization_methods.py")
text = path.read_text()
anchor = """    def _current_fisher_damping(self) -> float:\n        if self.iteration < self.fisher_damping_decay_start:\n            return self.fisher_damping_initial\n        decay_exponent = -(\n            self.iteration - self.fisher_damping_decay_start\n        ) / self.fisher_damping_decay_start\n        return max(\n            self.fisher_damping_initial * np.exp(decay_exponent),\n            self.fisher_damping_min,\n        )\n\n    def _prepare_gradient(self, gradient: np.ndarray, fisher_diagonal=None) -> np.ndarray:\n"""
replacement = """    def _current_fisher_damping(self) -> float:\n        if self.iteration < self.fisher_damping_decay_start:\n            return self.fisher_damping_initial\n        decay_exponent = -(\n            self.iteration - self.fisher_damping_decay_start\n        ) / self.fisher_damping_decay_start\n        return max(\n            self.fisher_damping_initial * np.exp(decay_exponent),\n            self.fisher_damping_min,\n        )\n\n    def resolved_learning_rate(self, gradient: np.ndarray) -> float:\n        \"\"\"Return the learning rate used for the supplied VI gradient.\"\"\"\n        return self._resolved_learning_rate(np.asarray(gradient))\n\n    def _prepare_gradient(self, gradient: np.ndarray, fisher_diagonal=None) -> np.ndarray:\n"""
text = replace_once(text, anchor, replacement, "Adam learning-rate diagnostic")

anchor = """        if self.gradient_clip_norm is not None:\n            gradient_norm = np.linalg.norm(gradient)\n            if gradient_norm > self.gradient_clip_norm:\n                gradient = gradient * (self.gradient_clip_norm / gradient_norm)\n        return gradient\n\n    def step(self, gradient: np.ndarray, fisher_diagonal=None) -> np.ndarray:\n"""
replacement = """        if self.gradient_clip_norm is not None:\n            gradient_norm = np.linalg.norm(gradient)\n            if gradient_norm > self.gradient_clip_norm:\n                gradient = gradient * (self.gradient_clip_norm / gradient_norm)\n        return gradient\n\n    def prepare_gradient(self, gradient: np.ndarray, fisher_diagonal=None) -> np.ndarray:\n        \"\"\"Apply Fisher damping and clipping without updating Adam moments.\"\"\"\n        return self._prepare_gradient(gradient, fisher_diagonal=fisher_diagonal)\n\n    def step(self, gradient: np.ndarray, fisher_diagonal=None) -> np.ndarray:\n"""
text = replace_once(text, anchor, replacement, "Adam prepared-gradient diagnostic")

anchor = """    def state_dict(self) -> dict:\n        return {\n            'iteration': self.iteration,\n            'first_moment': None if self.first_moment is None else self.first_moment.copy(),\n            'second_moment': None if self.second_moment is None else self.second_moment.copy(),\n        }\n\n"""
replacement = """    def restart_state_dict(self) -> dict:\n        \"\"\"Return restart-safe Adam state and configuration metadata.\"\"\"\n        return {\n            'adam_iteration': int(self.iteration),\n            'adam_first_moment': (\n                np.array([], dtype=float)\n                if self.first_moment is None else self.first_moment.copy()\n            ),\n            'adam_second_moment': (\n                np.array([], dtype=float)\n                if self.second_moment is None else self.second_moment.copy()\n            ),\n            'adam_learning_rate': (\n                np.nan if self.learning_rate is None else float(self.learning_rate)\n            ),\n            'adam_learning_rate_scale': float(self.learning_rate_scale),\n            'adam_beta1': float(self.beta1),\n            'adam_beta2': float(self.beta2),\n            'adam_epsilon': float(self.epsilon),\n            'adam_fisher_damping_initial': float(self.fisher_damping_initial),\n            'adam_fisher_damping_decay_start': int(self.fisher_damping_decay_start),\n            'adam_fisher_damping_min': float(self.fisher_damping_min),\n            'adam_gradient_clip_norm': (\n                np.nan if self.gradient_clip_norm is None else float(self.gradient_clip_norm)\n            ),\n            'adam_parameter_dimension': (\n                -1 if self.parameter_dimension is None else int(self.parameter_dimension)\n            ),\n        }\n\n    def load_restart_state_dict(self, restart_data) -> None:\n        \"\"\"Restore Adam state and reject incompatible optimizer configuration.\"\"\"\n        required_keys = tuple(self.restart_state_dict().keys())\n        missing_keys = [key for key in required_keys if key not in restart_data]\n        if missing_keys:\n            raise ValueError(\n                \"Restart file is missing Adam optimizer state: \"\n                + \", \".join(missing_keys)\n            )\n\n        def _check_float(key, current_value):\n            saved_value = float(restart_data[key])\n            if current_value is None:\n                if not np.isnan(saved_value):\n                    raise ValueError(f\"Restart file {key} does not match current Adam config.\")\n            elif np.isnan(saved_value) or not np.isclose(saved_value, current_value):\n                raise ValueError(f\"Restart file {key} does not match current Adam config.\")\n\n        _check_float('adam_learning_rate', self.learning_rate)\n        _check_float('adam_learning_rate_scale', self.learning_rate_scale)\n        _check_float('adam_beta1', self.beta1)\n        _check_float('adam_beta2', self.beta2)\n        _check_float('adam_epsilon', self.epsilon)\n        _check_float('adam_fisher_damping_initial', self.fisher_damping_initial)\n        _check_float('adam_fisher_damping_min', self.fisher_damping_min)\n        _check_float('adam_gradient_clip_norm', self.gradient_clip_norm)\n        if int(restart_data['adam_fisher_damping_decay_start']) != self.fisher_damping_decay_start:\n            raise ValueError(\n                \"Restart file adam_fisher_damping_decay_start does not match current Adam config.\"\n            )\n        saved_parameter_dimension = int(restart_data['adam_parameter_dimension'])\n        current_parameter_dimension = (\n            -1 if self.parameter_dimension is None else int(self.parameter_dimension)\n        )\n        if saved_parameter_dimension != current_parameter_dimension:\n            raise ValueError(\n                \"Restart file adam_parameter_dimension does not match current Adam config.\"\n            )\n\n        self.iteration = int(restart_data['adam_iteration'])\n        first_moment = np.asarray(restart_data['adam_first_moment'], dtype=float)\n        second_moment = np.asarray(restart_data['adam_second_moment'], dtype=float)\n        if first_moment.size == 0 and second_moment.size == 0:\n            self.first_moment = None\n            self.second_moment = None\n        else:\n            if first_moment.shape != second_moment.shape:\n                raise ValueError(\"Restarted Adam moments must have matching shapes.\")\n            self.first_moment = first_moment.copy()\n            self.second_moment = second_moment.copy()\n        if self.iteration > 0 and self.first_moment is None:\n            raise ValueError(\"Restarted Adam state has a positive iteration but no moments.\")\n\n    def state_dict(self) -> dict:\n        return {\n            'iteration': self.iteration,\n            'first_moment': None if self.first_moment is None else self.first_moment.copy(),\n            'second_moment': None if self.second_moment is None else self.second_moment.copy(),\n        }\n\n"""
text = replace_once(text, anchor, replacement, "Adam restart state")
path.write_text(text)


# --- VI driver ---
path = Path("romtools/workflows/inverse/vi_drivers.py")
text = path.read_text()

anchor = """def _compute_gradient_norm(state, optimization_method: str):\n    if optimization_method == 'newton':\n        return np.sqrt(\n            np.linalg.norm(state['gradient_mean']) ** 2\n            + np.linalg.norm(state['gradient_log_std']) ** 2\n        )\n    return np.sqrt(\n        np.linalg.norm(state['update_direction_mean']) ** 2\n        + np.linalg.norm(state['update_direction_log_std']) ** 2\n    )\n\n\n"""
replacement = anchor + """def _compute_adam_fisher_diagonal(variational_log_std: np.ndarray,\n                                  min_variational_std: float,\n                                  max_variational_std: float,\n                                  gradient_method: str):\n    if gradient_method == 'standard':\n        return None\n    if gradient_method != 'natural':\n        raise ValueError(\n            f\"Unsupported Adam gradient_method '{gradient_method}'. \"\n            \"Supported options are 'standard' and 'natural'.\"\n        )\n    variational_std, _ = _compute_variational_std(\n        variational_log_std,\n        min_variational_std,\n        max_variational_std,\n    )\n    return np.concatenate([\n        1.0 / (variational_std ** 2),\n        2.0 * np.ones_like(variational_std),\n    ])\n\n\ndef _compute_adam_diagnostics(adam_solver: AdamSolver,\n                              state,\n                              variational_log_std: np.ndarray,\n                              min_variational_std: float,\n                              max_variational_std: float,\n                              gradient_method: str):\n    gradient = np.concatenate([state['gradient_mean'], state['gradient_log_std']])\n    fisher_diagonal = _compute_adam_fisher_diagonal(\n        variational_log_std,\n        min_variational_std,\n        max_variational_std,\n        gradient_method,\n    )\n    prepared_gradient = adam_solver.prepare_gradient(\n        gradient, fisher_diagonal=fisher_diagonal\n    )\n    return (\n        adam_solver.resolved_learning_rate(gradient),\n        float(np.linalg.norm(prepared_gradient)),\n    )\n\n\n"""
text = replace_once(text, anchor, replacement, "VI Adam diagnostics helpers")

old = """    line_search_method, resolved_line_search_config = _resolve_line_search_config(\n        line_search_method,\n        line_search_config,\n        VILegacyLineSearchConfig(),\n        VIStochasticNonmonotoneLineSearchConfig(),\n    )\n\n    if optimization_method == 'adam':\n"""
new = """    if optimization_method == 'adam' and line_search_config is not None:\n        raise ValueError(\n            \"line_search_config is not supported with optimizer_method='adam'; \"\n            \"Adam supplies the complete update step.\"\n        )\n    line_search_method, resolved_line_search_config = _resolve_line_search_config(\n        line_search_method,\n        line_search_config,\n        VILegacyLineSearchConfig(),\n        VIStochasticNonmonotoneLineSearchConfig(),\n    )\n\n    if optimization_method == 'adam':\n"""
text = replace_once(text, old, new, "VI Adam line-search validation")

old = """        if restart_file is not None:\n            warnings.warn(\n                \"Adam optimizer moments are not stored in VI restart files; \"\n                \"restarting resets the Adam first- and second-moment state.\",\n                RuntimeWarning,\n                stacklevel=2,\n            )\n\n"""
text = replace_once(text, old, "", "remove VI Adam restart warning")

old = """    variational_distribution = _normalize_variational_distribution(variational_distribution)\n    line_search_objective = _normalize_line_search_objective(line_search_objective)\n"""
new = """    variational_distribution = _normalize_variational_distribution(variational_distribution)\n    if (\n        optimization_method == 'adam'\n        and gradient_method == 'natural'\n        and variational_distribution == 'multivariate'\n    ):\n        raise NotImplementedError(\n            \"Natural-gradient Adam is not supported for correlated multivariate \"\n            \"Gaussian VI. Use VIAdamOptimizerConfig(gradient_method='standard') instead.\"\n        )\n    line_search_objective = _normalize_line_search_objective(line_search_objective)\n"""
text = replace_once(text, old, new, "VI multivariate Adam guard")

# Extend restart writer.
old = """                     running_hessian: np.ndarray = None,\n                     accepted_elbo_history=None,\n                     dispatcher: Optional[BaseDispatcher] = None):\n"""
new = """                     running_hessian: np.ndarray = None,\n                     accepted_elbo_history=None,\n                     adam_restart_data=None,\n                     adam_gradient_method: str = None,\n                     dispatcher: Optional[BaseDispatcher] = None):\n"""
text = replace_once(text, old, new, "VI restart signature")
old = """    if variational_correlation_cholesky is not None:\n        save_data['variational_correlation_cholesky'] = variational_correlation_cholesky\n    resolve_dispatcher(dispatcher).np_savez(restart_path, **save_data)\n"""
new = """    if variational_correlation_cholesky is not None:\n        save_data['variational_correlation_cholesky'] = variational_correlation_cholesky\n    if adam_restart_data is not None:\n        save_data.update(adam_restart_data)\n        save_data['adam_gradient_method'] = adam_gradient_method\n    resolve_dispatcher(dispatcher).np_savez(restart_path, **save_data)\n"""
text = replace_once(text, old, new, "VI restart Adam payload")

# Initialize/load the optimizer before any restart save.
old = """    accepted_elbo_history = (\n        restart_data['accepted_elbo_history'].tolist()\n        if restart_file is not None and 'accepted_elbo_history' in restart_data\n        else [float(state['elbo'])]\n    )\n\n    if restart_file is None:\n"""
new = """    accepted_elbo_history = (\n        restart_data['accepted_elbo_history'].tolist()\n        if restart_file is not None and 'accepted_elbo_history' in restart_data\n        else [float(state['elbo'])]\n    )\n\n    steepest_descent_solver = (\n        AdamSolver.from_config(resolved_optimizer_config)\n        if optimization_method == 'adam'\n        else SteepestDescentSolver()\n    )\n    if optimization_method == 'adam' and restart_file is not None:\n        if 'adam_gradient_method' not in restart_data:\n            raise ValueError(\"Restart file is missing Adam gradient-method state.\")\n        restart_adam_gradient_method = str(restart_data['adam_gradient_method'].item())\n        if restart_adam_gradient_method != gradient_method:\n            raise ValueError(\n                \"Restart file adam_gradient_method does not match current Adam config.\"\n            )\n        steepest_descent_solver.load_restart_state_dict(restart_data)\n\n    if restart_file is None:\n"""
text = replace_once(text, old, new, "VI Adam optimizer initialization")

old = """            accepted_elbo_history=accepted_elbo_history,\n            dispatcher=dispatcher,\n        )\n"""
new = """            accepted_elbo_history=accepted_elbo_history,\n            adam_restart_data=(\n                steepest_descent_solver.restart_state_dict()\n                if optimization_method == 'adam' else None\n            ),\n            adam_gradient_method=(gradient_method if optimization_method == 'adam' else None),\n            dispatcher=dispatcher,\n        )\n"""
text = replace_once(text, old, new, "VI initial Adam restart save")

old = """    iteration += 1\n    step_failed_counter = 0\n    line_search_standard_normal_cache = None\n    steepest_descent_solver = (\n        AdamSolver.from_config(resolved_optimizer_config)\n        if optimization_method == 'adam'\n        else SteepestDescentSolver()\n    )\n    elbo_converged = False\n"""
new = """    iteration += 1\n    step_failed_counter = 0\n    line_search_standard_normal_cache = None\n    elbo_converged = False\n"""
text = replace_once(text, old, new, "remove duplicate VI Adam initialization")

# Use one Fisher helper in the Adam step.
old = """                fisher_diagonal = None\n                if gradient_method == 'natural':\n                    variational_std_for_fisher, _ = _compute_variational_std(\n                        variational_log_std,\n                        min_variational_std,\n                        max_variational_std,\n                    )\n                    fisher_diagonal = np.concatenate([\n                        1.0 / (variational_std_for_fisher ** 2),\n                        2.0 * np.ones_like(variational_std_for_fisher),\n                    ])\n"""
new = """                fisher_diagonal = _compute_adam_fisher_diagonal(\n                    variational_log_std,\n                    min_variational_std,\n                    max_variational_std,\n                    gradient_method,\n                )\n"""
text = replace_once(text, old, new, "VI Adam Fisher helper")

# Initial gradient norm/status.
old = """    gradient_norm = _compute_gradient_norm(state, optimization_method)\n    wall_time = time.time() - start_time\n"""
new = """    if optimization_method == 'adam':\n        adam_learning_rate, gradient_norm = _compute_adam_diagnostics(\n            steepest_descent_solver,\n            state,\n            variational_log_std,\n            min_variational_std,\n            max_variational_std,\n            gradient_method,\n        )\n    else:\n        adam_learning_rate = None\n        gradient_norm = _compute_gradient_norm(state, optimization_method)\n    wall_time = time.time() - start_time\n"""
text = replace_once(text, old, new, "VI initial Adam diagnostics")

old = """    print(\n        f'Iteration: {iteration}, Relative MSE: {state[\"mean_relative_mse\"]:.5f}, '\n        f'ELBO: {state[\"elbo\"]:.5f}, Step size: {step_size:.5f}, '\n        f'Gradient norm: {gradient_norm:.5f}, Wall time: {wall_time:.5f}'\n    )\n"""
new = """    optimizer_status = (\n        f'Adam learning rate: {adam_learning_rate:.5e}, '\n        f'Adam input gradient norm: {gradient_norm:.5f}'\n        if optimization_method == 'adam'\n        else f'Step size: {step_size:.5f}, Gradient norm: {gradient_norm:.5f}'\n    )\n    print(\n        f'Iteration: {iteration}, Relative MSE: {state[\"mean_relative_mse\"]:.5f}, '\n        f'ELBO: {state[\"elbo\"]:.5f}, {optimizer_status}, Wall time: {wall_time:.5f}'\n    )\n"""
text = replace_once(text, old, new, "VI initial Adam status")

# Accepted-step diagnostics and print.
old = """                gradient_norm = _compute_gradient_norm(state, optimization_method)\n                wall_time = time.time() - start_time\n"""
new = """                if optimization_method == 'adam':\n                    adam_learning_rate, gradient_norm = _compute_adam_diagnostics(\n                        steepest_descent_solver,\n                        state,\n                        variational_log_std,\n                        min_variational_std,\n                        max_variational_std,\n                        gradient_method,\n                    )\n                else:\n                    gradient_norm = _compute_gradient_norm(state, optimization_method)\n                wall_time = time.time() - start_time\n"""
text = replace_once(text, old, new, "VI accepted Adam diagnostics")

old = """                print(\n                    f'Iteration: {iteration}, Relative MSE: {state[\"mean_relative_mse\"]:.5f}, '\n                    f'ELBO: {state[\"elbo\"]:.5f}, Relative ELBO (initial ref): {relative_elbo_improvement:.5e}, '\n                    f'Step size: {step_size:.5e}, '\n                    f'Gradient norm: {gradient_norm:.5f}, Samples: {line_search_sample_size}, '\n                    f'Wall time: {wall_time:.5f}'\n                )\n"""
new = """                optimizer_status = (\n                    f'Adam learning rate: {adam_learning_rate:.5e}, '\n                    f'Adam input gradient norm: {gradient_norm:.5f}'\n                    if optimization_method == 'adam'\n                    else f'Step size: {step_size:.5e}, Gradient norm: {gradient_norm:.5f}'\n                )\n                print(\n                    f'Iteration: {iteration}, Relative MSE: {state[\"mean_relative_mse\"]:.5f}, '\n                    f'ELBO: {state[\"elbo\"]:.5f}, Relative ELBO (initial ref): {relative_elbo_improvement:.5e}, '\n                    f'{optimizer_status}, Samples: {line_search_sample_size}, '\n                    f'Wall time: {wall_time:.5f}'\n                )\n"""
text = replace_once(text, old, new, "VI accepted Adam status")

# Save Adam state on the gradient/Adam accepted path.
old = """                    max_mean_update_std=max_mean_update_std,\n                    dispatcher=dispatcher,\n                )\n"""
new = """                    max_mean_update_std=max_mean_update_std,\n                    adam_restart_data=(\n                        steepest_descent_solver.restart_state_dict()\n                        if optimization_method == 'adam' else None\n                    ),\n                    adam_gradient_method=(\n                        gradient_method if optimization_method == 'adam' else None\n                    ),\n                    dispatcher=dispatcher,\n                )\n"""
text = replace_once(text, old, new, "VI accepted Adam restart save")

path.write_text(text)


# --- MF-VI driver ---
path = Path("romtools/workflows/inverse/mf_vi_drivers.py")
text = path.read_text()

# Import shared Adam helpers.
old = """    _compute_gradient_norm,\n    _compute_leave_one_out_baseline,\n"""
new = """    _compute_gradient_norm,\n    _compute_adam_fisher_diagonal,\n    _compute_adam_diagnostics,\n    _compute_leave_one_out_baseline,\n"""
text = replace_once(text, old, new, "MFVI Adam helper imports")

old = """    line_search_method, resolved_line_search_config = _resolve_line_search_config(\n        line_search_method,\n        line_search_config,\n        VILegacyLineSearchConfig(\n"""
new = """    if optimization_method == 'adam' and line_search_config is not None:\n        raise ValueError(\n            \"line_search_config is not supported with optimizer_method='adam'; \"\n            \"Adam supplies the complete update step.\"\n        )\n    line_search_method, resolved_line_search_config = _resolve_line_search_config(\n        line_search_method,\n        line_search_config,\n        VILegacyLineSearchConfig(\n"""
text = replace_once(text, old, new, "MFVI Adam line-search validation")

old = """        if restart_file is not None:\n            warnings.warn(\n                \"Adam optimizer moments are not stored in MF-VI restart files; \"\n                \"restarting resets the Adam first- and second-moment state.\",\n                RuntimeWarning,\n                stacklevel=2,\n            )\n\n"""
text = replace_once(text, old, "", "remove MFVI Adam restart warning")

old = """    ) = _validate_gaussian_parameter_spaces(\n        prior_parameter_space,\n        initial_variational_parameter_space,\n    )\n    sampling_method = _normalize_sampling_method(sampling_method)\n"""
new = """    ) = _validate_gaussian_parameter_spaces(\n        prior_parameter_space,\n        initial_variational_parameter_space,\n    )\n    if (\n        optimization_method == 'adam'\n        and gradient_method == 'natural'\n        and variational_distribution == 'multivariate'\n    ):\n        raise NotImplementedError(\n            \"Natural-gradient Adam is not supported for correlated multivariate \"\n            \"Gaussian MF-VI. Use VIAdamOptimizerConfig(gradient_method='standard') instead.\"\n        )\n    sampling_method = _normalize_sampling_method(sampling_method)\n"""
text = replace_once(text, old, new, "MFVI multivariate Adam guard")

# Extend restart writer.
old = """                        running_hessian: np.ndarray = None,\n                        accepted_elbo_history=None,\n                        dispatcher: Optional[BaseDispatcher] = None):\n"""
new = """                        running_hessian: np.ndarray = None,\n                        accepted_elbo_history=None,\n                        adam_restart_data=None,\n                        adam_gradient_method: str = None,\n                        dispatcher: Optional[BaseDispatcher] = None):\n"""
text = replace_once(text, old, new, "MFVI restart signature")
old = """    if variational_correlation_cholesky is not None:\n        save_data['variational_correlation_cholesky'] = variational_correlation_cholesky\n    resolve_dispatcher(dispatcher).np_savez(restart_path, **save_data)\n"""
new = """    if variational_correlation_cholesky is not None:\n        save_data['variational_correlation_cholesky'] = variational_correlation_cholesky\n    if adam_restart_data is not None:\n        save_data.update(adam_restart_data)\n        save_data['adam_gradient_method'] = adam_gradient_method\n    resolve_dispatcher(dispatcher).np_savez(restart_path, **save_data)\n"""
text = replace_once(text, old, new, "MFVI restart Adam payload")

# Initialize/load Adam before initial restart save.
old = """    accepted_elbo_history = (\n        restart_data['accepted_elbo_history'].tolist()\n        if restart_file is not None and 'accepted_elbo_history' in restart_data\n        else [float(state['elbo'])]\n    )\n\n    _save_mf_vi_restart(\n"""
new = """    accepted_elbo_history = (\n        restart_data['accepted_elbo_history'].tolist()\n        if restart_file is not None and 'accepted_elbo_history' in restart_data\n        else [float(state['elbo'])]\n    )\n\n    steepest_descent_solver = (\n        AdamSolver.from_config(resolved_optimizer_config)\n        if optimization_method == 'adam'\n        else SteepestDescentSolver()\n    )\n    if optimization_method == 'adam' and restart_file is not None:\n        if 'adam_gradient_method' not in restart_data:\n            raise ValueError(\"restart_file is missing Adam gradient-method state.\")\n        restart_adam_gradient_method = str(restart_data['adam_gradient_method'].item())\n        if restart_adam_gradient_method != gradient_method:\n            raise ValueError(\n                \"restart_file adam_gradient_method does not match current Adam config.\"\n            )\n        steepest_descent_solver.load_restart_state_dict(restart_data)\n\n    _save_mf_vi_restart(\n"""
text = replace_once(text, old, new, "MFVI Adam optimizer initialization")

old = """        accepted_elbo_history=accepted_elbo_history,\n        dispatcher=dispatcher,\n    )\n"""
new = """        accepted_elbo_history=accepted_elbo_history,\n        adam_restart_data=(\n            steepest_descent_solver.restart_state_dict()\n            if optimization_method == 'adam' else None\n        ),\n        adam_gradient_method=(gradient_method if optimization_method == 'adam' else None),\n        dispatcher=dispatcher,\n    )\n"""
text = replace_once(text, old, new, "MFVI initial Adam restart save")

old = """    iteration += 1\n    step_failed_counter = 0\n    steepest_descent_solver = (\n        AdamSolver.from_config(resolved_optimizer_config)\n        if optimization_method == 'adam'\n        else SteepestDescentSolver()\n    )\n    elbo_converged = False\n"""
new = """    iteration += 1\n    step_failed_counter = 0\n    elbo_converged = False\n"""
text = replace_once(text, old, new, "remove duplicate MFVI Adam initialization")

old = """                fisher_diagonal = None\n                if gradient_method == 'natural':\n                    variational_std_for_fisher, _ = _compute_variational_std(\n                        variational_log_std,\n                        min_variational_std,\n                        max_variational_std,\n                    )\n                    fisher_diagonal = np.concatenate([\n                        1.0 / (variational_std_for_fisher ** 2),\n                        2.0 * np.ones_like(variational_std_for_fisher),\n                    ])\n"""
new = """                fisher_diagonal = _compute_adam_fisher_diagonal(\n                    variational_log_std,\n                    min_variational_std,\n                    max_variational_std,\n                    gradient_method,\n                )\n"""
text = replace_once(text, old, new, "MFVI Adam Fisher helper")

old = """    gradient_norm = _compute_gradient_norm(state, optimization_method)\n    wall_time = time.time() - start_time\n"""
new = """    if optimization_method == 'adam':\n        adam_learning_rate, gradient_norm = _compute_adam_diagnostics(\n            steepest_descent_solver,\n            state,\n            variational_log_std,\n            min_variational_std,\n            max_variational_std,\n            gradient_method,\n        )\n    else:\n        adam_learning_rate = None\n        gradient_norm = _compute_gradient_norm(state, optimization_method)\n    wall_time = time.time() - start_time\n"""
text = replace_once(text, old, new, "MFVI initial Adam diagnostics")

old = """    print(\n        f'Iteration: {iteration}, Relative MSE: {state[\"mean_relative_mse\"]:.5f}, '\n        f'ELBO: {state[\"elbo\"]:.5f}, ROM err: {state[\"rom_error\"]:.5f}, '\n        f'alpha_mean: {alpha_mean_scalar:.5f}, alpha_logstd: {alpha_log_scalar:.5f}, '\n        f'Step size: {step_size:.5f}, Gradient norm: {gradient_norm:.5f}, Wall time: {wall_time:.5f}'\n    )\n"""
new = """    optimizer_status = (\n        f'Adam learning rate: {adam_learning_rate:.5e}, '\n        f'Adam input gradient norm: {gradient_norm:.5f}'\n        if optimization_method == 'adam'\n        else f'Step size: {step_size:.5f}, Gradient norm: {gradient_norm:.5f}'\n    )\n    print(\n        f'Iteration: {iteration}, Relative MSE: {state[\"mean_relative_mse\"]:.5f}, '\n        f'ELBO: {state[\"elbo\"]:.5f}, ROM err: {state[\"rom_error\"]:.5f}, '\n        f'alpha_mean: {alpha_mean_scalar:.5f}, alpha_logstd: {alpha_log_scalar:.5f}, '\n        f'{optimizer_status}, Wall time: {wall_time:.5f}'\n    )\n"""
text = replace_once(text, old, new, "MFVI initial Adam status")

old = """            gradient_norm = _compute_gradient_norm(state, optimization_method)\n            wall_time = time.time() - start_time\n"""
new = """            if optimization_method == 'adam':\n                adam_learning_rate, gradient_norm = _compute_adam_diagnostics(\n                    steepest_descent_solver,\n                    state,\n                    variational_log_std,\n                    min_variational_std,\n                    max_variational_std,\n                    gradient_method,\n                )\n            else:\n                gradient_norm = _compute_gradient_norm(state, optimization_method)\n            wall_time = time.time() - start_time\n"""
text = replace_once(text, old, new, "MFVI accepted Adam diagnostics")

old = """            print(\n                f'Iteration: {iteration}, Relative MSE: {state[\"mean_relative_mse\"]:.5f}, ELBO: {state[\"elbo\"]:.5f}, '\n                f'Relative ELBO (initial ref): {relative_elbo_improvement:.5e}, '\n                f'ROM err: {state[\"rom_error\"]:.5f}, alpha_mean: {alpha_mean_scalar:.5f}, '\n                f'alpha_logstd: {alpha_log_scalar:.5f}, Step size: {step_size:.5e}, '\n                f'Gradient norm: {gradient_norm:.5f}, Wall time: {wall_time:.5f}'\n            )\n"""
new = """            optimizer_status = (\n                f'Adam learning rate: {adam_learning_rate:.5e}, '\n                f'Adam input gradient norm: {gradient_norm:.5f}'\n                if optimization_method == 'adam'\n                else f'Step size: {step_size:.5e}, Gradient norm: {gradient_norm:.5f}'\n            )\n            print(\n                f'Iteration: {iteration}, Relative MSE: {state[\"mean_relative_mse\"]:.5f}, ELBO: {state[\"elbo\"]:.5f}, '\n                f'Relative ELBO (initial ref): {relative_elbo_improvement:.5e}, '\n                f'ROM err: {state[\"rom_error\"]:.5f}, alpha_mean: {alpha_mean_scalar:.5f}, '\n                f'alpha_logstd: {alpha_log_scalar:.5f}, {optimizer_status}, Wall time: {wall_time:.5f}'\n            )\n"""
text = replace_once(text, old, new, "MFVI accepted Adam status")

old = """                accepted_elbo_history=accepted_elbo_history,\n                dispatcher=dispatcher,\n            )\n"""
new = """                accepted_elbo_history=accepted_elbo_history,\n                adam_restart_data=(\n                    steepest_descent_solver.restart_state_dict()\n                    if optimization_method == 'adam' else None\n                ),\n                adam_gradient_method=(\n                    gradient_method if optimization_method == 'adam' else None\n                ),\n                dispatcher=dispatcher,\n            )\n"""
text = replace_once(text, old, new, "MFVI accepted Adam restart save")
path.write_text(text)


# --- Tests ---
path = Path("tests/romtools/workflows/inverse/test_vi_adam.py")
text = path.read_text()
text = replace_once(
    text,
    "from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler\n",
    "from romtools.workflows.parameter_spaces import (\n    GaussianParameterSpace,\n    MultivariateGaussianParameterSpace,\n    MonteCarloSampler,\n)\n",
    "Adam test imports",
)

anchor = """class LinearQoiRomBuilderWithTrainingData:\n    def __init__(self, slope=1.0):\n        self._model = LinearQoiModel(slope=slope)\n\n    def build_from_training_dirs(self, offline_data_dir, training_data_dirs,\n                                 training_parameters, training_qois):\n        _ = (offline_data_dir, training_data_dirs, training_parameters, training_qois)\n        return self._model\n\n\n"""
addition = anchor + """class TwoParameterLinearQoiModel:\n    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:\n        return\n\n    def run_model(self, run_directory: str, parameter_sample: dict) -> int:\n        return 0\n\n    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:\n        return np.array([\n            float(parameter_sample['theta0']) + 0.5 * float(parameter_sample['theta1'])\n        ])\n\n\n"""
text = replace_once(text, anchor, addition, "two-parameter Adam test model")

anchor = """def test_adam_solver_clips_preconditioned_gradient_norm():\n    solver = AdamSolver(learning_rate=0.01, gradient_clip_norm=5.0)\n    gradient = np.array([30.0, 40.0])\n\n    solver.step(gradient)\n\n    prepared_gradient = solver.first_moment / (1.0 - solver.beta1)\n    assert np.isclose(np.linalg.norm(prepared_gradient), 5.0)\n\n\n"""
addition = anchor + """def test_adam_solver_restart_round_trip_preserves_state_and_schedule():\n    solver = AdamSolver(learning_rate=0.02, fisher_damping_decay_start=2)\n    solver.step(np.array([1.0, -2.0]), fisher_diagonal=np.array([2.0, 2.0]))\n    solver.step(np.array([3.0, 4.0]), fisher_diagonal=np.array([2.0, 2.0]))\n    restart_data = solver.restart_state_dict()\n\n    restored = AdamSolver(learning_rate=0.02, fisher_damping_decay_start=2)\n    restored.load_restart_state_dict(restart_data)\n\n    assert restored.iteration == solver.iteration\n    np.testing.assert_allclose(restored.first_moment, solver.first_moment)\n    np.testing.assert_allclose(restored.second_moment, solver.second_moment)\n    assert np.isclose(restored._current_fisher_damping(), solver._current_fisher_damping())\n\n\n"""
text = replace_once(text, anchor, addition, "Adam restart unit test")

append = r'''

@pytest.mark.mpi_skip
def test_run_vi_adam_restart_matches_uninterrupted_run(tmp_path):
    common = dict(
        model=LinearQoiModel(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.4]),
        observations_covariance=np.array([[0.2]]),
        sample_size=8,
        optimizer_method="adam",
        bounded_parameter_handling="clip",
        random_seed=11,
        evaluation_concurrency=1,
    )
    final_config = VIAdamOptimizerConfig(
        learning_rate=0.02,
        gradient_norm_tolerance=0.0,
        max_iterations=4,
    )
    uninterrupted = vi_drivers.run_vi(
        absolute_work_dir=str(tmp_path / "uninterrupted"),
        optimizer_config=final_config,
        **common,
    )
    vi_drivers.run_vi(
        absolute_work_dir=str(tmp_path / "split"),
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.02,
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        **common,
    )
    restarted = vi_drivers.run_vi(
        absolute_work_dir=str(tmp_path / "split"),
        restart_file=str(tmp_path / "split" / "iteration_1" / "restart.npz"),
        optimizer_config=final_config,
        **common,
    )

    for uninterrupted_value, restarted_value in zip(uninterrupted, restarted):
        np.testing.assert_allclose(uninterrupted_value, restarted_value)


@pytest.mark.mpi_skip
def test_run_mf_vi_adam_restart_matches_uninterrupted_run(tmp_path):
    common = dict(
        model=LinearQoiModel(),
        rom_model_builder=LinearQoiRomBuilderWithTrainingData(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.4]),
        observations_covariance=np.array([[0.2]]),
        fom_sample_size=4,
        rom_extra_sample_size=4,
        rom_tolerance=1.0,
        optimizer_method="adam",
        bounded_parameter_handling="clip",
        random_seed=11,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
    )
    final_config = VIAdamOptimizerConfig(
        learning_rate=0.02,
        gradient_norm_tolerance=0.0,
        max_iterations=4,
    )
    uninterrupted = mf_vi_drivers.run_mf_vi(
        absolute_work_dir=str(tmp_path / "mf_uninterrupted"),
        optimizer_config=final_config,
        **common,
    )
    mf_vi_drivers.run_mf_vi(
        absolute_work_dir=str(tmp_path / "mf_split"),
        optimizer_config=VIAdamOptimizerConfig(
            learning_rate=0.02,
            gradient_norm_tolerance=0.0,
            max_iterations=2,
        ),
        **common,
    )
    restarted = mf_vi_drivers.run_mf_vi(
        absolute_work_dir=str(tmp_path / "mf_split"),
        restart_file=str(tmp_path / "mf_split" / "iteration_1" / "restart.npz"),
        optimizer_config=final_config,
        **common,
    )

    for uninterrupted_value, restarted_value in zip(uninterrupted, restarted):
        np.testing.assert_allclose(uninterrupted_value, restarted_value)


@pytest.mark.mpi_skip
def test_adam_natural_gradient_rejects_correlated_multivariate_vi(tmp_path):
    parameter_space = MultivariateGaussianParameterSpace(
        parameter_names=["theta0", "theta1"],
        means=np.array([0.0, 0.0]),
        covariance=np.array([[1.0, 0.4], [0.4, 1.0]]),
        sampler=MonteCarloSampler,
    )
    with pytest.raises(NotImplementedError, match="correlated multivariate"):
        vi_drivers.run_vi(
            model=TwoParameterLinearQoiModel(),
            prior_parameter_space=parameter_space,
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            absolute_work_dir=str(tmp_path / "multivariate"),
            sample_size=4,
            optimizer_method="adam",
            optimizer_config=VIAdamOptimizerConfig(),
            bounded_parameter_handling="clip",
            evaluation_concurrency=1,
        )


@pytest.mark.mpi_skip
def test_adam_rejects_explicit_line_search_config(tmp_path):
    with pytest.raises(ValueError, match="line_search_config is not supported"):
        vi_drivers.run_vi(
            model=LinearQoiModel(),
            prior_parameter_space=_parameter_space(),
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            absolute_work_dir=str(tmp_path / "line_search"),
            optimizer_method="adam",
            optimizer_config=VIAdamOptimizerConfig(max_iterations=1),
            line_search_config=romtools.workflows.VIStochasticNonmonotoneLineSearchConfig(),
        )


@pytest.mark.mpi_skip
def test_adam_diagnostics_report_learning_rate_not_dummy_step_size(tmp_path, capsys):
    vi_drivers.run_vi(
        model=LinearQoiModel(),
        prior_parameter_space=_parameter_space(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        absolute_work_dir=str(tmp_path / "diagnostics"),
        sample_size=4,
        optimizer_method="adam",
        optimizer_config=VIAdamOptimizerConfig(max_iterations=1),
        bounded_parameter_handling="clip",
        evaluation_concurrency=1,
    )
    output = capsys.readouterr().out
    assert "Adam learning rate:" in output
    assert "Adam input gradient norm:" in output
    assert "Step size:" not in output
'''
text += append
path.write_text(text)
