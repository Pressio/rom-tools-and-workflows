from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_after(text, anchor, old, new, label):
    anchor_index = text.find(anchor)
    if anchor_index < 0:
        raise RuntimeError(f"{label}: anchor not found")
    match_index = text.find(old, anchor_index)
    if match_index < 0:
        raise RuntimeError(f"{label}: target not found after anchor")
    return text[:match_index] + new + text[match_index + len(old):]


def patch_vi(path):
    text = path.read_text()
    text = replace_once(
        text,
        "from romtools.workflows.inverse.vi_optimization_methods import (\n    NewtonSolver,\n    SteepestDescentSolver,\n    VIGradientOptimizerConfig,",
        "from romtools.workflows.inverse.vi_optimization_methods import (\n    AdamSolver,\n    NewtonSolver,\n    SteepestDescentSolver,\n    VIAdamOptimizerConfig,\n    VIGradientOptimizerConfig,",
        "VI imports",
    )
    text = replace_once(
        text,
        "When ``optimizer_method=\"newton\"``, the routine also forms a second-order",
        "When ``optimizer_method=\"adam\"``, the selected score gradient (standard or\n"
        "natural/Fisher-preconditioned) is passed to a stateful Adam ascent update.\n"
        "The Adam default uses the natural gradient and the ABRIS initial learning-rate\n"
        "convention :math:`0.1/d`, where :math:`d` is the parameter dimension.\n\n"
        "When ``optimizer_method=\"newton\"``, the routine also forms a second-order",
        "VI module Adam docs",
    )
    text = replace_once(
        text,
        "        optimizer_method: Optimizer used for variational updates. Supported\n            options are 'gradient' and 'newton'.\n        optimizer_config: Method-specific optimizer config. Expected types are\n            `VIGradientOptimizerConfig` for optimizer_method='gradient',\n            `VINewtonOptimizerConfig` for optimizer_method='newton'.",
        "        optimizer_method: Optimizer used for variational updates. Supported\n            options are 'gradient', 'adam', and 'newton'.\n        optimizer_config: Method-specific optimizer config. Expected types are\n            `VIGradientOptimizerConfig` for optimizer_method='gradient',\n            `VIAdamOptimizerConfig` for optimizer_method='adam', and\n            `VINewtonOptimizerConfig` for optimizer_method='newton'.",
        "VI run docs",
    )
    text = replace_once(
        text,
        "        VIGradientOptimizerConfig(),\n        VINewtonOptimizerConfig(),\n    )\n    line_search_method, resolved_line_search_config = _resolve_line_search_config(",
        "        VIGradientOptimizerConfig(),\n        VINewtonOptimizerConfig(),\n        VIAdamOptimizerConfig(),\n    )\n    line_search_method, resolved_line_search_config = _resolve_line_search_config(",
        "VI optimizer resolver",
    )
    text = replace_once(
        text,
        "    gradient_method = 'standard'\n    if optimization_method == 'gradient':\n        gradient_method = resolved_optimizer_config.gradient_method",
        "    if optimization_method == 'adam':\n        line_search_method = 'legacy'\n        resolved_line_search_config = VILegacyLineSearchConfig(\n            initial_step_size=1.0,\n            max_step_size=1.0,\n            step_size_growth_factor=1.0,\n            step_size_decay_factor=1.0,\n            max_step_size_decrease_trys=0,\n            relaxation_parameter=1.0,\n            line_search_objective='elbo',\n            line_search_sample_growth_factor=1.0,\n            log_std_learning_rate_factor=1.0,\n        )\n        if restart_file is not None:\n            warnings.warn(\n                \"Adam optimizer moments are not stored in VI restart files; \"\n                \"restarting resets the Adam first- and second-moment state.\",\n                RuntimeWarning,\n                stacklevel=2,\n            )\n\n    gradient_method = 'standard'\n    if optimization_method in ('gradient', 'adam'):\n        gradient_method = resolved_optimizer_config.gradient_method",
        "VI Adam setup",
    )
    text = replace_once(
        text,
        "    steepest_descent_solver = SteepestDescentSolver()",
        "    steepest_descent_solver = (\n        AdamSolver.from_config(resolved_optimizer_config)\n        if optimization_method == 'adam'\n        else SteepestDescentSolver()\n    )",
        "VI solver construction",
    )
    gradient_anchor = "        if optimization_method in ('gradient', 'adam'):\n"
    text = replace_once(
        text,
        "        if optimization_method == 'gradient':\n            gradient = np.concatenate([state['update_direction_mean'], state['update_direction_log_std']])",
        "        if optimization_method in ('gradient', 'adam'):\n            gradient = np.concatenate([state['update_direction_mean'], state['update_direction_log_std']])",
        "VI gradient branch",
    )
    text = replace_after(
        text,
        gradient_anchor,
        "            if line_search_objective == 'elbo':\n                test_state = test_candidate['state']\n                if line_search_method == 'legacy':",
        "            if optimization_method == 'adam':\n                test_state = test_candidate['state']\n                accept_step = True\n            elif line_search_objective == 'elbo':\n                test_state = test_candidate['state']\n                if line_search_method == 'legacy':",
        "VI Adam acceptance",
    )
    path.write_text(text)


def patch_mf_vi(path):
    text = path.read_text()
    text = replace_once(
        text,
        "from romtools.workflows.inverse.vi_optimization_methods import (\n    SteepestDescentSolver,\n    VIGradientOptimizerConfig,",
        "from romtools.workflows.inverse.vi_optimization_methods import (\n    AdamSolver,\n    SteepestDescentSolver,\n    VIAdamOptimizerConfig,\n    VIGradientOptimizerConfig,",
        "MFVI imports",
    )
    text = replace_once(
        text,
        "        optimizer_method: Optimizer used for variational updates. Supported\n            options are 'gradient' and 'newton'.\n        optimizer_config: Method-specific optimizer config. Expected types are\n            `VIGradientOptimizerConfig` for optimizer_method='gradient',\n            `VINewtonOptimizerConfig` for optimizer_method='newton'.",
        "        optimizer_method: Optimizer used for variational updates. Supported\n            options are 'gradient', 'adam', and 'newton'.\n        optimizer_config: Method-specific optimizer config. Expected types are\n            `VIGradientOptimizerConfig` for optimizer_method='gradient',\n            `VIAdamOptimizerConfig` for optimizer_method='adam', and\n            `VINewtonOptimizerConfig` for optimizer_method='newton'.",
        "MFVI run docs",
    )
    text = replace_once(
        text,
        "        VIGradientOptimizerConfig(),\n        VINewtonOptimizerConfig(newton_regularization=1e-8),\n    )\n    line_search_method, resolved_line_search_config = _resolve_line_search_config(",
        "        VIGradientOptimizerConfig(),\n        VINewtonOptimizerConfig(newton_regularization=1e-8),\n        VIAdamOptimizerConfig(),\n    )\n    line_search_method, resolved_line_search_config = _resolve_line_search_config(",
        "MFVI optimizer resolver",
    )
    text = replace_once(
        text,
        "    gradient_method = 'standard'\n    if optimization_method == 'gradient':\n        gradient_method = resolved_optimizer_config.gradient_method",
        "    if optimization_method == 'adam':\n        line_search_method = 'legacy'\n        resolved_line_search_config = VILegacyLineSearchConfig(\n            initial_step_size=1.0,\n            max_step_size=1.0,\n            step_size_growth_factor=1.0,\n            step_size_decay_factor=1.0,\n            max_step_size_decrease_trys=0,\n            relaxation_parameter=1.0,\n            line_search_sample_growth_factor=1.0,\n            log_std_learning_rate_factor=1.0,\n        )\n        if restart_file is not None:\n            warnings.warn(\n                \"Adam optimizer moments are not stored in MF-VI restart files; \"\n                \"restarting resets the Adam first- and second-moment state.\",\n                RuntimeWarning,\n                stacklevel=2,\n            )\n\n    gradient_method = 'standard'\n    if optimization_method in ('gradient', 'adam'):\n        gradient_method = resolved_optimizer_config.gradient_method",
        "MFVI Adam setup",
    )
    text = replace_once(
        text,
        "    steepest_descent_solver = SteepestDescentSolver()",
        "    steepest_descent_solver = (\n        AdamSolver.from_config(resolved_optimizer_config)\n        if optimization_method == 'adam'\n        else SteepestDescentSolver()\n    )",
        "MFVI solver construction",
    )
    text = replace_once(
        text,
        "        if optimization_method == 'gradient':\n            gradient = np.concatenate([state['update_direction_mean'], state['update_direction_log_std']])",
        "        if optimization_method in ('gradient', 'adam'):\n            gradient = np.concatenate([state['update_direction_mean'], state['update_direction_log_std']])",
        "MFVI gradient branch",
    )
    text = replace_once(
        text,
        "        if line_search_method == 'legacy':\n            allowable_elbo_drop = (relaxation_parameter - 1.0) * abs(state['elbo'])",
        "        if optimization_method == 'adam':\n            accept_step = True\n        elif line_search_method == 'legacy':\n            allowable_elbo_drop = (relaxation_parameter - 1.0) * abs(state['elbo'])",
        "MFVI Adam acceptance",
    )
    path.write_text(text)


patch_vi(Path("romtools/workflows/inverse/vi_drivers.py"))
patch_mf_vi(Path("romtools/workflows/inverse/mf_vi_drivers.py"))
