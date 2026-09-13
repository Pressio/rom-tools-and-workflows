from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_optimizer(path):
    text = path.read_text()
    text = replace_once(
        text,
        "    By default Adam is applied to the natural gradient. For the Gaussian\n"
        "    variational parameterization used by the VI drivers this corresponds to\n"
        "    preconditioning the mean gradient by ``std**2`` and the log-standard-\n"
        "    deviation gradient by ``1/2``. When ``learning_rate`` is ``None``, the\n"
        "    ABRIS convention ``0.1 / parameter_dimension`` is used.\n",
        "    By default Adam is applied to the mean-field Gaussian natural gradient.\n"
        "    Following the ABRIS setup, the Fisher matrix is damped early in the\n"
        "    optimization and the preconditioned gradient is clipped by its L2 norm.\n"
        "    When ``learning_rate`` is ``None``, the ABRIS convention\n"
        "    ``0.1 / parameter_dimension`` is used.\n",
        "Adam config docs",
    )
    text = replace_once(
        text,
        "    epsilon: float = 1e-8\n"
        "    gradient_norm_tolerance: float = 1e-5\n",
        "    epsilon: float = 1e-8\n"
        "    fisher_damping_initial: float = 1e-2\n"
        "    fisher_damping_decay_start: int = 50\n"
        "    fisher_damping_min: float = 1e-6\n"
        "    gradient_clip_norm: float = 1e6\n"
        "    gradient_norm_tolerance: float = 1e-5\n",
        "Adam config fields",
    )
    text = replace_once(
        text,
        "                 beta2: float = 0.999,\n"
        "                 epsilon: float = 1e-8,\n"
        "                 parameter_dimension: int = None):\n",
        "                 beta2: float = 0.999,\n"
        "                 epsilon: float = 1e-8,\n"
        "                 fisher_damping_initial: float = 1e-2,\n"
        "                 fisher_damping_decay_start: int = 50,\n"
        "                 fisher_damping_min: float = 1e-6,\n"
        "                 gradient_clip_norm: float = 1e6,\n"
        "                 parameter_dimension: int = None):\n",
        "Adam constructor signature",
    )
    text = replace_once(
        text,
        "        if epsilon <= 0.0:\n"
        "            raise ValueError(\"epsilon must be positive.\")\n"
        "        if parameter_dimension is not None and parameter_dimension < 1:\n",
        "        if epsilon <= 0.0:\n"
        "            raise ValueError(\"epsilon must be positive.\")\n"
        "        if fisher_damping_initial < 0.0:\n"
        "            raise ValueError(\"fisher_damping_initial must be nonnegative.\")\n"
        "        if fisher_damping_decay_start < 1:\n"
        "            raise ValueError(\"fisher_damping_decay_start must be positive.\")\n"
        "        if fisher_damping_min < 0.0:\n"
        "            raise ValueError(\"fisher_damping_min must be nonnegative.\")\n"
        "        if fisher_damping_min > fisher_damping_initial:\n"
        "            raise ValueError(\n"
        "                \"fisher_damping_min cannot exceed fisher_damping_initial.\"\n"
        "            )\n"
        "        if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:\n"
        "            raise ValueError(\"gradient_clip_norm must be positive or None.\")\n"
        "        if parameter_dimension is not None and parameter_dimension < 1:\n",
        "Adam validation",
    )
    text = replace_once(
        text,
        "        self.epsilon = epsilon\n"
        "        self.parameter_dimension = parameter_dimension\n",
        "        self.epsilon = epsilon\n"
        "        self.fisher_damping_initial = fisher_damping_initial\n"
        "        self.fisher_damping_decay_start = int(fisher_damping_decay_start)\n"
        "        self.fisher_damping_min = fisher_damping_min\n"
        "        self.gradient_clip_norm = gradient_clip_norm\n"
        "        self.parameter_dimension = parameter_dimension\n",
        "Adam fields",
    )
    text = replace_once(
        text,
        "            beta2=config.beta2,\n"
        "            epsilon=config.epsilon,\n"
        "        )\n",
        "            beta2=config.beta2,\n"
        "            epsilon=config.epsilon,\n"
        "            fisher_damping_initial=config.fisher_damping_initial,\n"
        "            fisher_damping_decay_start=config.fisher_damping_decay_start,\n"
        "            fisher_damping_min=config.fisher_damping_min,\n"
        "            gradient_clip_norm=config.gradient_clip_norm,\n"
        "        )\n",
        "Adam from config",
    )
    text = replace_once(
        text,
        "    def step(self, gradient: np.ndarray) -> np.ndarray:\n"
        "        gradient = np.asarray(gradient, dtype=float)\n"
        "        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)\n"
        "        if self.first_moment is None:\n",
        "    def _current_fisher_damping(self) -> float:\n"
        "        if self.iteration < self.fisher_damping_decay_start:\n"
        "            return self.fisher_damping_initial\n"
        "        decay_exponent = -(\n"
        "            self.iteration - self.fisher_damping_decay_start\n"
        "        ) / self.fisher_damping_decay_start\n"
        "        return max(\n"
        "            self.fisher_damping_initial * np.exp(decay_exponent),\n"
        "            self.fisher_damping_min,\n"
        "        )\n\n"
        "    def _prepare_gradient(self, gradient: np.ndarray, fisher_diagonal=None) -> np.ndarray:\n"
        "        gradient = np.asarray(gradient, dtype=float)\n"
        "        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)\n"
        "        if fisher_diagonal is not None:\n"
        "            fisher_diagonal = np.asarray(fisher_diagonal, dtype=float)\n"
        "            if fisher_diagonal.shape != gradient.shape:\n"
        "                raise ValueError(\"Fisher diagonal must have the same shape as gradient.\")\n"
        "            if np.any(fisher_diagonal < 0.0):\n"
        "                raise ValueError(\"Fisher diagonal entries must be nonnegative.\")\n"
        "            gradient = gradient / (\n"
        "                fisher_diagonal + self._current_fisher_damping()\n"
        "            )\n"
        "        if self.gradient_clip_norm is not None:\n"
        "            gradient_norm = np.linalg.norm(gradient)\n"
        "            if gradient_norm > self.gradient_clip_norm:\n"
        "                gradient = gradient * (self.gradient_clip_norm / gradient_norm)\n"
        "        return gradient\n\n"
        "    def step(self, gradient: np.ndarray, fisher_diagonal=None) -> np.ndarray:\n"
        "        gradient = self._prepare_gradient(gradient, fisher_diagonal=fisher_diagonal)\n"
        "        if self.first_moment is None:\n",
        "Adam preconditioning",
    )
    path.write_text(text)


def patch_driver(path, is_mf=False):
    text = path.read_text()
    old_docs = (
        "When ``optimizer_method=\"adam\"``, the selected score gradient (standard or\n"
        "natural/Fisher-preconditioned) is passed to a stateful Adam ascent update.\n"
        "The Adam default uses the natural gradient and the ABRIS initial learning-rate\n"
        "convention :math:`0.1/d`, where :math:`d` is the parameter dimension.\n"
    )
    new_docs = (
        "When ``optimizer_method=\"adam\"``, the selected score gradient is passed to a\n"
        "stateful Adam ascent update. By default, the raw score gradient is\n"
        "preconditioned with the damped mean-field Gaussian Fisher matrix before Adam,\n"
        "matching the ABRIS setup; the default initial learning rate is :math:`0.1/d`,\n"
        "where :math:`d` is the parameter dimension.\n"
    )
    if not is_mf:
        text = replace_once(text, old_docs, new_docs, "VI Adam docs")

    old = (
        "        if optimization_method in ('gradient', 'adam'):\n"
        "            gradient = np.concatenate([state['update_direction_mean'], state['update_direction_log_std']])\n"
        "            step = steepest_descent_solver.step(gradient)\n"
        "            dimensionality = state['update_direction_mean'].size\n"
    )
    new = (
        "        if optimization_method in ('gradient', 'adam'):\n"
        "            if optimization_method == 'adam':\n"
        "                gradient = np.concatenate([state['gradient_mean'], state['gradient_log_std']])\n"
        "                fisher_diagonal = None\n"
        "                if gradient_method == 'natural':\n"
        "                    variational_std_for_fisher, _ = _compute_variational_std(\n"
        "                        variational_log_std,\n"
        "                        min_variational_std,\n"
        "                        max_variational_std,\n"
        "                    )\n"
        "                    fisher_diagonal = np.concatenate([\n"
        "                        1.0 / (variational_std_for_fisher ** 2),\n"
        "                        2.0 * np.ones_like(variational_std_for_fisher),\n"
        "                    ])\n"
        "                step = steepest_descent_solver.step(\n"
        "                    gradient, fisher_diagonal=fisher_diagonal\n"
        "                )\n"
        "            else:\n"
        "                gradient = np.concatenate([\n"
        "                    state['update_direction_mean'], state['update_direction_log_std']\n"
        "                ])\n"
        "                step = steepest_descent_solver.step(gradient)\n"
        "            dimensionality = state['update_direction_mean'].size\n"
    )
    text = replace_once(text, old, new, "MFVI Adam gradient" if is_mf else "VI Adam gradient")
    path.write_text(text)


def patch_tests(path):
    text = path.read_text()
    insert_after = (
        "def test_adam_solver_accumulates_bias_corrected_moments():\n"
        "    solver = AdamSolver(learning_rate=0.02, beta1=0.8, beta2=0.9, epsilon=1e-12)\n"
        "    first_gradient = np.array([1.0, -2.0])\n"
        "    second_gradient = np.array([3.0, 4.0])\n\n"
        "    solver.step(first_gradient)\n"
        "    second_step = solver.step(second_gradient)\n\n"
        "    first_moment = 0.8 * (0.2 * first_gradient) + 0.2 * second_gradient\n"
        "    second_moment = 0.9 * (0.1 * first_gradient ** 2) + 0.1 * second_gradient ** 2\n"
        "    first_moment_hat = first_moment / (1.0 - 0.8 ** 2)\n"
        "    second_moment_hat = second_moment / (1.0 - 0.9 ** 2)\n"
        "    expected = 0.02 * first_moment_hat / np.sqrt(second_moment_hat)\n\n"
        "    np.testing.assert_allclose(second_step, expected)\n\n\n"
    )
    addition = (
        "def test_adam_solver_applies_abris_fisher_damping_before_moments():\n"
        "    solver = AdamSolver(learning_rate=0.01)\n"
        "    gradient = np.array([4.0, 6.0])\n"
        "    fisher_diagonal = np.array([2.0, 3.0])\n\n"
        "    solver.step(gradient, fisher_diagonal=fisher_diagonal)\n\n"
        "    expected_preconditioned = gradient / (fisher_diagonal + 1e-2)\n"
        "    np.testing.assert_allclose(\n"
        "        solver.first_moment,\n"
        "        (1.0 - solver.beta1) * expected_preconditioned,\n"
        "    )\n\n\n"
        "def test_adam_solver_clips_preconditioned_gradient_norm():\n"
        "    solver = AdamSolver(learning_rate=0.01, gradient_clip_norm=5.0)\n"
        "    gradient = np.array([30.0, 40.0])\n\n"
        "    solver.step(gradient)\n\n"
        "    prepared_gradient = solver.first_moment / (1.0 - solver.beta1)\n"
        "    assert np.isclose(np.linalg.norm(prepared_gradient), 5.0)\n\n\n"
    )
    text = replace_once(text, insert_after, insert_after + addition, "Adam stabilization tests")
    path.write_text(text)


patch_optimizer(Path("romtools/workflows/inverse/vi_optimization_methods.py"))
patch_driver(Path("romtools/workflows/inverse/vi_drivers.py"), is_mf=False)
patch_driver(Path("romtools/workflows/inverse/mf_vi_drivers.py"), is_mf=True)
patch_tests(Path("tests/romtools/workflows/inverse/test_vi_adam.py"))
