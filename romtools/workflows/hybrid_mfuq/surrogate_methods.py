"""
Surrogate Methods for Multi-Fidelity UQ

Builds surrogate models for correlation and cost functions using:
- Neural network approach (VeclNet) for valid correlation matrices
- Sigmoid fitting for smooth monotonic functions
- Polynomial fitting for simple trends

All surrogates are PyTorch-differentiable for backpropagation through s.

As of the T6 refactor (see hybrid_mfuq_simplification_plan.md), the
curve-fitting helpers live in `surrogate_fitting.py` and the batched
Archakov--Hansen matrix machinery lives in `archakov_hansen.py`. This
file keeps `SurrogateBuilder`, which orchestrates both, and re-exports
the split-out names so `from ...surrogate_methods import X` keeps working
for every X that was previously importable from here.
"""

import os

import numpy as np
import torch

from romtools.workflows.hybrid_mfuq.surrogate_fitting import (
    fit_polynomial,
    fit_sigmoid,
    fit_polynomial_torch,
    fit_sigmoid_torch,
)
from romtools.workflows.hybrid_mfuq.archakov_hansen import (
    to_symmetric_tracefree_batch,
    WarmStartedArchakovHansenMap,
    to_unique_corr_matrix_batch,
    VeclNet,
    train_ah_matrix_model,
    AHMatrixCorrelationSurrogate,
)


class SurrogateBuilder:
    """
    Builds cost and correlation surrogates.

    Supported correlation surrogate modes
    -------------------------------------

    ah_matrix:
        New default. Trains a matrix-valued AH surrogate and returns
        corr_matrix_fn(s). This preserves global admissibility of the returned
        correlation matrix.

    ah_componentwise_sigmoid:
        Legacy current behavior. Trains an AH/VeclNet matrix surrogate, samples
        its entries on a dense grid, then fits componentwise scalar surrogates.
        This does not preserve the global matrix guarantee after scalar refit.

    componentwise_sigmoid:
        Direct scalar sigmoid/polynomial fitting to pilot correlations and
        costs. Does not use AH.
    """

    def __init__(
        self,
        pilot_list,
        n_active,
        n_aux,
        work_dir=None,
        method="ah_matrix",
        tunable_range=None,
        use_torch=True,
        fixed_fixed_weight=1.0,
        ah_tol=1e-8,
        ah_max_iter=1000,
        ah_psd_check="none",
        ah_fixed_fixed_tol=0.05,
    ):
        self.pilot_list = list(pilot_list)
        self.n_active = n_active
        self.n_aux = n_aux
        self.n_models = 1 + n_aux + n_active
        self.work_dir = work_dir
        self.use_torch = use_torch
        self.tunable_range = tunable_range or [min(pilot_list), max(pilot_list)]
        self.fixed_fixed_weight = fixed_fixed_weight
        self.ah_tol = ah_tol
        self.ah_max_iter = ah_max_iter
        self.ah_psd_check = ah_psd_check
        self.ah_fixed_fixed_tol = ah_fixed_fixed_tol

        # Backward-compatible aliases.
        if method == "neural_network":
            method = "ah_componentwise_sigmoid"
        elif method == "sigmoid":
            method = "componentwise_sigmoid"

        valid_methods = {
            "ah_matrix",
            "ah_componentwise_sigmoid",
            "componentwise_sigmoid",
        }

        if method not in valid_methods:
            raise ValueError(
                f"Unknown surrogate method '{method}'. "
                f"Valid methods are {sorted(valid_methods)}."
            )

        self.method = method

        if work_dir:
            self.model_path = os.path.join(
                work_dir,
                f"vecl_correlation_model_{self.method}.pt",
            )
        else:
            self.model_path = None

    # ------------------------------------------------------------------
    # Public build
    # ------------------------------------------------------------------

    def build(self, data_npz):
        """
        Returns
        -------
        hf_corr_list, lf_corr_list, cost_list, corr_matrix_fn

        For scalar modes, corr_matrix_fn is None.

        For ah_matrix, hf_corr_list and lf_corr_list are None and the optimizer
        should use corr_matrix_fn.
        """
        with np.load(data_npz) as data:
            fom_aux_corrs = data["fom_aux_corrs"]
            aux_aux_corrs = data.get("aux_aux_corrs", np.array([]))
            fom_rom_corrs = data["fom_rom_corrs"]
            aux_rom_corrs_list = [
                data[f"aux{i}_rom_corrs"] for i in range(self.n_aux)
            ]
            norm_aux_times = data["normalized_aux_times"]
            norm_rom_times = data["normalized_rom_times"]

        if self.method == "componentwise_sigmoid":
            print("Building direct componentwise sigmoid surrogates")
            return self._build_componentwise_sigmoid(
                fom_aux_corrs,
                aux_aux_corrs,
                fom_rom_corrs,
                aux_rom_corrs_list,
                norm_aux_times,
                norm_rom_times,
            )

        if self.method == "ah_componentwise_sigmoid":
            print("Building legacy AH + componentwise sigmoid surrogates")
            return self._build_ah_componentwise_sigmoid(
                fom_aux_corrs,
                aux_aux_corrs,
                fom_rom_corrs,
                aux_rom_corrs_list,
                norm_aux_times,
                norm_rom_times,
            )

        print("Building matrix-valued Archakov--Hansen surrogate")
        return self._build_ah_matrix(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
            norm_aux_times,
            norm_rom_times,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _rom_input(self, s):
        """Extract ROM coordinate from state vector or pass through scalar."""
        if torch.is_tensor(s):
            return s if s.ndim == 0 else s[-1]
        return s if np.isscalar(s) else s[-1]

    def _wrap(self, func):
        """Wrap scalar surrogate to handle vector inputs."""
        def wrapped(s):
            s_input = self._rom_input(s)
            result = func(s_input)

            if self.use_torch and torch.is_tensor(s):
                if not torch.is_tensor(result):
                    result = torch.tensor(
                        result,
                        dtype=torch.float64,
                        device=s.device,
                    )

            return result

        return wrapped

    def _make_constant(self, value):
        """Create a constant function compatible with numpy and torch."""
        if self.use_torch:
            def const_fn(s):
                if torch.is_tensor(s):
                    return torch.tensor(
                        value,
                        dtype=torch.float64,
                        device=s.device,
                    )
                return value

            return const_fn

        return lambda s: value

    def _build_cost_list(self, norm_aux_times, norm_rom_times):
        if self.use_torch:
            cost_rom_surr = fit_polynomial_torch(
                np.array(self.pilot_list)[None, :],
                norm_rom_times,
                order=1,
            )
        else:
            cost_rom_surr = fit_polynomial(
                np.array(self.pilot_list)[None, :],
                norm_rom_times,
                order=1,
            )

        cost_list = [self._make_constant(float(t)) for t in norm_aux_times]
        cost_list.append(self._wrap(cost_rom_surr))

        return cost_list

    # ------------------------------------------------------------------
    # Direct scalar path
    # ------------------------------------------------------------------

    def _build_componentwise_sigmoid(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        norm_aux_times,
        norm_rom_times,
    ):
        pilots = np.array(self.pilot_list)

        if self.use_torch:
            fit_sig = fit_sigmoid_torch
        else:
            fit_sig = fit_sigmoid

        fom_rom_surr = fit_sig(pilots[None, :], fom_rom_corrs)
        aux_rom_surrs = [
            fit_sig(pilots[None, :], corrs) for corrs in aux_rom_corrs_list
        ]

        hf_corr_list = [self._make_constant(float(c)) for c in fom_aux_corrs]
        hf_corr_list.append(self._wrap(fom_rom_surr))

        lf_corr_list = [self._make_constant(float(c)) for c in aux_aux_corrs]
        lf_corr_list.extend([self._wrap(surr) for surr in aux_rom_surrs])

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times)

        return hf_corr_list, lf_corr_list, cost_list, None

    # ------------------------------------------------------------------
    # AH matrix path
    # ------------------------------------------------------------------

    def _build_ah_matrix(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        norm_aux_times,
        norm_rom_times,
    ):
        n = self.n_models
        hidden_size = 4

        model, ah_map = self._load_or_train_ah_matrix_model(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
            n,
            hidden_size,
        )

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times)

        corr_matrix_fn = AHMatrixCorrelationSurrogate(
            omega_model=model,
            ah_map=ah_map,
            s_min=self.tunable_range[0],
            s_max=self.tunable_range[1],
            psd_check=self.ah_psd_check,
        )

        return None, None, cost_list, corr_matrix_fn

    def _load_or_train_ah_matrix_model(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        n,
        hidden_size,
    ):
        # Needed both to validate a cached model and, if training is
        # required, to hand off to _train_ah_matrix_model.
        inputs = torch.tensor(self.pilot_list, dtype=torch.float64).reshape(-1, 1)
        inputs_norm = self._normalize_training_inputs(inputs)

        targets, weights = self._assemble_ah_targets_and_weights(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
        )

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location="cpu")

            compatible = (
                checkpoint.get("pilot_list") == self.pilot_list
                and checkpoint.get("n_active") == self.n_active
                and checkpoint.get("n_aux") == self.n_aux
                and checkpoint.get("method") == self.method
            )

            if compatible:
                model = VeclNet(1, hidden_size, n).double()
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                ah_map = WarmStartedArchakovHansenMap(
                    n,
                    tol=self.ah_tol,
                    max_iter=self.ah_max_iter,
                )

                # A cached checkpoint may predate this validate-and-retry
                # safeguard (e.g. saved by an earlier, unstable training
                # run). Loading it unconditionally would silently bypass
                # retraining forever, since the file looks "compatible" on
                # every subsequent run. Validate it exactly as a freshly
                # trained model would be validated.
                err = self._fixed_fixed_fit_error(
                    model, ah_map, inputs_norm, targets, weights
                )

                if err <= self.ah_fixed_fixed_tol:
                    print(
                        f"Loading AH matrix model from {self.model_path} "
                        f"(fixed-fixed fit error {err:.4f})\n"
                    )
                    return model, ah_map

                print(
                    f"Cached AH model at {self.model_path} fails fixed-fixed "
                    f"validation (error {err:.4f} > {self.ah_fixed_fixed_tol:.4f}); "
                    f"discarding and retraining\n"
                )
            else:
                print("Cached AH model incompatible; retraining\n")

            os.remove(self.model_path)

        return self._train_ah_matrix_model(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
            n,
            hidden_size,
        )

    def _train_ah_matrix_model(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        n,
        hidden_size,
        max_restarts=5,
        fixed_fixed_fit_tol=None,
    ):
        if fixed_fixed_fit_tol is None:
            fixed_fixed_fit_tol = self.ah_fixed_fixed_tol

        print(
            f"Training AH matrix surrogate: "
            f"n={n} (1 FOM + {self.n_aux} aux + {self.n_active} ROM)\n"
        )

        inputs = torch.tensor(
            self.pilot_list,
            dtype=torch.float64,
        ).reshape(-1, 1)

        inputs_norm = self._normalize_training_inputs(inputs)

        targets, weights = self._assemble_ah_targets_and_weights(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
        )

        # The fixed-fixed entries (FOM-aux, aux-aux) are known exactly from the
        # pilot data and should not depend on s at all. Training can
        # occasionally overshoot into the saturated |corr| -> 1 region of the
        # AH map and get stuck there (a spurious but low-gradient point that
        # the delta-loss stopping rule misreads as convergence). Rather than
        # silently pass a corrupted surrogate downstream to the ACV
        # optimizer, validate the fixed-fixed fit and retry with a fresh
        # random initialization if it is off.
        best_model, best_ah_map, best_err = None, None, np.inf

        for attempt in range(1, max_restarts + 1):
            model = VeclNet(1, hidden_size, n).double()

            ah_map = WarmStartedArchakovHansenMap(
                n,
                tol=self.ah_tol,
                max_iter=self.ah_max_iter,
            )

            model, _ = train_ah_matrix_model(
                model,
                ah_map,
                inputs_norm,
                targets,
                weights,
                lr=1e-2,
                max_steps=2000,
                tol=1e-9,
                grad_clip=1.0,
                print_every=50,
            )

            model.eval()

            err = self._fixed_fixed_fit_error(
                model, ah_map, inputs_norm, targets, weights
            )

            if err < best_err:
                best_model, best_ah_map, best_err = model, ah_map, err

            if err <= fixed_fixed_fit_tol:
                break

            print(
                f"AH matrix attempt {attempt}/{max_restarts}: "
                f"fixed-fixed fit error {err:.4f} exceeds tolerance "
                f"{fixed_fixed_fit_tol:.4f}; retrying with a new init\n"
            )
        else:
            print(
                f"Warning: AH matrix surrogate did not reach the fixed-fixed "
                f"fit tolerance after {max_restarts} attempts "
                f"(best error {best_err:.4f}). Using the best fit found.\n"
            )

        model, ah_map = best_model, best_ah_map

        if self.model_path:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pilot_list": self.pilot_list,
                    "n_active": self.n_active,
                    "n_aux": self.n_aux,
                    "n_models": n,
                    "hidden_size": hidden_size,
                    "method": self.method,
                    "fixed_fixed_weight": self.fixed_fixed_weight,
                },
                self.model_path,
            )

            print(f"Saved AH matrix model to {self.model_path}\n")

        return model, ah_map

    def _normalize_training_inputs(self, inputs):
        s_min, s_max = self.tunable_range
        return 2.0 * (inputs - s_min) / (s_max - s_min) - 1.0

    def _fixed_fixed_fit_error(self, model, ah_map, inputs_norm, targets, weights):
        """
        Max absolute error between the trained surrogate and the known
        fixed-fixed correlation targets (entries carrying fixed_fixed_weight).
        These entries do not depend on s, so a good fit should match them
        closely at every pilot point; a large error signals the training run
        landed in a saturated, low-gradient region of the AH map.
        """
        with torch.no_grad():
            ah_map.reset_cache()
            P_pred = ah_map(model(inputs_norm))

        mask = weights >= (0.5 * self.fixed_fixed_weight)

        if not torch.any(mask):
            return 0.0

        return float(torch.max(torch.abs(P_pred[mask] - targets[mask])))

    def _assemble_ah_targets_and_weights(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
    ):
        """
        Assemble target correlation matrices and matrix-entry weights.

        Current implementation supports one active ROM, matching the existing
        workflow. The layout is:

            index 0: FOM
            indices 1..n_aux: fixed auxiliary models
            final index: trainable ROM

        Fixed-fixed entries receive fixed_fixed_weight.
        ROM-dependent entries receive weight 1.
        """
        if self.n_active != 1:
            raise NotImplementedError(
                "The current AH target assembly supports one active ROM. "
                "Extend this function for multiple trainable ROMs."
            )

        pilots = np.array(self.pilot_list)
        G = len(pilots)

        n = self.n_models
        n_fixed = 1 + self.n_aux
        rom_idx = n - 1

        targets = np.zeros((G, n, n), dtype=float)
        weights = np.zeros((G, n, n), dtype=float)

        for g in range(G):
            P = np.eye(n, dtype=float)
            W = np.zeros((n, n), dtype=float)

            # FOM-aux fixed correlations.
            for i in range(self.n_aux):
                idx = i + 1
                P[idx, 0] = fom_aux_corrs[i]
                P[0, idx] = fom_aux_corrs[i]

                W[idx, 0] = self.fixed_fixed_weight
                W[0, idx] = self.fixed_fixed_weight

            # Aux-aux fixed correlations.
            aux_pair_idx = 0
            for i in range(self.n_aux):
                for j in range(i):
                    idx_i = i + 1
                    idx_j = j + 1

                    corr = aux_aux_corrs[aux_pair_idx]
                    P[idx_i, idx_j] = corr
                    P[idx_j, idx_i] = corr

                    W[idx_i, idx_j] = self.fixed_fixed_weight
                    W[idx_j, idx_i] = self.fixed_fixed_weight

                    aux_pair_idx += 1

            # FOM-ROM varying correlation.
            P[rom_idx, 0] = fom_rom_corrs[g]
            P[0, rom_idx] = fom_rom_corrs[g]

            W[rom_idx, 0] = 1.0
            W[0, rom_idx] = 1.0

            # Aux-ROM varying correlations.
            for i in range(self.n_aux):
                idx = i + 1
                corr = aux_rom_corrs_list[i][g]

                P[rom_idx, idx] = corr
                P[idx, rom_idx] = corr

                W[rom_idx, idx] = 1.0
                W[idx, rom_idx] = 1.0

            # Diagonal is guaranteed by AH, but a small weight can improve
            # numerical behavior. Keep this modest.
            for i in range(n):
                W[i, i] = 1.0

            targets[g] = P
            weights[g] = W

        return (
            torch.tensor(targets, dtype=torch.float64),
            torch.tensor(weights, dtype=torch.float64),
        )

    # ------------------------------------------------------------------
    # Legacy AH + componentwise sigmoid path
    # ------------------------------------------------------------------

    def _build_ah_componentwise_sigmoid(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs,
        aux_rom_corrs_list,
        norm_aux_times,
        norm_rom_times,
    ):
        n = self.n_models
        hidden_size = 4

        # Reuse the AH matrix training routine, but then intentionally discard
        # the matrix-valued surrogate by fitting scalar surrogates to entries.
        model, ah_map = self._load_or_train_ah_matrix_model(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs,
            aux_rom_corrs_list,
            n,
            hidden_size,
        )

        hf_corr_list, lf_corr_list = self._fit_surrogates_to_model(
            model,
            ah_map,
            n,
        )

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times)

        return hf_corr_list, lf_corr_list, cost_list, None

    def _fit_surrogates_to_model(self, model, ah_map, n):
        """
        Legacy behavior: query AH matrix model on a dense grid and fit
        componentwise scalar surrogates. This path does not preserve global
        admissibility after the scalar refit.
        """
        s_grid = np.unique(
            np.concatenate(
                [
                    self.pilot_list,
                    np.linspace(
                        self.tunable_range[0],
                        self.tunable_range[1],
                        200,
                    ),
                ]
            )
        )

        s_tensor = torch.tensor(s_grid, dtype=torch.float64).reshape(-1, 1)
        s_tensor_norm = self._normalize_training_inputs(s_tensor)

        with torch.no_grad():
            ah_map.reset_cache()
            corr_matrices = model.corr_matrix(s_tensor_norm, ah_map).cpu().numpy()

        if self.work_dir:
            self._plot_correlations(corr_matrices, s_grid, n)

        if self.use_torch:
            surrogates = self._fit_torch_surrogates(corr_matrices, s_grid, n)
        else:
            surrogates = self._fit_numpy_surrogates(corr_matrices, s_grid, n)

        hf_corr_list = [surrogates[(i, 0)] for i in range(1, n)]
        lf_corr_list = [
            surrogates[(i, j)]
            for i in range(1, n)
            for j in range(1, i)
        ]

        return hf_corr_list, lf_corr_list

    def _fit_torch_surrogates(self, corr_matrices, s_grid, n):
        surrogates = {}

        for i in range(n):
            for j in range(i):
                values = corr_matrices[:, i, j]

                if np.std(values) < 0.01:
                    surrogates[(i, j)] = self._make_constant(float(np.mean(values)))
                else:
                    sig = fit_sigmoid_torch(s_grid[None, :], values)
                    surrogates[(i, j)] = lambda s, f=sig: f(self._rom_input(s))

        return surrogates

    def _fit_numpy_surrogates(self, corr_matrices, s_grid, n):
        from scipy.interpolate import interp1d

        surrogates = {}

        for i in range(n):
            for j in range(i):
                values = corr_matrices[:, i, j]

                if np.std(values) < 0.01:
                    surrogates[(i, j)] = self._make_constant(float(np.mean(values)))
                else:
                    try:
                        sig = fit_sigmoid(s_grid[None, :], values)
                        test = np.array([sig(s) for s in s_grid])

                        if np.mean((test - values) ** 2) < 0.01:
                            surrogates[(i, j)] = (
                                lambda s, f=sig: float(f(self._rom_input(s)))
                            )
                        else:
                            raise ValueError("Poor sigmoid fit")

                    except Exception:
                        interp = interp1d(
                            s_grid,
                            values,
                            kind="cubic",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        surrogates[(i, j)] = (
                            lambda s, f=interp: float(f(self._rom_input(s)))
                        )

        return surrogates

    def _plot_correlations(self, corr_matrices, s_grid, n):
        """
        Keep your existing diagnostic plotting implementation here.

        The body of your current _plot_correlations method can remain mostly
        unchanged. It consumes corr_matrices, s_grid, and n in the same way.
        """
        try:
            import matplotlib.pyplot as plt

            debug_dir = os.path.join(self.work_dir, "debug_plots")
            os.makedirs(debug_dir, exist_ok=True)

            with np.load(os.path.join(self.work_dir, "pilot_results.npz")) as data:
                fom_aux = data["fom_aux_corrs"]
                aux_aux = data.get("aux_aux_corrs", np.array([]))
                fom_rom = data["fom_rom_corrs"]
                aux_rom = [data[f"aux{i}_rom_corrs"] for i in range(self.n_aux)]

            pilot_data, names = {}, {}

            for i in range(self.n_aux):
                pilot_data[(i + 1, 0)] = np.full(len(self.pilot_list), fom_aux[i])
                names[(i + 1, 0)] = f"FOM-aux{i}"

            if self.n_aux > 1:
                idx = 0
                for i in range(self.n_aux):
                    for j in range(i):
                        pilot_data[(i + 1, j + 1)] = np.full(
                            len(self.pilot_list),
                            aux_aux[idx],
                        )
                        names[(i + 1, j + 1)] = f"aux{j}-aux{i}"
                        idx += 1

            rom_idx = n - 1
            pilot_data[(rom_idx, 0)] = fom_rom
            names[(rom_idx, 0)] = "FOM-ROM"

            for i in range(self.n_aux):
                pilot_data[(rom_idx, i + 1)] = aux_rom[i]
                names[(rom_idx, i + 1)] = f"aux{i}-ROM"

            n_plots = n * (n - 1) // 2
            ncols = min(3, n_plots)
            nrows = (n_plots + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6 * ncols, 5 * nrows),
            )

            axes = np.array([axes]).flatten() if n_plots == 1 else axes.flatten()

            plot_idx = 0
            for i in range(n):
                for j in range(i):
                    ax = axes[plot_idx]
                    nn_vals = corr_matrices[:, i, j]

                    ax.plot(
                        s_grid,
                        nn_vals,
                        "b-",
                        label="AH surrogate",
                        linewidth=2,
                        alpha=0.7,
                    )

                    if (i, j) in pilot_data:
                        ax.plot(
                            self.pilot_list,
                            pilot_data[(i, j)],
                            "ro",
                            label="Pilot",
                            markersize=8,
                            zorder=5,
                        )

                    ax.set_title(
                        names.get((i, j), f"({i},{j})"),
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.set_xlabel("ROM basis size")
                    ax.set_ylabel("Correlation")
                    ax.set_ylim([-1.05, 1.05])
                    ax.axhline(0, color="k", linestyle=":", alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=9)

                    plot_idx += 1

            for idx in range(plot_idx, len(axes)):
                axes[idx].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(debug_dir, "correlation_fits.png"),
                dpi=150,
                bbox_inches="tight",
            )
            print(f"\nSaved plot to {debug_dir}/correlation_fits.png\n")
            plt.close()

        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")