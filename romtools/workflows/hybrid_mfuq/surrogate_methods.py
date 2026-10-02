"""
Cost and correlation surrogates for Multi-Fidelity UQ (writeup Sec. 4.3).

`SurrogateBuilder` turns the pilot statistics of Sec. 4.2 into the
differentiable surrogates the online ACV optimization queries: a normalized
cost vector w_hat(s) and a correlation matrix P(s).

The scalar curve fits live in `surrogate_fitting.py` and the Archakov--Hansen
matrix machinery in `archakov_hansen.py`; the split-out names are re-exported
here so existing imports keep working. Model ordering and pair flattening come
from `model_indices.py`, which also carries the writeup-to-code notation map.
"""

import os

import numpy as np
import torch

from romtools.workflows.hybrid_mfuq.model_indices import (
    aux_model_slot,
    aux_pairs,
    lf_pairs,
    matrix_pairs,
    n_models,
    rom_model_slot,
    rom_state_offset,
    tril_position,
)
from romtools.workflows.hybrid_mfuq.surrogate_fitting import (
    fit_cost_polynomial,
    fit_sigmoid,
)
from romtools.workflows.hybrid_mfuq.archakov_hansen import (
    to_symmetric_tracefree_batch,
    WarmStartedArchakovHansenMap,
    to_unique_corr_matrix_batch,
    VeclNet,
    train_ah_matrix_model,
    AHMatrixCorrelationSurrogate,
    insert_fixed_fixed_entries,
)


# Correlation-surrogate training settings. Named here rather than inline so
# they are discoverable and adjustable in one place.
AH_HIDDEN_SIZE = 4               # width of VeclNet's single hidden layer
AH_LEARNING_RATE = 1e-2
AH_MAX_STEPS = 10000
AH_LOSS_TOL = 1e-9               # delta-loss convergence threshold
AH_GRAD_CLIP = 1.0
AH_PRINT_EVERY = 50
AH_MAX_RESTARTS = 5              # fresh inits before giving up on the fit

# A pilot correlation whose spread across the basis grid is below this is
# treated as constant rather than fitted, since a sigmoid through flat data is
# unidentifiable.
CONSTANT_CORRELATION_STD = 0.01

# Mean squared error above which a fitted sigmoid is reported as a poor fit.
SIGMOID_FIT_MSE_WARN = 0.01

# Points used to sample a trained AH surrogate when refitting scalar
# surrogates to it (the ah_componentwise_sigmoid path).
AH_RESAMPLE_POINTS = 200


class SurrogateBuilder:
    """
    Builds cost and correlation surrogates.

    Supported correlation surrogate modes
    -------------------------------------

    ah_matrix_bandaid:
        Default. Same AH training as ah_matrix, but fixed-fixed entries of
        P(s) are hard-overwritten with their known pilot values after the AH
        map is applied (during training and at inference), instead of merely
        being up-weighted. This is the literal reading of writeup Algorithm 6
        line 20 ([P(s)]_{i,q} = p_hat_{i,q} for i, q in F): those entries do
        not depend on any basis size and are known exactly from the pilot, so
        they are imposed rather than regressed. It fixes extrapolation drift
        on those entries at the cost of the global PSD guarantee -- the
        overwrite is not a congruence, so the returned matrix is no longer
        admissible by construction. Set ah_psd_check='warn' (or 'raise') to
        monitor this during the ACV optimization.

    ah_matrix:
        Pure AH path: fixed-fixed entries are only up-weighted during
        training, never overwritten, so P(s) stays a valid correlation matrix
        for every s. Use this when global admissibility matters more than
        exactness of the fixed-fixed entries; note that the default
        fixed_fixed_weight=1.0 gives them no preference at all, so raise it
        (~1e3) if you select this mode.

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
        pilot_basis_grids,
        n_active,
        n_aux,
        work_dir=None,
        method="ah_matrix_bandaid",
        tunable_ranges=None,
        use_torch=True,
        fixed_fixed_weight=1.0,
        ah_tol=1e-8,
        ah_max_iter=1000,
        ah_psd_check="none",
        ah_fixed_fixed_tol=0.05,
        cost_poly_order=1,
    ):
        # pilot_basis_grids: List[List[int]], one basis-size grid B_t per
        # trainable ROM t (t in rom_model_builders order).
        self.pilot_basis_grids = [list(g) for g in pilot_basis_grids]
        self.n_active = n_active
        self.n_aux = n_aux
        self.n_models = n_models(n_aux, n_active)
        self.work_dir = work_dir
        self.use_torch = use_torch
        # tunable_ranges: List[[smin, smax]], one bound pair per ROM t.
        self.tunable_ranges = tunable_ranges or [
            [min(g), max(g)] for g in self.pilot_basis_grids
        ]
        self.fixed_fixed_weight = fixed_fixed_weight
        self.ah_tol = ah_tol
        self.ah_max_iter = ah_max_iter
        self.ah_psd_check = ah_psd_check
        self.ah_fixed_fixed_tol = ah_fixed_fixed_tol
        # Polynomial order r of the trainable-ROM cost surrogates
        # (writeup Sec. 4.3.1; r = 1 in the reported experiments).
        self.cost_poly_order = cost_poly_order

        # Backward-compatible aliases.
        if method == "neural_network":
            method = "ah_componentwise_sigmoid"
        elif method == "sigmoid":
            method = "componentwise_sigmoid"

        valid_methods = {
            "ah_matrix",
            "ah_matrix_bandaid",
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

            # Per-ROM keys; see PilotData.to_npz_dict for the schema.
            fom_rom_corrs_list = [
                data[f"rom{t}_fom_corrs"] for t in range(self.n_active)
            ]
            aux_rom_corrs_list = [
                [data[f"aux{i}_rom{t}_corrs"] for t in range(self.n_active)]
                for i in range(self.n_aux)
            ]
            norm_rom_times_list = [
                data[f"rom{t}_normalized_times"] for t in range(self.n_active)
            ]
            norm_aux_times = data["normalized_aux_times"]

            rom_rom_corrs = []
            if "rom_rom_pairs" in data:
                for t, q in data["rom_rom_pairs"]:
                    t, q = int(t), int(q)
                    rom_rom_corrs.append((
                        t,
                        q,
                        data[f"rom{t}_rom{q}_s_t"],
                        data[f"rom{t}_rom{q}_s_q"],
                        data[f"rom{t}_rom{q}_corrs"],
                    ))

        if self.method in ("componentwise_sigmoid", "ah_componentwise_sigmoid"):
            if self.method == "componentwise_sigmoid":
                print("Building direct componentwise sigmoid surrogates")
                return self._build_componentwise_sigmoid(
                    fom_aux_corrs,
                    aux_aux_corrs,
                    fom_rom_corrs_list,
                    aux_rom_corrs_list,
                    rom_rom_corrs,
                    norm_aux_times,
                    norm_rom_times_list,
                )

            print("Building legacy AH + componentwise sigmoid surrogates")
            return self._build_ah_componentwise_sigmoid(
                fom_aux_corrs,
                aux_aux_corrs,
                fom_rom_corrs_list,
                aux_rom_corrs_list,
                rom_rom_corrs,
                norm_aux_times,
                norm_rom_times_list,
            )

        print("Building matrix-valued Archakov--Hansen surrogate")
        return self._build_ah_matrix(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs_list,
            aux_rom_corrs_list,
            rom_rom_corrs,
            norm_aux_times,
            norm_rom_times_list,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _rom_input_t(self, s, t):
        """
        Extract trainable ROM t's coordinate from a full state vector.

        Cost functions in MFMC's cost_list are called with the expanded
        state s (length n_lofi); see model_indices.rom_state_offset.
        """
        idx = rom_state_offset(self.n_active, t)

        if torch.is_tensor(s):
            return s if s.ndim == 0 else s.reshape(-1)[idx]
        return s if np.isscalar(s) else np.reshape(s, -1)[idx]

    def _wrap_t(self, func, t):
        """Wrap a scalar ROM-t surrogate to handle full state-vector inputs."""
        def wrapped(s):
            s_input = self._rom_input_t(s, t)
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

    def _rom_pair_input(self, s, t, q):
        """
        Extract ROM t's and ROM q's coordinates from a full state vector as
        a length-2 vector/tensor, in that order. Feeds the dim=2 tensor-product
        sigmoid surrogate p_hat(s_t, s_q) of Sec. 4.3.2.
        """
        s_t = self._rom_input_t(s, t)
        s_q = self._rom_input_t(s, q)

        if torch.is_tensor(s_t) or torch.is_tensor(s_q):
            if not torch.is_tensor(s_t):
                s_t = torch.tensor(s_t, dtype=torch.float64)
            if not torch.is_tensor(s_q):
                s_q = torch.tensor(s_q, dtype=torch.float64)
            return torch.stack([s_t.reshape(()), s_q.reshape(())])

        return np.array([s_t, s_q], dtype=float)

    def _wrap_pair(self, func, t, q):
        """Wrap a scalar ROM_t-ROM_q surrogate to handle full state-vector inputs."""
        def wrapped(s):
            s_input = self._rom_pair_input(s, t, q)
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

    def _build_cost_list(self, norm_aux_times, norm_rom_times_list):
        """
        Costs stay independent per ROM (writeup Sec. 4.3.1: w_tilde_omega(s_omega)
        depends only on its own basis size) -- k independent 1-D polynomial
        fits by linear least squares, no tensor products and no cross terms.

        The same callable serves the numpy and torch backends: the fitted
        coefficients are constants, so `fit_cost_polynomial`'s Horner
        evaluation preserves autograd through a tensor input without needing
        a separate torch fit.
        """
        cost_list = [self._make_constant(float(t)) for t in norm_aux_times]

        for t in range(self.n_active):
            cost_rom_surr = fit_cost_polynomial(
                self.pilot_basis_grids[t],
                norm_rom_times_list[t],
                order=self.cost_poly_order,
            )
            cost_list.append(self._wrap_t(cost_rom_surr, t))

        return cost_list

    # ------------------------------------------------------------------
    # Direct scalar path
    # ------------------------------------------------------------------

    def _build_componentwise_sigmoid(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs_list,
        aux_rom_corrs_list,
        rom_rom_corrs,
        norm_aux_times,
        norm_rom_times_list,
    ):
        """
        Direct scalar sigmoid fitting to the pilot correlations
        (writeup Sec. 4.3.2).

        Fixed-fixed correlations are constants taken straight from the pilot.
        Each fixed-trainable pair gets a 1-D generalized sigmoid in that ROM's
        basis size; each trainable-trainable pair gets a 2-D tensor product of
        sigmoids in both basis sizes. This makes every entry individually a
        plausible correlation function but does not make P(s) globally
        admissible -- use an AH method for that.
        """
        k = self.n_active
        n_aux = self.n_aux

        # Fixed <-> ROM_t: one 1-D fit per (fixed model, ROM) pair.
        fom_rom_surrs = []
        aux_rom_surrs = [[None] * k for _ in range(n_aux)]

        for t in range(k):
            pilots_t = np.array(self.pilot_basis_grids[t], dtype=float)
            fom_rom_surrs.append(
                fit_sigmoid(pilots_t[None, :], fom_rom_corrs_list[t])
            )
            for j in range(n_aux):
                aux_rom_surrs[j][t] = fit_sigmoid(
                    pilots_t[None, :], aux_rom_corrs_list[j][t]
                )

        # ROM_t <-> ROM_q: one 2-D tensor-product fit per collected pair.
        rom_rom_surrs = {}
        for (t, q, s_t_vals, s_q_vals, corr_vals) in rom_rom_corrs:
            ins = np.stack([
                np.asarray(s_t_vals, dtype=float),
                np.asarray(s_q_vals, dtype=float),
            ])
            rom_rom_surrs[(t, q)] = fit_sigmoid(
                ins, np.asarray(corr_vals, dtype=float)
            )

        hf_corr_list = [self._make_constant(float(c)) for c in fom_aux_corrs]
        hf_corr_list.extend(self._wrap_t(fom_rom_surrs[t], t) for t in range(k))

        # lf_pairs yields the entries in exactly the flat order
        # MFMC.build_C reads lf_corr_list back in. For an aux_aux entry the
        # yielded (i, j) are auxiliary indices, and its position within
        # aux_aux_corrs is tril_position(i, j); for an aux_rom entry j is the
        # auxiliary index.
        lf_corr_list = []
        for i, j, kind, t, q in lf_pairs(n_aux, k):
            if kind == "aux_aux":
                corr = float(aux_aux_corrs[tril_position(i, j)])
                lf_corr_list.append(self._make_constant(corr))
            elif kind == "aux_rom":
                lf_corr_list.append(self._wrap_t(aux_rom_surrs[j][t], t))
            else:
                surr = rom_rom_surrs.get((t, q))
                if surr is None:
                    print(
                        f"Warning: no ROM{t}-ROM{q} correlation data "
                        "(every replicate was dropped); assuming zero "
                        "correlation."
                    )
                    lf_corr_list.append(self._make_constant(0.0))
                else:
                    lf_corr_list.append(self._wrap_pair(surr, t, q))

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times_list)

        return hf_corr_list, lf_corr_list, cost_list, None

    # ------------------------------------------------------------------
    # AH matrix path
    # ------------------------------------------------------------------

    def _build_ah_matrix(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs_list,
        aux_rom_corrs_list,
        rom_rom_corrs,
        norm_aux_times,
        norm_rom_times_list,
    ):
        n = self.n_models

        model, ah_map = self._load_or_train_ah_matrix_model(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs_list,
            aux_rom_corrs_list,
            rom_rom_corrs,
            n,
            AH_HIDDEN_SIZE,
        )

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times_list)

        fixed_mask, fixed_values = self._fixed_fixed_bandaid_terms(
            fom_aux_corrs, aux_aux_corrs, n
        )

        corr_matrix_fn = AHMatrixCorrelationSurrogate(
            omega_model=model,
            ah_map=ah_map,
            s_min=[tr[0] for tr in self.tunable_ranges],
            s_max=[tr[1] for tr in self.tunable_ranges],
            psd_check=self.ah_psd_check,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
        )

        return None, None, cost_list, corr_matrix_fn

    def _load_or_train_ah_matrix_model(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs_list,
        aux_rom_corrs_list,
        rom_rom_corrs,
        n,
        hidden_size,
    ):
        # Needed both to validate a cached model and, if training is
        # required, to hand off to _train_ah_matrix_model. inputs/targets/
        # weights come from one call so they can never drift out of sync
        # on the row count G.
        inputs, targets, weights = self._assemble_multi_rom_ah_design(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs_list,
            aux_rom_corrs_list,
            rom_rom_corrs,
        )
        inputs_norm = self._normalize_training_inputs(inputs)
        fixed_mask, fixed_values = self._fixed_fixed_bandaid_terms(
            fom_aux_corrs, aux_aux_corrs, n
        )

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location="cpu")

            compatible = (
                checkpoint.get("pilot_basis_grids") == self.pilot_basis_grids
                and checkpoint.get("tunable_ranges") == self.tunable_ranges
                and checkpoint.get("n_active") == self.n_active
                and checkpoint.get("n_aux") == self.n_aux
                and checkpoint.get("method") == self.method
            )

            if compatible:
                model = VeclNet(self.n_active, AH_HIDDEN_SIZE, n).double()
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
                    model, ah_map, inputs_norm, targets, weights,
                    fixed_mask, fixed_values,
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
            fom_rom_corrs_list,
            aux_rom_corrs_list,
            rom_rom_corrs,
            n,
            hidden_size,
        )

    def _train_ah_matrix_model(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs_list,
        aux_rom_corrs_list,
        rom_rom_corrs,
        n,
        hidden_size,
        max_restarts=AH_MAX_RESTARTS,
        fixed_fixed_fit_tol=None,
    ):
        if fixed_fixed_fit_tol is None:
            fixed_fixed_fit_tol = self.ah_fixed_fixed_tol

        print(
            f"Training AH matrix surrogate: "
            f"n={n} (1 FOM + {self.n_aux} aux + {self.n_active} ROM)\n"
        )

        inputs, targets, weights = self._assemble_multi_rom_ah_design(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs_list,
            aux_rom_corrs_list,
            rom_rom_corrs,
        )

        inputs_norm = self._normalize_training_inputs(inputs)
        fixed_mask, fixed_values = self._fixed_fixed_bandaid_terms(
            fom_aux_corrs, aux_aux_corrs, n
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
            model = VeclNet(self.n_active, hidden_size, n).double()

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
                lr=AH_LEARNING_RATE,
                max_steps=AH_MAX_STEPS,
                tol=AH_LOSS_TOL,
                grad_clip=AH_GRAD_CLIP,
                print_every=AH_PRINT_EVERY,
                fixed_mask=fixed_mask,
                fixed_values=fixed_values,
            )

            model.eval()

            err = self._fixed_fixed_fit_error(
                model, ah_map, inputs_norm, targets, weights,
                fixed_mask, fixed_values,
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
                    "pilot_basis_grids": self.pilot_basis_grids,
                    "tunable_ranges": self.tunable_ranges,
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
        """
        inputs: (..., k) tensor, one column per trainable ROM. Normalized
        elementwise per column against that ROM's own tunable_ranges[t].
        """
        s_min = torch.tensor(
            [tr[0] for tr in self.tunable_ranges],
            dtype=inputs.dtype,
            device=inputs.device,
        )
        s_max = torch.tensor(
            [tr[1] for tr in self.tunable_ranges],
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return 2.0 * (inputs - s_min) / (s_max - s_min) - 1.0

    def _fixed_fixed_position_mask(self, n):
        """
        (n, n) boolean mask, True at every FOM-aux and aux-aux off-diagonal
        position -- the entries _fill_fixed_fixed_entries pins at every
        training row, regardless of surrogate strategy. Used to identify
        those entries exactly, rather than inferring them from weights
        (see _fixed_fixed_fit_error).
        """
        mask = torch.zeros((n, n), dtype=torch.bool)

        for i in range(self.n_aux):
            idx = aux_model_slot(i)
            mask[idx, 0] = mask[0, idx] = True

        for _, i, j in aux_pairs(self.n_aux):
            idx_i, idx_j = aux_model_slot(i), aux_model_slot(j)
            mask[idx_i, idx_j] = mask[idx_j, idx_i] = True

        return mask

    def _fixed_fixed_bandaid_terms(self, fom_aux_corrs, aux_aux_corrs, n):
        """
        (mask, values) template for the 'ah_matrix_bandaid' strategy: True /
        known correlation at every FOM-aux and aux-aux off-diagonal position.
        Returns (None, None) outside bandaid mode, so callers can pass the
        result straight through without an extra method check.
        """
        if self.method != "ah_matrix_bandaid":
            return None, None

        mask = self._fixed_fixed_position_mask(n)
        values = torch.eye(n, dtype=torch.float64)

        for i in range(self.n_aux):
            idx = aux_model_slot(i)
            values[idx, 0] = values[0, idx] = float(fom_aux_corrs[i])

        for pair, i, j in aux_pairs(self.n_aux):
            idx_i, idx_j = aux_model_slot(i), aux_model_slot(j)
            values[idx_i, idx_j] = values[idx_j, idx_i] = float(aux_aux_corrs[pair])

        return mask, values

    def _fixed_fixed_fit_error(
        self,
        model,
        ah_map,
        inputs_norm,
        targets,
        weights,
        fixed_mask=None,
        fixed_values=None,
    ):
        """
        Max absolute error between the trained surrogate and the known
        fixed-fixed correlation targets (FOM-aux and aux-aux entries). These
        entries do not depend on s, so a good fit should match them closely
        at every pilot point; a large error signals the training run landed
        in a saturated, low-gradient region of the AH map.

        Uses the exact fixed-fixed position mask rather than inferring the
        entries from `weights >= 0.5*fixed_fixed_weight`: that threshold
        only isolates the right entries when fixed_fixed_weight is set well
        above every other entry's weight. Diagonal entries and every
        FOM/aux<->ROM entry the multi-ROM design adds also carry weight 1,
        which the default fixed_fixed_weight=1.0 collides with -- and the
        more trainable ROMs there are, the more such entries dilute the
        weight-based mask, so the reported error increasingly reflects
        ordinary ROM-regression residual rather than the fixed-fixed fit.

        If fixed_mask/fixed_values are given (bandaid mode), they are
        inserted into P_pred first, same as at inference, so this reports
        the (trivially ~0) post-bandaid error rather than the raw AH fit.
        """
        with torch.no_grad():
            ah_map.reset_cache()
            P_pred = ah_map(model(inputs_norm))

            if fixed_mask is not None:
                P_pred = insert_fixed_fixed_entries(P_pred, fixed_mask, fixed_values)

        position_mask = fixed_mask if fixed_mask is not None else self._fixed_fixed_position_mask(targets.shape[-1])

        if not torch.any(position_mask):
            return 0.0

        mask = position_mask.unsqueeze(0).expand_as(targets)

        return float(torch.max(torch.abs(P_pred[mask] - targets[mask])))

    def _grid_index(self, t, s_val):
        """
        Locate s_val's position in ROM t's own pilot grid. Trainable-trainable
        pilot triples carry basis-size values rather than grid indices, so this
        recovers the index needed to look up fom_rom_corrs_list[t] and
        aux_rom_corrs_list[j][t] at that point. Falls back to the nearest grid
        value defensively.
        """
        grid = self.pilot_basis_grids[t]
        try:
            return grid.index(s_val)
        except ValueError:
            arr = np.asarray(grid, dtype=float)
            return int(np.argmin(np.abs(arr - float(s_val))))

    def _fill_fixed_fixed_entries(self, P, W, fom_aux_corrs, aux_aux_corrs):
        """FOM-aux / aux-aux entries: identical at every row, weight fixed_fixed_weight."""
        for i in range(self.n_aux):
            idx = aux_model_slot(i)
            P[idx, 0] = fom_aux_corrs[i]
            P[0, idx] = fom_aux_corrs[i]

            W[idx, 0] = self.fixed_fixed_weight
            W[0, idx] = self.fixed_fixed_weight

        for pair, i, j in aux_pairs(self.n_aux):
            idx_i, idx_j = aux_model_slot(i), aux_model_slot(j)

            corr = aux_aux_corrs[pair]
            P[idx_i, idx_j] = corr
            P[idx_j, idx_i] = corr

            W[idx_i, idx_j] = self.fixed_fixed_weight
            W[idx_j, idx_i] = self.fixed_fixed_weight

    def _fill_fom_aux_rom_entries(
        self, P, W, t, grid_idx, rom_idx_t, fom_rom_corrs_list, aux_rom_corrs_list
    ):
        """FOM-ROM_t / aux_j-ROM_t entries at ROM t's grid index grid_idx, weight 1."""
        corr = fom_rom_corrs_list[t][grid_idx]
        P[rom_idx_t, 0] = corr
        P[0, rom_idx_t] = corr
        W[rom_idx_t, 0] = 1.0
        W[0, rom_idx_t] = 1.0

        for j in range(self.n_aux):
            idx_j = aux_model_slot(j)
            corr = aux_rom_corrs_list[j][t][grid_idx]

            P[rom_idx_t, idx_j] = corr
            P[idx_j, rom_idx_t] = corr

            W[rom_idx_t, idx_j] = 1.0
            W[idx_j, rom_idx_t] = 1.0

    def _assemble_multi_rom_ah_design(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs_list,
        aux_rom_corrs_list,
        rom_rom_corrs,
    ):
        """
        Assemble the AH training design (inputs, targets, weights) that
        Sec. 4.3.3's normalized Frobenius objective is minimized over.

        Model layout is the full index space of model_indices. Two groups of
        rows, built jointly so inputs/targets/weights cannot drift out of sync
        on the row count G:

        1. Own-grid rows: for each ROM t and each of its own pilot points,
           one row sweeping s_t with every other ROM q held at its own
           reference point B_q[0]. Fills FOM/aux <-> ROM_t (from t's own
           pilot data at this point) and, "for free", FOM/aux <-> ROM_q
           for every other q (from q's own pilot data at its reference
           point). ROM_t <-> ROM_q stays weight 0 in these rows.

        2. Cross rows: one row per (t, q, s_t, s_q, corr) triple of pilot
           trainable-trainable data, sweeping (s_t, s_q) jointly with every
           other ROM held at its own reference point. Fills ROM_t<->ROM_q
           plus, again "for free", FOM/aux <-> ROM_t at s_t and
           FOM/aux <-> ROM_q at s_q.

        Fixed-fixed entries (FOM-aux, aux-aux) keep fixed_fixed_weight at
        every row.

        With one trainable ROM there are no ROM pairs, so rom_rom_corrs is
        empty, every weight is 1, and the objective reduces exactly to the
        unweighted normalized Frobenius loss of Sec. 4.3.3.
        """
        n = self.n_models
        k = self.n_active
        grids = self.pilot_basis_grids

        s_rows, P_rows, W_rows = [], [], []

        def _new_matrices():
            return np.eye(n, dtype=float), np.zeros((n, n), dtype=float)

        # 1. Own-grid rows.
        for t in range(k):
            for i, s_ti in enumerate(grids[t]):
                s_vec = [grids[q][0] for q in range(k)]
                s_vec[t] = s_ti

                P, W = _new_matrices()
                self._fill_fixed_fixed_entries(P, W, fom_aux_corrs, aux_aux_corrs)

                rom_idx_t = rom_model_slot(self.n_aux, t)
                self._fill_fom_aux_rom_entries(
                    P, W, t, i, rom_idx_t, fom_rom_corrs_list, aux_rom_corrs_list
                )

                for q in range(k):
                    if q == t:
                        continue
                    rom_idx_q = rom_model_slot(self.n_aux, q)
                    self._fill_fom_aux_rom_entries(
                        P, W, q, 0, rom_idx_q, fom_rom_corrs_list, aux_rom_corrs_list
                    )
                    # ROM_t <-> ROM_q left at weight 0: this row carries no
                    # cross-correlation data for that pair (the cross rows do).

                for i_diag in range(n):
                    W[i_diag, i_diag] = 1.0

                s_rows.append(s_vec)
                P_rows.append(P)
                W_rows.append(W)

        # 2. Cross rows.
        for (t, q, s_t_vals, s_q_vals, corr_vals) in rom_rom_corrs:
            rom_idx_t = rom_model_slot(self.n_aux, t)
            rom_idx_q = rom_model_slot(self.n_aux, q)

            for idx in range(len(corr_vals)):
                s_t_val, s_q_val = s_t_vals[idx], s_q_vals[idx]
                i_t = self._grid_index(t, s_t_val)
                i_q = self._grid_index(q, s_q_val)

                s_vec = [grids[r][0] for r in range(k)]
                s_vec[t] = s_t_val
                s_vec[q] = s_q_val

                P, W = _new_matrices()
                self._fill_fixed_fixed_entries(P, W, fom_aux_corrs, aux_aux_corrs)

                corr = corr_vals[idx]
                P[rom_idx_t, rom_idx_q] = corr
                P[rom_idx_q, rom_idx_t] = corr
                W[rom_idx_t, rom_idx_q] = 1.0
                W[rom_idx_q, rom_idx_t] = 1.0

                self._fill_fom_aux_rom_entries(
                    P, W, t, i_t, rom_idx_t, fom_rom_corrs_list, aux_rom_corrs_list
                )
                self._fill_fom_aux_rom_entries(
                    P, W, q, i_q, rom_idx_q, fom_rom_corrs_list, aux_rom_corrs_list
                )

                for i_diag in range(n):
                    W[i_diag, i_diag] = 1.0

                s_rows.append(s_vec)
                P_rows.append(P)
                W_rows.append(W)

        inputs = torch.tensor(np.array(s_rows, dtype=float), dtype=torch.float64)
        targets = torch.tensor(np.stack(P_rows), dtype=torch.float64)
        weights = torch.tensor(np.stack(W_rows), dtype=torch.float64)

        return inputs, targets, weights

    # ------------------------------------------------------------------
    # Legacy AH + componentwise sigmoid path
    # ------------------------------------------------------------------

    def _build_ah_componentwise_sigmoid(
        self,
        fom_aux_corrs,
        aux_aux_corrs,
        fom_rom_corrs_list,
        aux_rom_corrs_list,
        rom_rom_corrs,
        norm_aux_times,
        norm_rom_times_list,
    ):
        n = self.n_models

        # Reuse the AH matrix training routine, but then intentionally discard
        # the matrix-valued surrogate by fitting scalar surrogates to entries.
        model, ah_map = self._load_or_train_ah_matrix_model(
            fom_aux_corrs,
            aux_aux_corrs,
            fom_rom_corrs_list,
            aux_rom_corrs_list,
            rom_rom_corrs,
            n,
            AH_HIDDEN_SIZE,
        )

        hf_corr_list, lf_corr_list = self._fit_surrogates_to_model(
            model,
            ah_map,
            n,
            rom_rom_corrs,
        )

        cost_list = self._build_cost_list(norm_aux_times, norm_rom_times_list)

        return hf_corr_list, lf_corr_list, cost_list, None

    def _query_ah_model(self, model, ah_map, z):
        """Evaluate the trained AH model at raw (unnormalized) k-column
        coordinate rows z, shape (batch, k)."""
        z_tensor = torch.tensor(np.asarray(z, dtype=float), dtype=torch.float64)
        z_norm = self._normalize_training_inputs(z_tensor)

        with torch.no_grad():
            ah_map.reset_cache()
            return model.corr_matrix(z_norm, ah_map).cpu().numpy()

    def _fit_surrogates_to_model(self, model, ah_map, n, rom_rom_corrs=()):
        """
        Query the trained AH matrix model and fit componentwise scalar
        surrogates to its entries:

        - Own-grid sweeps: for each ROM t, a dense 1-D grid over s_t with
          every other ROM held at its own reference point, mirroring the
          own-grid rows of the training design. Feeds the fixed <-> ROM_t
          1-D fits.
        - Cross sweeps: for each (t, q) pair, the pilot's own (s_t, s_q)
          sample points rather than a dense grid over both, which would grow
          combinatorially. Feeds the ROM_t <-> ROM_q 2-D fits.

        Refitting scalars discards the global admissibility that the AH map
        provided, so P(s) may not be PSD afterwards.
        """
        k = self.n_active
        ref = np.array([g[0] for g in self.pilot_basis_grids], dtype=float)

        own_grids, own_corrs = [], []
        for t in range(k):
            grid_t = np.unique(np.concatenate([
                np.asarray(self.pilot_basis_grids[t], dtype=float),
                np.linspace(
                    self.tunable_ranges[t][0],
                    self.tunable_ranges[t][1],
                    AH_RESAMPLE_POINTS,
                ),
            ]))
            z = np.tile(ref, (len(grid_t), 1))
            z[:, t] = grid_t
            own_grids.append(grid_t)
            own_corrs.append(self._query_ah_model(model, ah_map, z))

        cross_grids, cross_corrs = {}, {}
        for (t, q, s_t_vals, s_q_vals, _corr_vals) in rom_rom_corrs:
            s_t_vals = np.asarray(s_t_vals, dtype=float)
            s_q_vals = np.asarray(s_q_vals, dtype=float)
            z = np.tile(ref, (len(s_t_vals), 1))
            z[:, t] = s_t_vals
            z[:, q] = s_q_vals
            cross_grids[(t, q)] = (s_t_vals, s_q_vals)
            cross_corrs[(t, q)] = self._query_ah_model(model, ah_map, z)

        if self.work_dir and k >= 1:
            # Diagnostic-only: plots ROM 0's own-grid sweep. Per-ROM profile
            # plots for the full set live in postprocess_hybrid_mfuq.py.
            self._plot_correlations(own_corrs[0], own_grids[0], n)

        surrogates = self._fit_componentwise_surrogates(
            own_grids, own_corrs, cross_grids, cross_corrs
        )

        hf_corr_list = [surrogates[(i, 0)] for i in range(1, n)]
        lf_corr_list = [
            surrogates[(i, j)]
            for i in range(1, n)
            for j in range(1, i)
        ]

        return hf_corr_list, lf_corr_list

    def _sigmoid_or_constant(self, ins, values, wrap, label):
        """
        Fit one scalar correlation surrogate to sampled AH entries.

        Flat data is returned as a constant, since a sigmoid through it is
        unidentifiable. Otherwise a sigmoid is fitted and its fit quality
        reported; a poor fit is a loud warning rather than a silent swap to a
        different model class. Earlier revisions fell back to a cubic spline
        (numpy backend) or kept the sigmoid regardless (torch backend), so the
        two backends could return different surrogates from identical data.
        Both now take this single path.
        """
        values = np.asarray(values, dtype=float)

        if np.std(values) < CONSTANT_CORRELATION_STD:
            return self._make_constant(float(np.mean(values)))

        surrogate = fit_sigmoid(ins, values)
        mse = float(np.mean((surrogate(ins) - values) ** 2))

        if mse >= SIGMOID_FIT_MSE_WARN:
            print(
                f"Warning: sigmoid fit for {label} has MSE {mse:.3e}; the "
                f"componentwise surrogate for this entry may be unreliable."
            )

        return wrap(surrogate)

    def _fit_componentwise_surrogates(
        self, own_grids, own_corrs, cross_grids, cross_corrs
    ):
        """
        Fit scalar surrogates to correlation entries sampled from a trained AH
        model. Backend-independent: `fit_sigmoid` dispatches on argument type,
        so one fitted surrogate serves both the numpy and torch optimizers.
        """
        surrogates = {}

        for i, j, kind, t, q in matrix_pairs(self.n_aux, self.n_active):
            if kind == "fixed_fixed":
                values = own_corrs[0][:, i, j] if own_corrs else np.array([0.0])
                surrogates[(i, j)] = self._make_constant(float(np.mean(values)))

            elif kind == "fixed_rom":
                surrogates[(i, j)] = self._sigmoid_or_constant(
                    own_grids[t][None, :],
                    own_corrs[t][:, i, j],
                    lambda f, t=t: self._wrap_t(f, t),
                    f"entry ({i},{j}) vs ROM {t} basis size",
                )

            else:
                pair = cross_corrs.get((t, q))
                if pair is None:
                    surrogates[(i, j)] = self._make_constant(0.0)
                    continue

                s_t_vals, s_q_vals = cross_grids[(t, q)]
                surrogates[(i, j)] = self._sigmoid_or_constant(
                    np.stack([s_t_vals, s_q_vals]),
                    pair[:, i, j],
                    lambda f, t=t, q=q: self._wrap_pair(f, t, q),
                    f"entry ({i},{j}) vs ROM {t}/ROM {q} basis sizes",
                )

        return surrogates

    def _plot_correlations(self, corr_matrices, s_grid, n):
        """
        Diagnostic plot of the trained AH surrogate's correlation entries
        against ROM 0's pilot data, written to work_dir/debug_plots. Purely
        informational; failures here never interrupt surrogate construction.
        """
        try:
            import matplotlib.pyplot as plt

            debug_dir = os.path.join(self.work_dir, "debug_plots")
            os.makedirs(debug_dir, exist_ok=True)

            with np.load(os.path.join(self.work_dir, "pilot_results.npz")) as data:
                fom_aux = data["fom_aux_corrs"]
                aux_aux = data.get("aux_aux_corrs", np.array([]))
                # Diagnostic plot always covers ROM 0's own-grid sweep (see
                # the sole call site in _fit_surrogates_to_model).
                fom_rom = data["rom0_fom_corrs"]
                aux_rom = [data[f"aux{i}_rom0_corrs"] for i in range(self.n_aux)]

            pilot_data, names = {}, {}

            n_grid = len(self.pilot_basis_grids[0])

            for i in range(self.n_aux):
                slot = aux_model_slot(i)
                pilot_data[(slot, 0)] = np.full(n_grid, fom_aux[i])
                names[(slot, 0)] = f"FOM-aux{i}"

            for pair, i, j in aux_pairs(self.n_aux):
                slot_i, slot_j = aux_model_slot(i), aux_model_slot(j)
                pilot_data[(slot_i, slot_j)] = np.full(n_grid, aux_aux[pair])
                names[(slot_i, slot_j)] = f"aux{j}-aux{i}"

            rom_idx = rom_model_slot(self.n_aux, 0)
            pilot_data[(rom_idx, 0)] = fom_rom
            names[(rom_idx, 0)] = "FOM-ROM0"

            for i in range(self.n_aux):
                pilot_data[(rom_idx, aux_model_slot(i))] = aux_rom[i]
                names[(rom_idx, aux_model_slot(i))] = f"aux{i}-ROM0"

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
                            self.pilot_basis_grids[0],
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

        except (ImportError, OSError, KeyError, ValueError, IndexError) as e:
            print(f"Warning: diagnostic plot generation failed: {e}")