"""
Pilot sampling for Multi-Fidelity UQ (writeup Sec. 4.2).

Collects the cost and correlation statistics that the Sec. 4.3 surrogates are
fitted to: fixed-model statistics over the whole pilot set, and, for each
trainable ROM basis size, statistics estimated on the pilot samples that ROM
was not trained on. `Pilot` owns the resampling bookkeeping; `PilotSampler`
runs the models and assembles `PilotData`.

Model ordering and pair flattening come from `model_indices.py`, which also
carries the writeup-to-code notation map.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Tuple
from math import comb
from itertools import combinations

import numpy as np
import scipy.stats

from romtools.workflows.models import QoiModel
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.hybrid_mfuq.model_indices import aux_pairs


def run_model_sample(model, run_dir, params, overwrite=False, verbose=False):
    """
    Run a model sample, using cached qoi.txt/time.txt if available.

    Parameters
    ----------
    model : QoiModel
        Model to run (FOM, auxiliary model, or ROM).
    run_dir : str
        Directory to run the model in (and to cache qoi.txt/time.txt in).
    params : dict
        Parameter values for this sample.
    overwrite : bool
        If True, ignore any cached qoi.txt/time.txt and re-run.
    verbose : bool
        If True, print a message when reading a cached result.

    Returns
    -------
    qoi : np.ndarray
    runtime : np.ndarray
    """
    qoi_path = os.path.join(run_dir, "qoi.txt")
    time_path = os.path.join(run_dir, "time.txt")

    if not overwrite and os.path.exists(qoi_path) and os.path.exists(time_path):
        if verbose:
            print("Reading in QoI value and runtime\n")
        return np.loadtxt(qoi_path), np.loadtxt(time_path)

    create_empty_dir(run_dir)
    model.populate_run_directory(run_dir, params)

    passed_file = os.path.join(run_dir, "passed.txt")
    t0 = time.time()
    code = model.run_model(run_dir, params)
    qoi = model.compute_qoi(run_dir, params)
    runtime = time.time() - t0

    if code == 0:
        np.savetxt(passed_file, [0], fmt="%i")

    np.savetxt(qoi_path, [qoi])
    np.savetxt(time_path, [runtime])

    return np.array(qoi), np.array(runtime)


def pearsonr_with_axis(x, y, axis=0):
    """
    Compute the Pearson correlation coefficient between two arrays along a specified axis.

    Parameters
    ----------
    x, y : array_like
        Input arrays.
    axis : int
        Axis along which to compute the correlation.

    Returns
    -------
    statistic : ndarray
        Pearson correlation coefficient.
    pvalue : ndarray
        Two-tailed p-value.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mean_x = np.mean(x, axis=axis, keepdims=True)
    mean_y = np.mean(y, axis=axis, keepdims=True)

    numerator = np.sum((x - mean_x) * (y - mean_y), axis=axis)
    denominator = np.sqrt(
        np.sum((x - mean_x) ** 2, axis=axis)
        * np.sum((y - mean_y) ** 2, axis=axis)
    )

    statistic = numerator / denominator

    n = np.sum(~np.isnan(x) & ~np.isnan(y), axis=axis)
    df = n - 2
    t_stat = statistic * np.sqrt(df / (1 - statistic ** 2))
    pvalue = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), df))

    return statistic, pvalue


def _pearson_statistic(x, y):
    """
    Pearson correlation coefficient only (no p-value).

    Used for trainable-trainable (ROM-ROM) correlations, whose validation
    subsets can be as small as 2 points (see `min_pair_validation_size`),
    too small for `pearsonr_with_axis`'s p-value (df = n - 2 <= 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = x - x.mean()
    dy = y - y.mean()
    denom = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))

    if denom == 0:
        return np.nan

    return float(np.sum(dx * dy) / denom)


class Pilot:
    """
    Abstract class which handles pilot sampling.
    Includes strategies to estimate correlations and costs.
    """

    def __init__(self, s_lists, num_pilot, random_seed=2025, min_pair_validation_size=1):
        """
        s_lists: one basis-size grid B_t per trainable ROM t, so
            n_active == len(s_lists).
        min_pair_validation_size: smallest validation set a trainable-trainable
            replicate may have before it is discarded (the writeup's h). See
            `set_ROM_correlation_labels` for why discarding is not equivalent
            to reserving a holdout. No-op with one trainable ROM.
        """
        self.s_lists = s_lists
        self.num_pilot = num_pilot
        self.rng = np.random.default_rng(random_seed)
        self.min_pair_validation_size = min_pair_validation_size

    def set_train_and_test_labels(self, max_groups=int(1e6)):
        """
        Create train/test index groups for pilot sampling, for every
        trainable ROM.

        Populates self.train_labels[t][i] / self.test_labels[t][i] for each
        ROM t and each grid point index i in s_lists[t], all drawn from the
        shared pilot pool range(num_pilot). Each training set A of size s has
        validation set V = I_p \\ A, the maximal validation set of Sec. 4.2.

        Combination index i indexes the writeup's resampling replicates: all
        C(N_p, s) training sets when there are few enough, otherwise
        max_groups random ones.
        """
        pilot_set = set(range(self.num_pilot))
        train_labels = [[0 for _ in s_list] for s_list in self.s_lists]
        test_labels = [[0 for _ in s_list] for s_list in self.s_lists]

        for t, s_list in enumerate(self.s_lists):
            for i, s in enumerate(s_list):
                NpCs = comb(self.num_pilot, s)

                if NpCs <= max_groups:
                    labels_i = [
                        list(c) for c in combinations(range(self.num_pilot), s)
                    ]
                else:
                    labels_i = []
                    seen = set()
                    while len(labels_i) < max_groups:
                        label = sorted(
                            self.rng.choice(range(self.num_pilot), s, replace=False)
                        )
                        tup = tuple(label)
                        if tup not in seen:
                            seen.add(tup)
                            labels_i.append(label)

                train_labels[t][i] = labels_i
                test_labels[t][i] = [
                    list(pilot_set - set(labels_i[k]))
                    for k in range(len(labels_i))
                ]

        self.train_labels = train_labels
        self.test_labels = test_labels

    def split_data_using_labels(self, data_list, rom_idx=0):
        """
        Split data arrays using train/test labels for one trainable ROM
        (rom_idx, default 0 — the single-ROM case).

        Returns lists of arrays shaped [NpCs, s, ...] for train and
        [NpCs, Np-s, ...] for test data.
        """
        train_data, test_data = [], []
        train_labels = self.train_labels[rom_idx]
        test_labels = self.test_labels[rom_idx]

        for data in data_list:
            train_data_i, test_data_i = [], []
            for train, test in zip(train_labels, test_labels):
                train_data_i.append(data[train])
                test_data_i.append(data[test])
            train_data.append(train_data_i)
            test_data.append(test_data_i)

        return train_data, test_data

    def estimate_pairwise_correlations(self, X_test_list, y_test_list):
        """
        Mean Pearson correlation between two sets of model outputs, averaged
        over pilot replicates.

        Serves every correlation in Sec. 4.2 that is a plain average of
        per-replicate Pearson coefficients: fixed-fixed (6) over the whole
        pilot set, and fixed-trainable (7)-(8) over each replicate's
        validation set. Trainable-trainable pairs need the matched-replicate
        machinery of `estimate_ROM_ROM_correlation` instead.
        """
        corr_s = []

        for X, y in zip(X_test_list, y_test_list):
            out, _ = pearsonr_with_axis(X, y, axis=1)
            corr_s.append(np.mean(out, axis=0).squeeze())

        return np.array(corr_s)

    def set_ROM_correlation_labels(self, min_pair_validation_size=None):
        """
        Compute shared, validation-size-guaranteed test indices usable for
        trainable-trainable (ROM-ROM) correlations, across every pair of
        trainable ROMs (t, q), t < q, over the full cross product of their
        basis-size grids B_t x B_q.

        Relationship to the writeup's Algorithm 5 (see writeup §4.2).

        The replicate structure IS preserved, contrary to what an earlier
        version of this docstring claimed. `set_train_and_test_labels` draws
        each (ROM t, basis size s) its own combo list once; this method pairs
        those lists by index, so combo index k plays the role of the writeup's
        replicate index b, and the *same* ROM instance k of model t is reused
        in every pair involving t. Correlations for different pairs at the
        same k are therefore computed from one coherent set of ROM instances,
        exactly as Algorithm 5 intends.

        What is genuinely missing is the reserved holdout: Algorithm 5 draws
        A^(b)_omega ⊆ D^(b) = I_p \\ H^(b) with |H^(b)| = h, so every pair's
        validation set
            V = I_p \\ (A_t u A_q) = test_t[k] ^ test_q[k]
        automatically satisfies |V| >= h. Here A_t and A_q are drawn from all
        of I_p, |V| is random, and undersized draws are *dropped* after the
        fact via `min_pair_validation_size`.

        That post-hoc drop is not equivalent to reserving a holdout, and the
        difference is statistical, not just bookkeeping. Since

            |V| = Np - s_t - s_q + |A_t ^ A_q|,

        filtering on |V| >= h is filtering on |A_t ^ A_q|, i.e. it *retains
        preferentially those replicates in which the two ROMs were trained on
        overlapping snapshots* -- and two ROMs sharing training data tend to
        agree more. Whenever the floor binds, the surviving average is biased
        upward relative to the estimand. Reserving H^(b) up front avoids this
        because the guarantee holds for every draw, so nothing is conditioned
        on.

        The floor is harmless when it never binds, which is the usual case:
        |V| >= Np - s_t - s_q even at zero overlap. It starts to bind as
        s_t + s_q approaches Np, or when `min_pair_validation_size` is raised
        -- note that raising it to stabilize the estimates is exactly what
        introduces the bias. The drop rate is reported below; if it is
        materially nonzero, prefer reserving a holdout in
        `set_train_and_test_labels` over raising the floor here.

        Randomness: the pairing order comes from `self.rng` (seeded by the
        Pilot's `random_seed`), not from a hardcoded global `np.random.seed`.
        It only permutes the order in which matched combos are stored, so it
        does not affect any estimate; routing it through `self.rng` keeps the
        top-level `random_seed` in control and stops the pilot from resetting
        the global numpy stream that `MFMC._initial_guess` also draws from.

        Implementation note: ROM blocks are flattened, in ROM order, into
        one list of grid points before running the (now floor-filtered)
        pairwise-intersection loop. Because later ROMs' points always sit
        after earlier ROMs' points in this flattened order, pairing a
        later point against *every* earlier point automatically produces
        the full cross product B_t x B_q for every t < q — with no dense
        k-way grid required. It also produces same-ROM (t == q) pairs as a
        side effect (matching the original k=1 behavior byte-for-byte);
        those are simply not trainable-trainable pairs and are filtered
        out by callers (see PilotSampler._compute_rom_rom_corrs).

        Populates:
            self.rom_correlation_owner: flat list of (rom_idx, grid_idx,
                basis_size), one entry per flattened grid point, in the
                same order used below.
            self.ROM_correlation_indices: nested list; indices[a][b] (for
                b < a in flattened order) is a list of (k, validation)
                tuples — k the matched combo index, validation the list of
                shared held-out pilot sample indices for that combo — for
                every matched combo whose validation set meets the floor.
        """
        if min_pair_validation_size is None:
            min_pair_validation_size = self.min_pair_validation_size

        flat_test_labels = []
        owner = []
        for t, labels_t in enumerate(self.test_labels):
            for i, label in enumerate(labels_t):
                flat_test_labels.append(label)
                owner.append((t, i, self.s_lists[t][i]))

        indices = []
        n_considered = 0
        n_dropped = 0

        for a, label_a in enumerate(flat_test_labels):
            indices_a = []
            NpCs_a = len(label_a)

            for b, label_b in enumerate(flat_test_labels[:a]):
                NpCs_b = len(label_b)
                m = min(NpCs_a, NpCs_b)
                random_ab = self.rng.permutation(m)

                pairs = []
                for k in random_ab:
                    validation = set(label_a[k]) & set(label_b[k])
                    n_considered += 1

                    if len(validation) >= min_pair_validation_size:
                        pairs.append((int(k), list(validation)))
                    else:
                        n_dropped += 1

                if not pairs:
                    print(
                        f"Warning: no basis-size pair for groups "
                        f"{owner[a]},{owner[b]} meets "
                        f"min_pair_validation_size={min_pair_validation_size}"
                    )

                indices_a.append(pairs)

            indices.append(indices_a)

        self.ROM_correlation_indices = indices
        self.rom_correlation_owner = owner
        self.rom_correlation_drop_rate = (
            n_dropped / n_considered if n_considered else 0.0
        )

        # Dropping conditions on |A_t ^ A_q| and so biases the retained
        # replicates toward overlapping training sets (see docstring). Silent
        # at a zero drop rate; loud once the floor actually binds.
        if n_dropped:
            print(
                f"Note: dropped {n_dropped}/{n_considered} "
                f"({100.0 * self.rom_correlation_drop_rate:.1f}%) "
                f"trainable-trainable replicates for having fewer than "
                f"{min_pair_validation_size} validation samples."
            )

            if self.rom_correlation_drop_rate > 0.1:
                print(
                    "Warning: this filter selects on training-set overlap, so "
                    "at this drop rate the trainable-trainable correlations "
                    "are likely biased upward. Consider reserving a holdout "
                    "set instead of raising min_pair_validation_size."
                )

    def estimate_ROM_ROM_correlation(
        self, qois_a, test_labels_a, qois_b, test_labels_b, combo_val_pairs
    ):
        """
        Trainable-trainable correlation (10)-(11): the Pearson correlation of
        two ROMs' QoI values on the samples neither was trained on, averaged
        over the matched replicates retained by set_ROM_correlation_labels.

        qois_a, qois_b: arrays of shape [NpCs, Np-s] (test-fold QoI
            values; row k is aligned with test_labels_a[k]/test_labels_b[k]
            in sample-index order).
        combo_val_pairs: list of (k, validation_indices) tuples, as
            produced by set_ROM_correlation_labels.

        Returns np.nan if no combo has a validation set of size >= 2
        (Pearson correlation is undefined for fewer than 2 points).
        """
        corrs = []

        for k, validation in combo_val_pairs:
            if len(validation) < 2:
                continue

            pos_a = {idx: p for p, idx in enumerate(test_labels_a[k])}
            pos_b = {idx: p for p, idx in enumerate(test_labels_b[k])}

            x = np.array([qois_a[k, pos_a[idx]] for idx in validation])
            y = np.array([qois_b[k, pos_b[idx]] for idx in validation])

            corr = _pearson_statistic(x, y)
            if not np.isnan(corr):
                corrs.append(corr)

        if not corrs:
            return np.nan

        return float(np.mean(corrs))


@dataclass
class PilotData:
    """
    Container for pilot sampling results (writeup Sec. 4.2).

    fom_rom_corrs_list, aux_rom_corrs_list, and normalized_rom_times_list are
    indexed by trainable ROM t; aux_rom_corrs_list is [aux i][rom t]. Each
    rom_rom_corrs entry is one pair of trainable ROMs (t, q), t < q, as
    (t, q, s_t_vals, s_q_vals, corr_vals) with parallel 1-D arrays. It is
    empty when there is only one trainable ROM, since no pairs exist.

    This class owns the on-disk schema: `to_npz_dict` and `from_npz` are the
    only places the npz key names appear, so writer and reader cannot drift.
    """
    fom_qois: np.ndarray
    aux_qois_list: List[np.ndarray]
    fom_aux_corrs: np.ndarray
    aux_aux_corrs: np.ndarray
    fom_rom_corrs_list: List[np.ndarray]
    aux_rom_corrs_list: List[List[np.ndarray]]
    rom_rom_corrs: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]
    fom_times: np.ndarray
    normalized_aux_times: np.ndarray
    normalized_rom_times_list: List[np.ndarray]
    parameter_samples: np.ndarray
    training_dirs: np.ndarray

    @property
    def n_aux(self):
        return len(self.aux_qois_list)

    @property
    def n_active(self):
        return len(self.fom_rom_corrs_list)

    def to_npz_dict(self):
        """Flatten to the key/array mapping stored in pilot_results.npz."""
        out = {
            "fom_qois_master": self.fom_qois,
            "fom_times_master": self.fom_times,
            "fom_aux_corrs": self.fom_aux_corrs,
            "aux_aux_corrs": self.aux_aux_corrs,
            "normalized_aux_times": self.normalized_aux_times,
            "parameter_samples": self.parameter_samples,
            "training_dirs": self.training_dirs,
            "n_aux": self.n_aux,
            "n_active": self.n_active,
        }

        for i, aux_qois in enumerate(self.aux_qois_list):
            out[f"aux{i}_qois_master"] = aux_qois

        for t in range(self.n_active):
            out[f"rom{t}_fom_corrs"] = self.fom_rom_corrs_list[t]
            out[f"rom{t}_normalized_times"] = self.normalized_rom_times_list[t]
            for i in range(self.n_aux):
                out[f"aux{i}_rom{t}_corrs"] = self.aux_rom_corrs_list[i][t]

        if self.rom_rom_corrs:
            out["rom_rom_pairs"] = np.array(
                [[t, q] for (t, q, _, _, _) in self.rom_rom_corrs], dtype=int
            )
            for t, q, s_t_vals, s_q_vals, corr_vals in self.rom_rom_corrs:
                out[f"rom{t}_rom{q}_s_t"] = s_t_vals
                out[f"rom{t}_rom{q}_s_q"] = s_q_vals
                out[f"rom{t}_rom{q}_corrs"] = corr_vals

        return out

    @classmethod
    def from_npz(cls, data, n_aux, n_active):
        """
        Rebuild from an opened npz mapping. Inverse of `to_npz_dict`.

        n_aux/n_active are passed in rather than read back from the file so a
        mismatch between the run configuration and a stale pilot file surfaces
        here as a missing-key error rather than silently later.
        """
        return cls(
            fom_qois=data["fom_qois_master"],
            aux_qois_list=[data[f"aux{i}_qois_master"] for i in range(n_aux)],
            fom_aux_corrs=data["fom_aux_corrs"],
            aux_aux_corrs=data.get("aux_aux_corrs", np.array([])),
            fom_rom_corrs_list=[
                data[f"rom{t}_fom_corrs"] for t in range(n_active)
            ],
            aux_rom_corrs_list=[
                [data[f"aux{i}_rom{t}_corrs"] for t in range(n_active)]
                for i in range(n_aux)
            ],
            rom_rom_corrs=cls._rom_rom_from_npz(data),
            fom_times=data["fom_times_master"],
            normalized_aux_times=data["normalized_aux_times"],
            normalized_rom_times_list=[
                data[f"rom{t}_normalized_times"] for t in range(n_active)
            ],
            parameter_samples=data["parameter_samples"],
            training_dirs=data["training_dirs"],
        )

    @staticmethod
    def _rom_rom_from_npz(data):
        if "rom_rom_pairs" not in data:
            return []

        return [
            (
                int(t),
                int(q),
                data[f"rom{int(t)}_rom{int(q)}_s_t"],
                data[f"rom{int(t)}_rom{int(q)}_s_q"],
                data[f"rom{int(t)}_rom{int(q)}_corrs"],
            )
            for t, q in data["rom_rom_pairs"]
        ]


class PilotSampler:
    """Manages the pilot sampling workflow."""

    def __init__(
        self,
        fom_model,
        aux_models: List[QoiModel],
        rom_builders: List[QoiModel],
        param_space,
        pilot_mgr,
        work_dir,
    ):
        self.fom_model = fom_model
        self.aux_models = aux_models
        self.n_aux = len(aux_models)
        self.rom_builders = rom_builders
        self.n_active = len(rom_builders)
        self.param_space = param_space
        self.pilot_mgr = pilot_mgr
        self.work_dir = work_dir

    def run(self, max_combinations: int = 10, overwrite: bool = False) -> PilotData:
        """Execute pilot sampling workflow."""
        log_path = os.path.join(self.work_dir, "pilot_status.log")

        with open(log_path, "w", encoding="utf-8") as log_file:
            self._log(log_file, "Creating train and test labels")
            self.pilot_mgr.set_train_and_test_labels(max_groups=max_combinations)

            if self.n_active > 1:
                self._log(log_file, "Computing trainable-trainable correlation labels")
                self.pilot_mgr.set_ROM_correlation_labels()

            self._log(log_file, "Generating parameter samples")
            param_samples = self.param_space.generate_samples(
                self.pilot_mgr.num_pilot
            )
            param_names = self.param_space.get_names()

            self._log(log_file, "Sampling fixed models")
            fom_qois, fom_times = [], []
            aux_qois_list = [[] for _ in range(self.n_aux)]
            aux_times_list = [[] for _ in range(self.n_aux)]
            train_dirs = []

            for i, sample in enumerate(param_samples):
                print(f"===========  Sample {i} ============\n")
                params = dict(zip(param_names, sample))

                fom_dir = f"{self.work_dir}/pilot/fom/run_{i}"
                fom_qoi, fom_time = self._run_model(
                    self.fom_model, fom_dir, params, overwrite
                )
                fom_qois.append(fom_qoi)
                fom_times.append(fom_time)
                train_dirs.append(fom_dir)

                for j, aux_model in enumerate(self.aux_models):
                    aux_dir = f"{self.work_dir}/pilot/aux{j}/run_{i}"
                    aux_qoi, aux_time = self._run_model(
                        aux_model, aux_dir, params, overwrite
                    )
                    aux_qois_list[j].append(aux_qoi)
                    aux_times_list[j].append(aux_time)

            fom_qois = np.array(fom_qois)
            fom_times = np.array(fom_times)
            aux_qois_list = [np.array(q) for q in aux_qois_list]
            aux_times_list = [np.array(t) for t in aux_times_list]

            self._log(log_file, "Creating ROM bases")
            rom_models = self._build_roms(train_dirs)

            self._log(log_file, "Sampling ROMs on test parameters")
            rom_qois, rom_times = self._sample_roms(
                param_samples, param_names, rom_models, overwrite
            )

            self._log(log_file, "Computing pilot statistics")
            return self._compute_stats(
                fom_qois,
                fom_times,
                aux_qois_list,
                aux_times_list,
                rom_qois,
                rom_times,
                param_samples,
                train_dirs,
            )

    def _log(self, log_file, message: str):
        """Write message to log file and console."""
        log_file.write(f"{message}\n")
        print(f"{message}\n")

    def _run_model(self, model, run_dir, params, overwrite):
        """Run a model and compute QoI and runtime."""
        return run_model_sample(model, run_dir, params, overwrite, verbose=True)

    def _build_roms(self, train_dirs):
        """Build ROM models for every trainable ROM's basis-size grid.

        Returns rom_models[t][i] = list of built ROM objects, one per
        training combo, for ROM t's i-th basis-size grid point.
        """
        rom_models = []

        for t, rom_builder in enumerate(self.rom_builders):
            rom_models_t = []

            for idx, basis_size in enumerate(self.pilot_mgr.s_lists[t]):
                print(f"ROM {t}, basis size {basis_size}\n")
                base_dir = f"{self.work_dir}/pilot/rom{t}/basis_size_{basis_size}"
                train_labels = self.pilot_mgr.train_labels[t][idx]
                models = []

                for train_label in train_labels:
                    print(f"Training ROM {t} from samples {train_label}\n")
                    combo_id = "-".join(str(i) for i in train_label)
                    offline_dir = os.path.join(
                        base_dir, f"combination_{combo_id}"
                    )
                    create_empty_dir(offline_dir)

                    rom = rom_builder.build_from_training_dirs(
                        offline_dir, [train_dirs[i] for i in train_label]
                    )
                    models.append(rom)

                rom_models_t.append(models)

            rom_models.append(rom_models_t)

        return rom_models

    def _sample_roms(self, param_samples, param_names, rom_models, overwrite):
        """Sample every trainable ROM's models on its own test parameters.

        Returns all_qois[t][i] / all_times[t][i] = np.ndarray of shape
        [NpCs, Np-s] for ROM t's i-th basis-size grid point.
        """
        all_qois, all_times = [], []

        for t in range(self.n_active):
            print(f"===========  ROM {t} ============\n")
            qois_t, times_t = [], []

            for i, basis_size in enumerate(self.pilot_mgr.s_lists[t]):
                print(f"ROM {t}, basis size {basis_size}\n")
                base_dir = f"{self.work_dir}/pilot/rom{t}/basis_size_{basis_size}"
                test_labels = self.pilot_mgr.test_labels[t][i]
                train_labels = self.pilot_mgr.train_labels[t][i]

                qois_i, times_i = [], []

                for j, test_label in enumerate(test_labels):
                    print(f"Testing ROM {t} built from samples {train_labels[j]}\n")
                    combo_id = "-".join(str(k) for k in train_labels[j])
                    rom_dir = os.path.join(base_dir, f"combination_{combo_id}")
                    rom = rom_models[t][i][j]

                    qois_ij, times_ij = [], []

                    for sample_idx in test_label:
                        print(f"Testing on sample {sample_idx}\n")
                        params = dict(zip(param_names, param_samples[sample_idx]))
                        run_dir = os.path.join(rom_dir, f"run_test_sample_{sample_idx}")
                        qoi, runtime = self._run_model(rom, run_dir, params, overwrite)
                        qois_ij.append(qoi)
                        times_ij.append(runtime)

                    qois_i.append(qois_ij)
                    times_i.append(times_ij)

                qois_t.append(np.array(qois_i))
                times_t.append(np.array(times_i))

            all_qois.append(qois_t)
            all_times.append(times_t)

        return all_qois, all_times

    def _compute_stats(
        self,
        fom_qois,
        fom_times,
        aux_qois_list,
        aux_times_list,
        rom_qois,
        rom_times,
        param_samples,
        train_dirs,
    ):
        """Compute correlation and cost statistics and save results."""
        # FOM-aux correlations (unaffected by ROM count)
        fom_aux_corrs = []
        for aux_qois in aux_qois_list:
            corr = self.pilot_mgr.estimate_pairwise_correlations(
                [fom_qois[None, :]], [aux_qois[None, :]]
            )[0]
            fom_aux_corrs.append(corr)
        fom_aux_corrs = np.array(fom_aux_corrs)

        # Aux-aux correlations (6), stored in model_indices.aux_pairs order.
        aux_aux_corrs = []
        for _, i, j in aux_pairs(self.n_aux):
            corr = self.pilot_mgr.estimate_pairwise_correlations(
                [aux_qois_list[j][None, :]], [aux_qois_list[i][None, :]]
            )[0]
            aux_aux_corrs.append(corr)
        aux_aux_corrs = np.array(aux_aux_corrs) if aux_aux_corrs else np.array([])

        # FOM-ROM / aux-ROM correlations and normalized times, per trainable
        # ROM t. Each ROM has its own basis-size grid, so the FOM/aux data
        # must be reshaped against that ROM's own test_labels[t] before
        # comparing against rom_qois[t].
        fom_rom_corrs_list = []
        aux_rom_corrs_per_rom = []  # [t][i] = corr array for aux i vs ROM t
        norm_rom_times_list = []

        for t in range(self.n_active):
            fom_q_t = self._reshape_data(fom_qois, t)
            fom_rom_corrs_list.append(
                self.pilot_mgr.estimate_pairwise_correlations(fom_q_t, rom_qois[t])
            )

            aux_q_t = [self._reshape_data(aux_qois, t) for aux_qois in aux_qois_list]
            aux_rom_corrs_per_rom.append([
                self.pilot_mgr.estimate_pairwise_correlations(aux_q, rom_qois[t])
                for aux_q in aux_q_t
            ])

            norm_rom_times_list.append(
                self._normalized_rom_times(rom_times[t], fom_times, t)
            )

        # Transpose to aux_rom_corrs_list[i][t] (outer index = aux model).
        aux_rom_corrs_list = [
            [aux_rom_corrs_per_rom[t][i] for t in range(self.n_active)]
            for i in range(self.n_aux)
        ]

        # Trainable-trainable (ROM-ROM) correlations: empty when
        # n_active == 1, since no ROM pairs exist.
        rom_rom_corrs = (
            self._compute_rom_rom_corrs(rom_qois) if self.n_active > 1 else []
        )

        # Normalized aux times (unaffected by ROM count)
        norm_aux_times = [np.mean(aux_times / fom_times) for aux_times in aux_times_list]

        pilot_data = PilotData(
            fom_qois=fom_qois,
            aux_qois_list=aux_qois_list,
            fom_aux_corrs=fom_aux_corrs,
            aux_aux_corrs=aux_aux_corrs,
            fom_rom_corrs_list=fom_rom_corrs_list,
            aux_rom_corrs_list=aux_rom_corrs_list,
            rom_rom_corrs=rom_rom_corrs,
            fom_times=fom_times,
            normalized_aux_times=np.array(norm_aux_times),
            normalized_rom_times_list=norm_rom_times_list,
            parameter_samples=param_samples,
            training_dirs=np.array(train_dirs),
        )

        np.savez(
            f"{self.work_dir}/pilot_results.npz", **pilot_data.to_npz_dict()
        )

        return pilot_data

    def _normalized_rom_times(self, rom_times_t, fom_times, t):
        """
        Normalized cost of trainable estimator t at each of its pilot basis
        sizes, per writeup Eq. (9):

            w_tilde_omega(s_omega) = (1/B) sum_b [ (1/n_b) sum_{j in V_b}
                                        w_omega,j(s_omega) / w_0,j ]

        Each ROM evaluation cost is divided by the FOM cost *at the same
        pilot parameter j*, then averaged over that replicate's validation
        set V_b, then over replicates.

        The previous implementation zipped rom_times_t (indexed by
        basis-grid point) against fom_times (indexed by pilot sample), so
        grid point g was normalized by the runtime of the g-th pilot FOM run
        -- a single arbitrary sample rather than the matched one, and
        silently truncated to min(G, Np) grid points. The fixed-model cost
        (5) in _compute_stats was, and remains, matched per sample; the two
        estimators now agree.

        rom_times_t : list over grid points g of arrays [n_combos, |V|],
            column order matching pilot_mgr.test_labels[t][g][combo].
        fom_times   : (Np,) high-fidelity runtimes w_0,j.
        """
        fom_times = np.asarray(fom_times, dtype=float).reshape(-1)
        test_labels_t = self.pilot_mgr.test_labels[t]

        normalized = []

        for g, times_g in enumerate(rom_times_t):
            replicate_means = []

            for combo, validation in enumerate(test_labels_t[g]):
                w0 = fom_times[np.asarray(validation, dtype=int)]
                w_rom = np.asarray(times_g[combo], dtype=float).reshape(-1)

                replicate_means.append(float(np.mean(w_rom / w0)))

            normalized.append(float(np.mean(replicate_means)))

        return np.array(normalized)

    def _compute_rom_rom_corrs(self, rom_qois):
        """
        Build trainable-trainable correlation triples for every ROM pair
        (t, q), t < q, from the matched replicates of
        Pilot.set_ROM_correlation_labels (whose docstring covers how this
        differs from Algorithm 5).

        Returns a list of (t, q, s_t_vals, s_q_vals, corr_vals) tuples,
        one per ROM pair that had at least one surviving triple. s_t_vals,
        s_q_vals, corr_vals are parallel 1-D arrays (ragged across pairs,
        since B_t and B_q need not be the same size).
        """
        owner = self.pilot_mgr.rom_correlation_owner
        indices = self.pilot_mgr.ROM_correlation_indices

        triples_by_pair = {}

        for pos_a, (rom_a, i_a, s_a) in enumerate(owner):
            for pos_b, (rom_b, i_b, s_b) in enumerate(owner[:pos_a]):
                if rom_a == rom_b:
                    continue  # within-ROM basis-size pair, not a ROM-ROM pair

                # Flattened owner order is ROM-block order, so rom_b <= rom_a
                # always; since rom_a != rom_b here, rom_b < rom_a strictly.
                t, q = rom_b, rom_a
                s_t, s_q = s_b, s_a
                i_t, i_q = i_b, i_a  # grid indices within ROM t's / q's own grids

                combo_val_pairs = indices[pos_a][pos_b]
                if not combo_val_pairs:
                    continue

                corr = self.pilot_mgr.estimate_ROM_ROM_correlation(
                    rom_qois[t][i_t],
                    self.pilot_mgr.test_labels[t][i_t],
                    rom_qois[q][i_q],
                    self.pilot_mgr.test_labels[q][i_q],
                    combo_val_pairs,
                )
                if np.isnan(corr):
                    continue

                triples_by_pair.setdefault((t, q), []).append((s_t, s_q, corr))

        rom_rom_corrs = []
        for (t, q), triples in sorted(triples_by_pair.items()):
            s_t_vals = np.array([tr[0] for tr in triples])
            s_q_vals = np.array([tr[1] for tr in triples])
            corr_vals = np.array([tr[2] for tr in triples])
            rom_rom_corrs.append((t, q, s_t_vals, s_q_vals, corr_vals))

        return rom_rom_corrs

    def _reshape_data(self, data, rom_idx=0):
        """Reshape data by test labels, for one trainable ROM, to
        [NpCs, Np-s, ...]."""
        return [
            np.array([[data[idx] for idx in group] for group in test])
            for test in self.pilot_mgr.test_labels[rom_idx]
        ]
