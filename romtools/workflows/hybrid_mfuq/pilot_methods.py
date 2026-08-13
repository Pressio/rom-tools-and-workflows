import os
import time
from dataclasses import dataclass
from typing import List
from math import comb
from itertools import combinations

import numpy as np
import scipy.stats

from romtools.workflows.models import QoiModel
from romtools.workflows.workflow_utils import create_empty_dir


def run_model_sample(model, run_dir, params, overwrite=False, verbose=False):
    """
    Run a model sample, using cached qoi.txt/time.txt if available.

    Shared by `PilotSampler._run_model` and `run_hybrid_mfuq.py`'s
    orchestration code, which previously each carried a near-identical
    copy of this cache-check + populate + run + compute_qoi sequence.

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


class Pilot:
    """
    Abstract class which handles pilot sampling.
    Includes strategies to estimate correlations and costs.
    """

    def __init__(self, s_list, num_pilot, random_seed=2025):
        self.s_list = s_list
        self.num_pilot = num_pilot
        self.rng = np.random.default_rng(random_seed)

    def set_train_and_test_labels(self, max_groups=int(1e6)):
        """
        Create train/test index groups for pilot sampling.
        """
        pilot_set = set(range(self.num_pilot))
        train_labels = [0 for _ in self.s_list]
        test_labels = [0 for _ in self.s_list]

        for i, s in enumerate(self.s_list):
            NpCs = comb(self.num_pilot, s)

            if NpCs <= max_groups:
                train_labels[i] = [
                    list(c) for c in combinations(range(self.num_pilot), s)
                ]
            else:
                train_labels[i] = []
                seen = set()
                while len(train_labels[i]) < max_groups:
                    label = sorted(
                        self.rng.choice(range(self.num_pilot), s, replace=False)
                    )
                    tup = tuple(label)
                    if tup not in seen:
                        seen.add(tup)
                        train_labels[i].append(label)

            test_labels[i] = [
                list(pilot_set - set(train_labels[i][k]))
                for k in range(len(train_labels[i]))
            ]

        self.train_labels = train_labels
        self.test_labels = test_labels

    def split_data_using_labels(self, data_list):
        """
        Split data arrays using train/test labels.

        Returns lists of arrays shaped [NpCs, s, ...] for train and
        [NpCs, Np-s, ...] for test data.
        """
        train_data, test_data = [], []

        for data in data_list:
            train_data_i, test_data_i = [], []
            for train, test in zip(self.train_labels, self.test_labels):
                train_data_i.append(data[train])
                test_data_i.append(data[test])
            train_data.append(train_data_i)
            test_data.append(test_data_i)

        return train_data, test_data

    def estimate_FOM_correlations(self, X_test_list, y_test_list):
        """
        Estimate correlations between models by averaging Pearson correlations
        over pilot groups.
        """
        corr_s = []

        for X, y in zip(X_test_list, y_test_list):
            out, _ = pearsonr_with_axis(X, y, axis=1)
            corr_s.append(np.mean(out, axis=0).squeeze())

        return np.array(corr_s)

    def set_ROM_correlation_labels(self, seed=2025):
        """
        Compute shared test indices usable for ROM–ROM correlations.
        """
        np.random.seed(seed)
        indices = []

        for i, label_i in enumerate(self.test_labels):
            indices_i = []
            NpCs_i = len(label_i)

            for j, label_j in enumerate(self.test_labels[:i]):
                NpCs_j = len(label_j)
                m = min(NpCs_i, NpCs_j)
                random_ij = np.random.permutation(m)

                indices_i.append(
                    [
                        list(set(label_i[k]) & set(label_j[k]))
                        for k in random_ij
                    ]
                )

            indices.append(indices_i)

        self.ROM_correlation_indices = indices


@dataclass
class PilotData:
    """Container for pilot sampling results."""
    fom_qois: np.ndarray
    aux_qois_list: List[np.ndarray]
    fom_aux_corrs: np.ndarray
    aux_aux_corrs: np.ndarray
    fom_rom_corrs: np.ndarray
    aux_rom_corrs_list: List[np.ndarray]
    fom_times: np.ndarray
    normalized_aux_times: np.ndarray
    normalized_rom_times: np.ndarray
    parameter_samples: np.ndarray
    training_dirs: np.ndarray


class PilotSampler:
    """Manages the pilot sampling workflow."""

    def __init__(
        self,
        fom_model,
        aux_models: List[QoiModel],
        rom_builder,
        param_space,
        pilot_mgr,
        work_dir,
    ):
        self.fom_model = fom_model
        self.aux_models = aux_models
        self.n_aux = len(aux_models)
        self.rom_builder = rom_builder
        self.param_space = param_space
        self.pilot_mgr = pilot_mgr
        self.work_dir = work_dir

    def run(self, max_combinations: int = 10, overwrite: bool = False) -> PilotData:
        """Execute pilot sampling workflow."""
        log_path = os.path.join(self.work_dir, "pilot_status.log")

        with open(log_path, "w", encoding="utf-8") as log_file:
            self._log(log_file, "Creating train and test labels")
            self.pilot_mgr.set_train_and_test_labels(max_groups=max_combinations)

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
        rom_models = []

        for idx, basis_size in enumerate(self.pilot_mgr.s_list):
            print(f"Basis size {basis_size}\n")
            base_dir = f"{self.work_dir}/pilot/rom/basis_size_{basis_size}"
            train_labels = self.pilot_mgr.train_labels[idx]
            models = []

            for train_label in train_labels:
                print(f"Training ROM from samples {train_label}\n")
                combo_id = "-".join(str(i) for i in train_label)
                offline_dir = os.path.join(
                    base_dir, f"combination_{combo_id}"
                )
                create_empty_dir(offline_dir)

                rom = self.rom_builder.build_from_training_dirs(
                    offline_dir, [train_dirs[i] for i in train_label]
                )
                models.append(rom)

            rom_models.append(models)

        return rom_models

    def _sample_roms(self, param_samples, param_names, rom_models, overwrite):
        all_qois, all_times = [], []

        for i, basis_size in enumerate(self.pilot_mgr.s_list):
            print(f"Basis size {basis_size}\n")
            base_dir = f"{self.work_dir}/pilot/rom/basis_size_{basis_size}"
            test_labels = self.pilot_mgr.test_labels[i]
            train_labels = self.pilot_mgr.train_labels[i]

            qois_i, times_i = [], []

            for j, test_label in enumerate(test_labels):
                print(f"Testing ROM built from samples {train_labels[j]}\n")
                combo_id = "-".join(str(k) for k in train_labels[j])
                rom_dir = os.path.join(base_dir, f"combination_{combo_id}")
                rom = rom_models[i][j]

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

            all_qois.append(np.array(qois_i))
            all_times.append(np.array(times_i))

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
        fom_q = self._reshape_data(fom_qois)
        aux_q_list = [self._reshape_data(aux_qois) for aux_qois in aux_qois_list]

        # FOM-aux correlations
        fom_aux_corrs = []
        for aux_qois in aux_qois_list:
            corr = self.pilot_mgr.estimate_FOM_correlations(
                [fom_qois[None, :]], [aux_qois[None, :]]
            )[0]
            fom_aux_corrs.append(corr)
        fom_aux_corrs = np.array(fom_aux_corrs)

        # Aux-aux correlations
        aux_aux_corrs = []
        if self.n_aux > 1:
            for i in range(self.n_aux):
                for j in range(i + 1, self.n_aux):
                    corr = self.pilot_mgr.estimate_FOM_correlations(
                        [aux_qois_list[i][None, :]], [aux_qois_list[j][None, :]]
                    )[0]
                    aux_aux_corrs.append(corr)
        aux_aux_corrs = np.array(aux_aux_corrs) if aux_aux_corrs else np.array([])

        # FOM-ROM correlations
        fom_rom_corrs = self.pilot_mgr.estimate_FOM_correlations(fom_q, rom_qois)

        # Aux-ROM correlations
        aux_rom_corrs_list = [
            self.pilot_mgr.estimate_FOM_correlations(aux_q, rom_qois)
            for aux_q in aux_q_list
        ]

        # Normalized times
        norm_aux_times = [np.mean(aux_times / fom_times) for aux_times in aux_times_list]
        norm_rom_times = [np.mean(rt / ft) for rt, ft in zip(rom_times, fom_times)]

        pilot_data = PilotData(
            fom_qois=fom_qois,
            aux_qois_list=aux_qois_list,
            fom_aux_corrs=fom_aux_corrs,
            aux_aux_corrs=aux_aux_corrs,
            fom_rom_corrs=fom_rom_corrs,
            aux_rom_corrs_list=aux_rom_corrs_list,
            fom_times=fom_times,
            normalized_aux_times=np.array(norm_aux_times),
            normalized_rom_times=np.array(norm_rom_times),
            parameter_samples=param_samples,
            training_dirs=np.array(train_dirs),
        )

        # Save to npz
        save_dict = {
            "fom_qois_master": fom_qois,
            "fom_times_master": fom_times,
            "fom_aux_corrs": fom_aux_corrs,
            "aux_aux_corrs": aux_aux_corrs,
            "fom_rom_corrs": fom_rom_corrs,
            "normalized_aux_times": norm_aux_times,
            "normalized_rom_times": norm_rom_times,
            "parameter_samples": param_samples,
            "training_dirs": train_dirs,
            "n_aux": self.n_aux,
        }

        for i, aux_qois in enumerate(aux_qois_list):
            save_dict[f"aux{i}_qois_master"] = aux_qois
        for i, aux_rom_corrs in enumerate(aux_rom_corrs_list):
            save_dict[f"aux{i}_rom_corrs"] = aux_rom_corrs

        np.savez(f"{self.work_dir}/pilot_results.npz", **save_dict)

        return pilot_data

    def _reshape_data(self, data):
        """Reshape data by test labels to [NpCs, Np-s, ...]."""
        return [
            np.array([[data[idx] for idx in group] for group in test])
            for test in self.pilot_mgr.test_labels
        ]
