import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional

import numpy as np
import torch

import romtools.workflows.hybrid_mfuq.mfuq_methods as mfmc
import romtools.workflows.hybrid_mfuq.surrogate_methods as surg
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.model_builders import QoiModelBuilder

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@dataclass
class PilotData:
    """Container for pilot sampling results."""
    fom_qois: np.ndarray
    aux_qois: np.ndarray
    fom_aux_corr: float
    fom_rom_corrs: np.ndarray
    aux_rom_corrs: np.ndarray
    fom_times: np.ndarray
    normalized_aux_time: float
    normalized_rom_times: np.ndarray
    parameter_samples: np.ndarray
    training_dirs: np.ndarray


class PilotSampler:
    """Manages pilot sampling workflow."""
    
    def __init__(self, fom_model, aux_model, rom_builder, param_space, pilot_mgr, work_dir):
        self.fom_model = fom_model
        self.aux_model = aux_model
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
            param_samples = self.param_space.generate_samples(self.pilot_mgr.num_pilot)
            param_names = self.param_space.get_names()
            
            # Sample FOM and auxiliary models
            self._log(log_file, "Sampling fixed models")
            fom_qois, fom_times, aux_qois, aux_times, train_dirs = [], [], [], [], []
            
            for i, sample in enumerate(param_samples):
                print(f"===========  Sample {i} ============\n")
                params = dict(zip(param_names, sample))
                fom_dir = f'{self.work_dir}/pilot/fom/run_{i}'
                aux_dir = f'{self.work_dir}/pilot/aux/run_{i}'
                
                fom_qoi, fom_time = self._run_model(self.fom_model, fom_dir, params, overwrite)
                aux_qoi, aux_time = self._run_model(self.aux_model, aux_dir, params, overwrite)
                
                fom_qois.append(fom_qoi)
                fom_times.append(fom_time)
                aux_qois.append(aux_qoi)
                aux_times.append(aux_time)
                train_dirs.append(fom_dir)
            
            fom_qois = np.array(fom_qois)
            fom_times = np.array(fom_times)
            aux_qois = np.array(aux_qois)
            aux_times = np.array(aux_times)
            
            # Build and sample ROMs
            self._log(log_file, "Creating ROM bases")
            rom_models = self._build_roms(train_dirs)
            
            self._log(log_file, "Sampling ROMs on test parameters")
            rom_qois, rom_times = self._sample_roms(param_samples, param_names, rom_models, overwrite)
            
            # Compute statistics
            self._log(log_file, "Computing pilot statistics")
            return self._compute_stats(fom_qois, fom_times, aux_qois, aux_times,
                                       rom_qois, rom_times, param_samples, train_dirs)
    
    def _log(self, log_file, message: str):
        """Write message to log file and console."""
        log_file.write(f"{message}\n")
        print(f"{message}\n")
    
    def _run_model(self, model, run_dir, params, overwrite):
        """Run model and compute QoI with timing."""
        qoi_path = os.path.join(run_dir, "qoi.txt")
        time_path = os.path.join(run_dir, "time.txt")
        
        if not overwrite and os.path.exists(qoi_path) and os.path.exists(time_path):
            print("Reading in QoI value and runtime\n")
            return np.loadtxt(qoi_path), np.loadtxt(time_path)
        
        print("Computing QoI value and runtime\n")
        create_empty_dir(run_dir)
        model.populate_run_directory(run_dir, params)
        
        passed_file = os.path.join(run_dir, "passed.txt")
        if os.path.isfile(passed_file) and not overwrite:
            print("Skipping (Sample has already run successfully)\n")
        else:
            print("Running...\n")
            
        t0 = time.time()
        code = model.run_model(run_dir, params)
        qoi = model.compute_qoi(run_dir, params)
        runtime = time.time() - t0
        
        if code == 0:
            np.savetxt(passed_file, [0], fmt="%i")
        np.savetxt(qoi_path, [qoi])
        np.savetxt(time_path, [runtime])
        
        return np.array(qoi), np.array(runtime)
    
    def _build_roms(self, train_dirs):
        """Build ROM models for all pilot basis sizes."""
        rom_models = []
        for idx, basis_size in enumerate(self.pilot_mgr.s_list):
            print(f"Basis size {basis_size}\n")
            base_dir = f'{self.work_dir}/pilot/rom/basis_size_{basis_size}'
            train_labels = self.pilot_mgr.train_labels[idx]
            models = []
            
            for train_label in train_labels:
                print(f"Training ROM from samples {train_label}\n")
                combo_id = '-'.join(str(i) for i in train_label)
                offline_dir = os.path.join(base_dir, f'combination_{combo_id}')
                create_empty_dir(offline_dir)
                
                rom = self.rom_builder.build_from_training_dirs(
                    offline_dir, [train_dirs[i] for i in train_label]
                )
                models.append(rom)
            
            rom_models.append(models)
        return rom_models
    
    def _sample_roms(self, param_samples, param_names, rom_models, overwrite):
        """Sample ROM models on test parameters."""
        all_qois, all_times = [], []
        
        for i, basis_size in enumerate(self.pilot_mgr.s_list):
            print(f"Basis size {basis_size}\n")
            base_dir = f'{self.work_dir}/pilot/rom/basis_size_{basis_size}'
            test_labels = self.pilot_mgr.test_labels[i]
            train_labels = self.pilot_mgr.train_labels[i]
            qois_i, times_i = [], []
            
            for j, test_label in enumerate(test_labels):
                print(f"Testing ROM built from samples {train_labels[j]}\n")
                combo_id = '-'.join(str(k) for k in train_labels[j])
                rom_dir = os.path.join(base_dir, f'combination_{combo_id}')
                rom = rom_models[i][j]
                
                qois_ij, times_ij = [], []
                for k, sample_idx in enumerate(test_label):
                    print(f"Testing on sample {sample_idx}\n")
                    params = dict(zip(param_names, param_samples[sample_idx]))
                    run_dir = os.path.join(rom_dir, f'run_test_sample_{sample_idx}')
                    
                    qoi, runtime = self._run_model(rom, run_dir, params, overwrite)
                    qois_ij.append(qoi)
                    times_ij.append(runtime)
                
                qois_i.append(qois_ij)
                times_i.append(times_ij)
            
            all_qois.append(np.array(qois_i))
            all_times.append(np.array(times_i))
        
        return all_qois, all_times
    
    def _compute_stats(self, fom_qois, fom_times, aux_qois, aux_times,
                      rom_qois, rom_times, param_samples, train_dirs):
        """Compute correlation and cost statistics."""
        # Reshape for comparison
        fom_q = self._reshape_data(fom_qois)
        aux_q = self._reshape_data(aux_qois)
        fom_t = self._reshape_data(fom_times)
        
        # Compute correlations
        fom_aux_corr = self.pilot_mgr.estimate_FOM_correlations(
            [fom_qois[None, :]], [aux_qois[None, :]]
        )[0]
        fom_rom_corrs = self.pilot_mgr.estimate_FOM_correlations(fom_q, rom_qois)
        aux_rom_corrs = self.pilot_mgr.estimate_FOM_correlations(aux_q, rom_qois)
        
        # Compute normalized times
        norm_aux_time = np.mean(aux_times / fom_times)
        norm_rom_times = [np.mean(rt / ft) for rt, ft in zip(rom_times, fom_t)]
        
        # Save and return
        pilot_data = PilotData(
            fom_qois=fom_qois, aux_qois=aux_qois,
            fom_aux_corr=fom_aux_corr,
            fom_rom_corrs=fom_rom_corrs, aux_rom_corrs=aux_rom_corrs,
            fom_times=fom_times,
            normalized_aux_time=norm_aux_time,
            normalized_rom_times=np.array(norm_rom_times),
            parameter_samples=param_samples,
            training_dirs=np.array(train_dirs)
        )
        
        np.savez(f"{self.work_dir}/pilot_results.npz",
                fom_qois_master=fom_qois, aux_qois_master=aux_qois,
                fom_aux_corr=fom_aux_corr,
                fom_rom_corrs=fom_rom_corrs, aux_rom_corrs=aux_rom_corrs,
                fom_times_master=fom_times,
                normalized_aux_time=norm_aux_time,
                normalized_rom_times=norm_rom_times,
                parameter_samples=param_samples, training_dirs=train_dirs)
        
        return pilot_data
    
    def _reshape_data(self, data):
        """Reshape data by test labels."""
        return [np.array([[data[idx] for idx in group] for group in test])
                for test in self.pilot_mgr.test_labels]


class SurrogateBuilder:
    """Builds surrogate models for correlation and cost functions."""
    
    def __init__(self, pilot_list, n_active, work_dir=None, method='neural_network'):
        self.pilot_list = pilot_list
        self.n_active = n_active
        self.work_dir = work_dir
        self.method = method
        self.model_path = os.path.join(work_dir, "vecl_correlation_model.pt") if work_dir else None
        
        if method not in ['neural_network', 'sigmoid']:
            raise ValueError("method must be 'neural_network' or 'sigmoid'")
    
    def build(self, data_npz):
        """Build surrogate models from pilot data."""
        with np.load(data_npz) as data:
            fom_aux_corr = float(data['fom_aux_corr'])
            fom_rom_corrs = data['fom_rom_corrs']
            aux_rom_corrs = data['aux_rom_corrs']
            norm_aux_time = float(data['normalized_aux_time'])
            norm_rom_times = data['normalized_rom_times']
        
        if self.method == 'sigmoid':
            return self._build_sigmoid(fom_aux_corr, fom_rom_corrs, aux_rom_corrs,
                                      norm_aux_time, norm_rom_times)
        else:
            return self._build_neural_net(fom_aux_corr, fom_rom_corrs, aux_rom_corrs,
                                         norm_aux_time, norm_rom_times)
    
    def _wrap(self, func):
        """Wrap function to extract s[-1] from full s vector."""
        return lambda s: func(s[-1] if isinstance(s, (list, tuple, np.ndarray)) and len(s) >= self.n_active else s)
    
    def _build_sigmoid(self, fom_aux_corr, fom_rom_corrs, aux_rom_corrs, norm_aux_time, norm_rom_times):
        """Build using sigmoid fitting."""
        pilots = np.array(self.pilot_list)
        rho13_surr = surg.fit_sigmoid(pilots[None, :], fom_rom_corrs)
        rho23_surr = surg.fit_sigmoid(pilots[None, :], aux_rom_corrs)
        cost3_surr = surg.fit_polynomial(pilots[None, :], norm_rom_times, order=1)
        
        return (
            [lambda s: fom_aux_corr, self._wrap(lambda s: float(rho13_surr(s)))],
            [self._wrap(lambda s: float(rho23_surr(s)))],
            [lambda s: norm_aux_time, self._wrap(lambda s: float(cost3_surr(s)))]
        )
    
    def _build_neural_net(self, fom_aux_corr, fom_rom_corrs, aux_rom_corrs, norm_aux_time, norm_rom_times):
        """Build using neural network."""
        model = self._train_or_load_model(fom_aux_corr, fom_rom_corrs, aux_rom_corrs)
        rho12, rho13, rho23 = self._create_corr_funcs(model, fom_aux_corr)
        cost3_surr = surg.fit_polynomial(np.array(self.pilot_list)[None, :], norm_rom_times, order=1)
        
        return (
            [lambda s: rho12(0), self._wrap(rho13)],
            [self._wrap(rho23)],
            [lambda s: norm_aux_time, self._wrap(lambda s: float(cost3_surr(s)))]
        )
    
    def _train_or_load_model(self, fom_aux_corr, fom_rom_corrs, aux_rom_corrs):
        """Train or load neural network model."""
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading existing VeclNet model from {self.model_path}\n")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if (checkpoint.get('pilot_list') == self.pilot_list and
                checkpoint.get('n_active') == self.n_active):
                model = surg.VeclNet(1, 1, 3)
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total trainable parameters: {trainable_params}")
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                return model
            print(f"Warning: Saved model incompatible, retraining\n")
            os.remove(self.model_path)
        
        # Train new model
        print("Training new VeclNet correlation model\n")
        ins = torch.tensor(self.pilot_list, dtype=torch.float32).reshape(-1, 1)
        half = torch.stack([
            torch.full((len(self.pilot_list),), float(fom_aux_corr)),
            torch.tensor(fom_rom_corrs, dtype=torch.float32),
            torch.tensor(aux_rom_corrs, dtype=torch.float32)
        ], dim=1)
        
        matrices = surg.to_symmetric_tracefree_batch(half, 3)
        matrices += torch.diag_embed(torch.ones(ins.shape[0], 3))
        
        model = surg.VeclNet(1, 1, 3)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {trainable_params}")

        model, _ = surg.train_model(model, ins, matrices, n=3, lr=1e-1,
                                   max_steps=2000, tol=1e-8, print_every=50)
        model.eval()
        
        if self.model_path:
            torch.save({'model_state_dict': model.state_dict(),
                       'pilot_list': self.pilot_list,
                       'n_active': self.n_active}, self.model_path)
            print(f"Saved VeclNet model to {self.model_path}\n")
        
        return model
    
    def _create_corr_funcs(self, model, fom_aux_corr):
        """Create correlation functions with validity guarantee."""
        # Generate augmented data
        s_aug = np.unique(np.concatenate([
            self.pilot_list,
            np.linspace(min(self.pilot_list), max(self.pilot_list), 20)
        ]))
        
        with torch.no_grad():
            corr_mats = model.corr_matrix(torch.tensor(s_aug, dtype=torch.float32).reshape(-1, 1), 3)
        
        ridx, cidx = torch.tril_indices(3, 3, offset=-1)
        rho12_val = float(fom_aux_corr)
        rho13_aug = corr_mats[:, ridx[1], cidx[1]].numpy()
        rho23_aug = corr_mats[:, ridx[2], cidx[2]].numpy()
        
        # Fit sigmoid surrogates
        rho13_sig = surg.fit_sigmoid(s_aug[None, :], rho13_aug)
        rho23_sig = surg.fit_sigmoid(s_aug[None, :], rho23_aug)
        
        def project(r12, r13, r23):
            """Project to nearest valid correlation matrix."""
            C = np.array([[1, r12, r13], [r12, 1, r23], [r13, r23, 1]])
            eigvals, eigvecs = np.linalg.eigh(C)
            C_valid = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
            d = np.sqrt(np.diag(C_valid))
            return (C_valid / np.outer(d, d))[[0, 0, 1], [1, 2, 2]]
        
        return (
            lambda s: rho12_val,
            lambda s: project(rho12_val, float(rho13_sig(s)), float(rho23_sig(s)))[1],
            lambda s: project(rho12_val, float(rho13_sig(s)), float(rho23_sig(s)))[2]
        )


def optimize_allocation(budget_list, hf_corrs, lf_corrs, costs, bounds, log_obj, hybrid=False):
    """Run MFUQ optimization for multiple budgets."""
    mf_vars, mf_allocs = [], []
    is_vars, is_allocs = [], []
    
    for budget in budget_list:
        for typ, v_list, a_list in [('MF', mf_vars, mf_allocs), ('IS', is_vars, is_allocs)]:
            obj = mfmc.MFMC(budget, typ, hybrid=hybrid)
            obj.set_corrs_and_costs(hf_corrs, lf_corrs, costs)
            obj.set_objective_and_constraint(log=log_obj, bounds=bounds)
            
            best_fval, best_x = float('inf'), None
            for _ in range(50):
                obj.solve()
                if obj.result.success:
                    fval = np.exp(obj.result.fun) if log_obj else obj.result.fun
                    if 0 <= fval < best_fval:
                        best_fval, best_x = fval, obj.result.x
            
            print(f'Variance ratio for {typ} at budget {budget}: {best_fval} at {best_x}\n')
            v_list.append(best_fval)
            a_list.append(best_x)
    
    return (mf_vars, mf_allocs), (is_vars, is_allocs)


def train_optimized_rom(fom_model, rom_builder, param_space, work_dir, rom_basis_num,
                       pilot_mgr, param_samples, train_dirs, data_npz, overwrite=False):
    """Train optimized ROM and compute statistics."""
    print(f"Training ROM from first {rom_basis_num} samples\n")
    
    # Generate additional samples if needed
    if len(train_dirs) < rom_basis_num:
        print("Sampling extra FOMs for training\n")
        num_extra = rom_basis_num - len(train_dirs)
        extra = param_space.generate_samples(num_extra)
        param_names = param_space.get_names()
        sampler = PilotSampler(fom_model, None, None, param_space, None, work_dir)
        
        for i, sample in enumerate(extra):
            print(f"===========  Sample {i} ============\n")
            idx = len(train_dirs) + i
            fom_dir = f'{work_dir}/pilot/fom/run_{idx}'
            params = dict(zip(param_names, sample))
            sampler._run_model(fom_model, fom_dir, params, overwrite)
            train_dirs.append(fom_dir)
    
    # Build ROM
    print(f"Building ROM with basis size {rom_basis_num}\n")
    rom_dir = f'{work_dir}/pilot/rom_optimized/basis_size_{rom_basis_num}'
    create_empty_dir(rom_dir)
    rom_model = rom_builder.build_from_training_dirs(rom_dir, train_dirs[:rom_basis_num])
    
    # Sample on pilot parameters
    print("Computing pilot QoIs for trained ROM\n")
    param_names = param_space.get_names()
    sampler = PilotSampler(fom_model, None, None, param_space, None, work_dir)
    
    rom_qois, rom_times = [], []
    for i, sample in enumerate(param_samples):
        print(f"===========  Sample {i} ============\n")
        params = dict(zip(param_names, sample))
        run_dir = os.path.join(rom_dir, f'run_{i}')
        qoi, runtime = sampler._run_model(rom_model, run_dir, params, overwrite)
        rom_qois.append(qoi)
        rom_times.append(runtime)
    
    rom_qois = np.array(rom_qois)
    rom_times = np.array(rom_times)
    
    # Compute exact statistics
    print("Computing correlation and cost statistics\n")
    with np.load(data_npz) as data:
        fom_qois = data['fom_qois_master']
        aux_qois = data['aux_qois_master']
        fom_times = data['fom_times_master']
    
    fom_rom_corr = pilot_mgr.estimate_FOM_correlations(
        [fom_qois[None, :]], [rom_qois[None, :]]
    )[0]
    aux_rom_corr = pilot_mgr.estimate_FOM_correlations(
        [aux_qois[None, :]], [rom_qois[None, :]]
    )[0]
    norm_rom_time = np.mean(rom_times / fom_times)
    
    return fom_rom_corr, aux_rom_corr, norm_rom_time


def run_hybrid_mfuq(fom_model, aux_model, rom_model_builder, parameter_space,
                   absolute_hybrid_MFMC_work_directory,
                   pilot_sample_size=20, pilot_list=[1, 3, 5, 7, 9],
                   max_combinations=25, tunable_range=[1, 20], budget=40,
                   allocate_based_on='min', log_of_objective=True,
                   overwrite=True, random_seed=2025, surrogate_method='neural_network'):
    """Main hybrid MFUQ algorithm."""
    if allocate_based_on not in ['min', 'max']:
        raise ValueError("allocate_based_on must be 'min' or 'max'")
    
    np.random.seed(random_seed)
    work_dir = absolute_hybrid_MFMC_work_directory
    create_empty_dir(work_dir)
    
    log_path = os.path.join(work_dir, "hybrid_status.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Hybrid MFUQ status\n")
        log_file.write(f"Surrogate method: {surrogate_method}\n")
        
        # Step 1: Pilot sampling
        pilot_mgr = mfmc.Pilot(pilot_list, pilot_sample_size, random_seed=random_seed)
        data_npz = f"{work_dir}/pilot_results.npz"
        
        if not os.path.exists(data_npz):
            log_file.write("Doing pilot sampling\n")
            print("Doing pilot sampling\n")
            sampler = PilotSampler(fom_model, aux_model, rom_model_builder,
                                  parameter_space, pilot_mgr, work_dir)
            pilot_data = sampler.run(max_combinations, overwrite)
        else:
            log_file.write("Loading previous pilot results\n")
            print("Loading previous pilot results\n")
            with np.load(data_npz) as data:
                pilot_data = PilotData(
                    fom_qois=data['fom_qois_master'],
                    aux_qois=data['aux_qois_master'],
                    fom_aux_corr=data['fom_aux_corr'],
                    fom_rom_corrs=data['fom_rom_corrs'],
                    aux_rom_corrs=data['aux_rom_corrs'],
                    fom_times=data['fom_times_master'],
                    normalized_aux_time=data['normalized_aux_time'],
                    normalized_rom_times=data['normalized_rom_times'],
                    parameter_samples=data['parameter_samples'],
                    training_dirs=data['training_dirs']
                )
        
        # Step 2: Build surrogates and optimize
        log_file.write(f"Training surrogates using {surrogate_method} method\n")
        print(f"Training surrogates using {surrogate_method} method\n")
        
        builder = SurrogateBuilder(pilot_list, n_active=1, work_dir=work_dir, method=surrogate_method)
        hf_corrs, lf_corrs, costs = builder.build(data_npz)
        
        log_file.write("Solving hybrid MFUQ optimization problem\n")
        print("Solving hybrid MFUQ optimization problem\n")
        
        budget_list = [budget * (i + 1) for i in range(6)]
        bounds = [(1, None), (1.001, None), (1.001, None), tuple(tunable_range)]
        
        (mf_vars, mf_allocs), (is_vars, is_allocs) = optimize_allocation(
            budget_list, hf_corrs, lf_corrs, costs, bounds, log_of_objective, hybrid=False
        )
        
        # Step 3: Train optimized ROM
        s_star = is_allocs[0 if allocate_based_on == 'min' else -1]
        rom_basis_num = int(round(s_star[-1]))
        
        rom_npz = f"{work_dir}/trained_{rom_basis_num}_sample_rom_results.npz"
        if not os.path.exists(rom_npz):
            log_file.write("Training optimized ROM\n")
            print("Training optimized ROM\n")
            
            fom_rom_corr, aux_rom_corr, norm_rom_time = train_optimized_rom(
                fom_model, rom_model_builder, parameter_space, work_dir,
                rom_basis_num, pilot_mgr, pilot_data.parameter_samples,
                pilot_data.training_dirs.tolist(), data_npz, overwrite
            )
            
            np.savez(rom_npz, fom_rom_corr=fom_rom_corr,
                    aux_rom_corr=aux_rom_corr, normalized_rom_time=norm_rom_time)
        else:
            log_file.write("Using previously trained ROM\n")
            print("Using previously trained ROM\n")
        
        # Step 4: Validate with exact statistics
        log_file.write("Solving MFUQ with exact statistics\n")
        print("Solving MFUQ with exact statistics\n")
        
        with np.load(rom_npz) as data:
            fom_rom_corr_val = float(data['fom_rom_corr'])
            aux_rom_corr_val = float(data['aux_rom_corr'])
            normalized_rom_time_val = float(data['normalized_rom_time'])
        
        exact_hf = [hf_corrs[0], lambda s: fom_rom_corr_val]
        exact_lf = [lambda s: aux_rom_corr_val]
        exact_costs = [costs[0], lambda s: normalized_rom_time_val]
        
        bounds_exact = [(1, None), (1.001, None), (1.001, None), (rom_basis_num, rom_basis_num)]
        (mf_vars_ex, mf_allocs_ex), (is_vars_ex, is_allocs_ex) = optimize_allocation(
            budget_list, exact_hf, exact_lf, exact_costs, bounds_exact, log_of_objective, hybrid=True
        )
        
        # Step 5: Save visualization data
        log_file.write("Saving visualization data\n")
        print("Saving visualization data\n")
        
        s_plot = np.arange(1, tunable_range[-1] + 1)
        
        rho12_val = hf_corrs[0](0)
        rho13_vals = np.array([hf_corrs[1]([0, s]) for s in s_plot])
        rho23_vals = np.array([lf_corrs[0]([0, s]) for s in s_plot])
        cost2_val = costs[0](0)
        cost3_vals = np.array([costs[1]([0, s]) for s in s_plot])
        
        np.savez(f"{work_dir}/visualization_data.npz",
                rho12s=np.full_like(s_plot, rho12_val, dtype=float),
                rho13s=rho13_vals, rho23s=rho23_vals,
                cost2s=np.full_like(s_plot, cost2_val, dtype=float),
                cost3s=cost3_vals,
                fom_rom_corrs=pilot_data.fom_rom_corrs,
                aux_rom_corrs=pilot_data.aux_rom_corrs,
                normalized_rom_times=pilot_data.normalized_rom_times,
                ss=np.tile(s_plot, (2, 1)), pp=np.tile(pilot_list, (2, 1)),
                s_star=s_star, xx=budget_list,
                fMFs=mf_vars, fMFs_ex=mf_vars_ex,
                fISs=is_vars, fISs_ex=is_vars_ex)
        
        print(f"Saved visualization data to {work_dir}/visualization_data.npz\n")
        
        # # Debug output
        # print(f"Debug - Sample values:")
        # print(f"  rho12s range: [{rho12_val:.4f}, {rho12_val:.4f}]")
        # print(f"  rho13s range: [{rho13_vals.min():.4f}, {rho13_vals.max():.4f}]")
        # print(f"  rho23s range: [{rho23_vals.min():.4f}, {rho23_vals.max():.4f}]")
        # print(f"  cost2s range: [{cost2_val:.4f}, {cost2_val:.4f}]")
        # print(f"  cost3s range: [{cost3_vals.min():.4f}, {cost3_vals.max():.4f}]")
        # print()
        
        log_file.write("Hybrid MFUQ complete\n")
        print("Hybrid MFUQ complete\n")