"""
Hybrid Multi-Fidelity Uncertainty Quantification (MFUQ) Workflow

Implements hybrid MFUQ combining FOM, multiple auxiliary models, and variable-fidelity ROMs.
"""

import os
import time
from typing import List, Tuple, Callable, Dict, Optional

import numpy as np
import torch

from romtools.workflows.hybrid_mfuq.mfuq_methods import MFMC
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.hybrid_mfuq.pilot_methods import PilotData, PilotSampler, Pilot
from romtools.workflows.hybrid_mfuq.surrogate_methods import SurrogateBuilder

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================================
# MODEL EXECUTION
# ============================================================================

def run_model_sample(model, run_dir: str, params: dict, overwrite: bool = False) -> Tuple[float, float]:
    """Run model sample, using cache if available. Returns (qoi, runtime)."""
    qoi_path = os.path.join(run_dir, "qoi.txt")
    time_path = os.path.join(run_dir, "time.txt")
    
    if not overwrite and os.path.exists(qoi_path) and os.path.exists(time_path):
        return np.loadtxt(qoi_path), np.loadtxt(time_path)
    
    create_empty_dir(run_dir)
    model.populate_run_directory(run_dir, params)
    
    t0 = time.time()
    return_code = model.run_model(run_dir, params)
    qoi = model.compute_qoi(run_dir, params)
    runtime = time.time() - t0
    
    if return_code == 0:
        np.savetxt(os.path.join(run_dir, "passed.txt"), [0], fmt="%i")
    np.savetxt(qoi_path, [qoi])
    np.savetxt(time_path, [runtime])
    
    return qoi, runtime


def run_model_on_samples(model, run_dir_prefix: str, param_space, samples: np.ndarray, 
                         overwrite: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on multiple parameter samples. Returns (qois, runtimes)."""
    param_names = param_space.get_names()
    qois, times = [], []
    
    for i, sample in enumerate(samples):
        params = dict(zip(param_names, sample))
        run_dir = f"{run_dir_prefix}/run_{i}"
        qoi, runtime = run_model_sample(model, run_dir, params, overwrite)
        qois.append(qoi)
        times.append(runtime)
    
    return np.array(qois), np.array(times)


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_single_allocation(budget: float, allocation_type: str, hf_corrs: List[Callable],
                               lf_corrs: List[Callable], costs: List[Callable], 
                               bounds: List[Tuple], log_objective: bool, hybrid: bool,
                               use_torch: bool = True, n_restarts: int = 50) -> Tuple[float, np.ndarray]:
    """Optimize allocation for given budget and type. Returns (variance, allocation)."""
    opt = MFMC(budget, allocation_type, hybrid=hybrid, use_torch=use_torch)
    opt.set_corrs_and_costs(hf_corrs, lf_corrs, costs)
    opt.set_objective_and_constraint(log=log_objective, bounds=bounds)
    
    best_var, best_alloc = float('inf'), None
    for _ in range(n_restarts):
        opt.solve()
        if opt.result.success:
            var = np.exp(opt.result.fun) if log_objective else opt.result.fun
            if 0 <= var < best_var:
                best_var, best_alloc = var, opt.result.x
    
    print(f"{allocation_type} at budget {budget}: variance={best_var:.6f}")
    return best_var, best_alloc


# ============================================================================
# ROM TRAINING
# ============================================================================

def train_optimized_rom(fom_model, rom_builder, param_space, work_dir: str, rom_basis_num: int,
                       pilot_mgr, pilot_data: PilotData, data_npz: str, 
                       overwrite: bool = False) -> Tuple[float, List[float], float]:
    """Train ROM and compute statistics. Returns (fom_rom_corr, aux_rom_corrs, normalized_rom_time)."""
    print(f"\nTraining ROM with {rom_basis_num} basis functions")
    
    # Ensure we have enough FOM training samples
    train_dirs = list(pilot_data.training_dirs)
    if len(train_dirs) < rom_basis_num:
        num_extra = rom_basis_num - len(train_dirs)
        extra_samples = param_space.generate_samples(num_extra)
        param_names = param_space.get_names()
        
        for i, sample in enumerate(extra_samples):
            fom_dir = f'{work_dir}/pilot/fom/run_{len(train_dirs)}'
            params = dict(zip(param_names, sample))
            run_model_sample(fom_model, fom_dir, params, overwrite)
            train_dirs.append(fom_dir)
    
    # Build ROM
    rom_dir = f'{work_dir}/pilot/rom_optimized/basis_size_{rom_basis_num}'
    create_empty_dir(rom_dir)
    rom_model = rom_builder.build_from_training_dirs(rom_dir, train_dirs[:rom_basis_num])
    
    # Evaluate ROM
    rom_qois, rom_times = run_model_on_samples(
        rom_model, rom_dir, param_space, pilot_data.parameter_samples, overwrite
    )
    
    # Compute correlations
    with np.load(data_npz) as data:
        fom_qois = data['fom_qois_master']
        fom_times = data['fom_times_master']
    
    fom_rom_corr = pilot_mgr.estimate_FOM_correlations([fom_qois[None, :]], [rom_qois[None, :]])[0]
    aux_rom_corrs = [
        pilot_mgr.estimate_FOM_correlations([aux_qois[None, :]], [rom_qois[None, :]])[0]
        for aux_qois in pilot_data.aux_qois_list
    ]
    normalized_rom_time = np.mean(rom_times / fom_times)
    
    return fom_rom_corr, aux_rom_corrs, normalized_rom_time


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

def load_pilot_data(data_npz: str, n_aux: int) -> PilotData:
    """Load pilot data from NPZ file."""
    with np.load(data_npz) as data:
        aux_aux_corrs = data.get('aux_aux_corrs', np.array([]))
        
        return PilotData(
            fom_qois=data['fom_qois_master'],
            aux_qois_list=[data[f'aux{i}_qois_master'] for i in range(n_aux)],
            fom_aux_corrs=data['fom_aux_corrs'],
            aux_aux_corrs=aux_aux_corrs,
            fom_rom_corrs=data['fom_rom_corrs'],
            aux_rom_corrs_list=[data[f'aux{i}_rom_corrs'] for i in range(n_aux)],
            fom_times=data['fom_times_master'],
            normalized_aux_times=data['normalized_aux_times'],
            normalized_rom_times=data['normalized_rom_times'],
            parameter_samples=data['parameter_samples'],
            training_dirs=data['training_dirs']
        )


def build_visualization_dict(pilot_data: PilotData, surrogate_vals: Dict[str, np.ndarray], 
                             s_star: np.ndarray, budget_list: List[float], 
                             mf_vars: List, mf_vars_ex: List, is_vars: List, is_vars_ex: List, 
                             is_allocs: List, is_allocs_ex: List, pilot_list: List[int], 
                             s_plot: np.ndarray, n_aux: int) -> Dict:
    """Assemble complete visualization data dictionary."""
    vis_data = {
        # Pilot data
        'fom_rom_corrs_pilot': pilot_data.fom_rom_corrs,
        'normalized_rom_times_pilot': pilot_data.normalized_rom_times,
        # Grid data
        'ss': np.tile(s_plot, (2, 1)),
        'pp': np.tile(pilot_list, (2, 1)),
        's_star': s_star,
        'xx': budget_list,
        # Optimization results
        'fMFs': mf_vars,
        'fMFs_ex': mf_vars_ex,
        'fISs': is_vars,
        'fISs_ex': is_vars_ex,
        'fISs_alloc': is_allocs,
        'fISs_alloc_ex': is_allocs_ex,
        'n_aux': n_aux,
        # Surrogate evaluations
        'rho_fom_rom_vals': surrogate_vals['rho_fom_rom'],
        'cost_rom_vals': surrogate_vals['cost_rom']
    }
    
    # Add aux-specific data
    for i in range(n_aux):
        vis_data[f'rho_fom_aux{i}_vals'] = surrogate_vals[f'rho_fom_aux{i}']
        vis_data[f'rho_fom_aux{i}_pilot'] = pilot_data.fom_aux_corrs[i]
        vis_data[f'cost_aux{i}_vals'] = surrogate_vals[f'cost_aux{i}']
        vis_data[f'cost_aux{i}_pilot'] = pilot_data.normalized_aux_times[i]
        vis_data[f'rho_aux{i}_rom_vals'] = surrogate_vals[f'rho_aux{i}_rom']
        vis_data[f'rho_aux{i}_rom_pilot'] = pilot_data.aux_rom_corrs_list[i]
    
    # Add aux-aux data if multiple aux models
    if n_aux > 1:
        idx = 0
        for i in range(n_aux):
            for j in range(i):
                vis_data[f'rho_aux{j}_aux{i}_vals'] = surrogate_vals[f'rho_aux{j}_aux{i}']
                vis_data[f'rho_aux{j}_aux{i}_pilot'] = pilot_data.aux_aux_corrs[idx]
                idx += 1
    
    return vis_data


# ============================================================================
# WORKFLOW LOGGING
# ============================================================================

class WorkflowLogger:
    """Simple logger for workflow progress."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, "w", encoding="utf-8")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_file.close()
        return False
    
    def write(self, message: str):
        """Write to both console and log file."""
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()
    
    def section(self, title: str, step: Optional[int] = None):
        """Write a section header."""
        separator = "=" * 70
        if step is not None:
            header = f"STEP {step}: {title}"
        else:
            header = title
        
        self.write("\n" + separator)
        self.write(header)
        self.write(separator)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_hybrid_mfuq(fom_model, aux_models, rom_model_builder, parameter_space,
                   absolute_hybrid_MFMC_work_directory: str,
                   pilot_sample_size: int = 20,
                   pilot_list: List[int] = None,
                   max_combinations: int = 5,
                   tunable_range: List[int] = None,
                   budget: float = 40,
                   allocate_based_on: str = 'min',
                   log_of_objective: bool = True,
                   overwrite: bool = True,
                   random_seed: int = 2025,
                   surrogate_method: str = 'neural_network',
                   use_torch: bool = True):
    """
    Hybrid MFUQ algorithm with multiple auxiliary models.
    
    Workflow: pilot sampling → surrogate building → optimization → ROM training → validation
    
    Args:
        fom_model: Full-order model
        aux_models: List of auxiliary models (or single model for backward compatibility)
        rom_model_builder: Variable-fidelity ROM builder
        parameter_space: Parameter space for sampling
        absolute_hybrid_MFMC_work_directory: Work directory path
        pilot_sample_size: Number of pilot samples
        pilot_list: ROM basis sizes to test
        max_combinations: Max training combinations
        tunable_range: Tunable ROM basis range
        budget: Computational budget
        allocate_based_on: 'min' or 'max' for allocation selection
        log_of_objective: Use log transform in optimization
        overwrite: Overwrite existing results
        random_seed: Random seed
        surrogate_method: 'neural_network' or 'sigmoid'
        use_torch: Use PyTorch for optimization gradients
    """
    # Validate inputs
    if allocate_based_on not in ['min', 'max']:
        raise ValueError("allocate_based_on must be 'min' or 'max'")
    
    # Setup defaults and configuration
    pilot_list = pilot_list or [1, 3, 5, 7, 9]
    tunable_range = tunable_range or [1, 20]
    aux_models = aux_models if isinstance(aux_models, list) else [aux_models]
    n_aux = len(aux_models)
    
    np.random.seed(random_seed)
    work_dir = absolute_hybrid_MFMC_work_directory
    create_empty_dir(work_dir)
    
    with WorkflowLogger(os.path.join(work_dir, "hybrid_status.log")) as logger:
        logger.write(f"Hybrid MFUQ Configuration:")
        logger.write(f"  n_aux={n_aux}, surrogate={surrogate_method}, seed={random_seed}, use_torch={use_torch}")
        
        # ====================================================================
        # STEP 1: Pilot Sampling
        # ====================================================================
        logger.section("Pilot Sampling", step=1)
        
        pilot_mgr = Pilot(pilot_list, pilot_sample_size, random_seed=random_seed)
        data_npz = f"{work_dir}/pilot_results.npz"
        
        if os.path.exists(data_npz):
            pilot_data = load_pilot_data(data_npz, n_aux)
            logger.write("Loaded existing pilot data")
        else:
            sampler = PilotSampler(fom_model, aux_models, rom_model_builder,
                                  parameter_space, pilot_mgr, work_dir)
            pilot_data = sampler.run(max_combinations, overwrite)
            logger.write("Completed pilot sampling")
        
        # ====================================================================
        # STEP 2: Surrogate Building and Optimization (Surrogates Only)
        # ====================================================================
        logger.section("Surrogate Building and Optimization", step=2)
        
        builder = SurrogateBuilder(pilot_list, n_active=1, n_aux=n_aux,
                                 work_dir=work_dir, method=surrogate_method, use_torch=use_torch)
        hf_corrs, lf_corrs, costs = builder.build(data_npz)
        
        budget_list = [budget * (i + 1) for i in range(6)]
        bounds = [(1, None)] + [(1.001, None)] * (n_aux + 1) + [tuple(tunable_range)]
        
        logger.write(f"Optimization bounds: {bounds}")
        
        # Run optimization with surrogate models only (hybrid=False)
        mf_vars, mf_allocs = [], []
        is_vars, is_allocs = [], []
        
        for bgt in budget_list:
            var, alloc = optimize_single_allocation(
                bgt, 'MF', hf_corrs, lf_corrs, costs, bounds, log_of_objective, 
                hybrid=False, use_torch=use_torch
            )
            print(f"  allocation={alloc}")
            mf_vars.append(var)
            mf_allocs.append(alloc)
            
            var, alloc = optimize_single_allocation(
                bgt, 'IS', hf_corrs, lf_corrs, costs, bounds, log_of_objective,
                hybrid=False, use_torch=use_torch
            )
            print(f"  allocation={alloc}")
            is_vars.append(var)
            is_allocs.append(alloc)
        
        # ====================================================================
        # STEP 3: Train Optimized ROM
        # ====================================================================
        logger.section("Training Optimized ROM", step=3)
        
        allocation_idx = 0 if allocate_based_on == 'min' else -1
        s_star = is_allocs[allocation_idx]
        rom_basis_num = int(round(s_star[-1]))
        
        logger.write(f"Optimal allocation: {s_star}")
        logger.write(f"ROM basis size: {rom_basis_num}")
        
        rom_npz = f"{work_dir}/trained_{rom_basis_num}_sample_rom_results.npz"
        
        if os.path.exists(rom_npz):
            logger.write("Using previously trained ROM")
        else:
            fom_rom_corr, aux_rom_corrs, norm_rom_time = train_optimized_rom(
                fom_model, rom_model_builder, parameter_space, work_dir,
                rom_basis_num, pilot_mgr, pilot_data, data_npz, overwrite
            )
            
            save_dict = {
                'fom_rom_corr': fom_rom_corr,
                'normalized_rom_time': norm_rom_time
            }
            for i, corr in enumerate(aux_rom_corrs):
                save_dict[f'aux{i}_rom_corr'] = corr
            
            np.savez(rom_npz, **save_dict)
            logger.write("Trained and saved ROM")
        
        # ====================================================================
        # STEP 4: Validation with Exact Statistics
        # ====================================================================
        logger.section("Validation with Exact Statistics", step=4)
        
        with np.load(rom_npz) as data:
            fom_rom_corr_val = float(data['fom_rom_corr'])
            aux_rom_corr_vals = [float(data[f'aux{i}_rom_corr']) for i in range(n_aux)]
            normalized_rom_time_val = float(data['normalized_rom_time'])
        
        logger.write(f"FOM-ROM correlation: {fom_rom_corr_val:.4f}")
        logger.write(f"Normalized ROM time: {normalized_rom_time_val:.4f}")
        
        # Build exact functions by replacing ROM entries with exact values
        # HF correlations: FOM-aux (surrogate) + FOM-ROM (exact)
        exact_hf = hf_corrs[:n_aux] + [
            (lambda v: (lambda s: torch.as_tensor(v, dtype=torch.double) if use_torch else v))(fom_rom_corr_val)
        ]
        
        # LF correlations: aux-aux (surrogate) + aux-ROM (exact)
        n_aux_pairs = n_aux * (n_aux - 1) // 2 if n_aux > 1 else 0
        exact_lf = lf_corrs[:n_aux_pairs] + [
            (lambda v: (lambda s: torch.as_tensor(v, dtype=torch.double) if use_torch else v))(corr)
            for corr in aux_rom_corr_vals
        ]
        
        # Costs: aux (surrogate) + ROM (exact)
        exact_costs = costs[:n_aux] + [
            (lambda v: (lambda s: torch.as_tensor(v, dtype=torch.double) if use_torch else v))(normalized_rom_time_val)
        ]
        
        bounds_exact = [(1, None)] + [(1.001, None)] * (n_aux + 1) + [(rom_basis_num, rom_basis_num)]
        
        # Run optimization with exact ROM statistics (hybrid=True)
        mf_vars_ex, mf_allocs_ex = [], []
        is_vars_ex, is_allocs_ex = [], []
        
        for bgt in budget_list:
            var, alloc = optimize_single_allocation(
                bgt, 'MF', exact_hf, exact_lf, exact_costs, bounds_exact, log_of_objective,
                hybrid=True, use_torch=use_torch
            )
            print(f"  allocation={alloc}")
            mf_vars_ex.append(var)
            mf_allocs_ex.append(alloc)
            
            var, alloc = optimize_single_allocation(
                bgt, 'IS', exact_hf, exact_lf, exact_costs, bounds_exact, log_of_objective,
                hybrid=True, use_torch=use_torch
            )
            print(f"  allocation={alloc}")
            is_vars_ex.append(var)
            is_allocs_ex.append(alloc)
        
        # ====================================================================
        # STEP 5: Generate Visualization Data
        # ====================================================================
        logger.section("Generating Visualization Data", step=5)
        
        s_min, s_max = 1, tunable_range[-1] + 1
        n_vis = 200
        s_plot = np.linspace(s_min, s_max, n_vis)
        
        # Evaluate surrogates over the range of s values
        surrogate_vals = {}
        
        with torch.no_grad():
            # FOM-aux correlations and aux costs (constant w.r.t. s)
            for i in range(n_aux):
                rho_val = hf_corrs[i]([0, s_plot[0]])
                cost_val = costs[i]([0, s_plot[0]])
                
                # Convert to float if needed
                if hasattr(rho_val, "detach"):
                    rho_val = float(rho_val.detach().cpu().numpy())
                if hasattr(cost_val, "detach"):
                    cost_val = float(cost_val.detach().cpu().numpy())
                
                surrogate_vals[f'rho_fom_aux{i}'] = np.full_like(s_plot, rho_val, dtype=float)
                surrogate_vals[f'cost_aux{i}'] = np.full_like(s_plot, cost_val, dtype=float)
            
            # FOM-ROM correlation and ROM cost (vary with s)
            rho_fom_rom_vals = []
            cost_rom_vals = []
            for s in s_plot:
                rho_val = hf_corrs[n_aux]([0, s])
                cost_val = costs[n_aux]([0, s])
                
                if hasattr(rho_val, "detach"):
                    rho_val = float(rho_val.detach().cpu().numpy())
                if hasattr(cost_val, "detach"):
                    cost_val = float(cost_val.detach().cpu().numpy())
                
                rho_fom_rom_vals.append(rho_val)
                cost_rom_vals.append(cost_val)
            
            surrogate_vals['rho_fom_rom'] = np.array(rho_fom_rom_vals)
            surrogate_vals['cost_rom'] = np.array(cost_rom_vals)
            
            # Aux-aux correlations (constant, only if n_aux > 1)
            if n_aux > 1:
                lf_idx = 0
                for i in range(n_aux):
                    for j in range(i):
                        rho_val = lf_corrs[lf_idx]([0, s_plot[0]])
                        if hasattr(rho_val, "detach"):
                            rho_val = float(rho_val.detach().cpu().numpy())
                        
                        surrogate_vals[f'rho_aux{j}_aux{i}'] = np.full_like(s_plot, rho_val, dtype=float)
                        lf_idx += 1
            
            # Aux-ROM correlations (vary with s)
            lf_start = n_aux * (n_aux - 1) // 2 if n_aux > 1 else 0
            for i in range(n_aux):
                rho_aux_rom_vals = []
                for s in s_plot:
                    rho_val = lf_corrs[lf_start + i]([0, s])
                    if hasattr(rho_val, "detach"):
                        rho_val = float(rho_val.detach().cpu().numpy())
                    rho_aux_rom_vals.append(rho_val)
                
                surrogate_vals[f'rho_aux{i}_rom'] = np.array(rho_aux_rom_vals)
        
        vis_data = build_visualization_dict(
            pilot_data, surrogate_vals, s_star, budget_list,
            mf_vars, mf_vars_ex, is_vars, is_vars_ex, is_allocs, is_allocs_ex,
            pilot_list, s_plot, n_aux
        )
        
        vis_path = f"{work_dir}/visualization_data.npz"
        np.savez(vis_path, **vis_data)
        logger.write(f"Saved visualization data to {vis_path}")
        
        # ====================================================================
        # Complete
        # ====================================================================
        logger.section("HYBRID MFUQ COMPLETE")
        logger.write("Workflow completed successfully")