"""
Hybrid Multi-Fidelity Uncertainty Quantification (MFUQ) Workflow

Implements hybrid MFUQ combining FOM, multiple auxiliary models, and variable-fidelity ROMs.
"""

import os
import time
from typing import List, Tuple, Callable

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
    qoi_path, time_path = os.path.join(run_dir, "qoi.txt"), os.path.join(run_dir, "time.txt")
    
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
                               n_restarts: int = 50) -> Tuple[float, np.ndarray]:
    """Optimize allocation for given budget and type. Returns (variance, allocation)."""
    opt = MFMC(budget, allocation_type, hybrid=hybrid)
    opt.set_corrs_and_costs(hf_corrs, lf_corrs, costs)
    opt.set_objective_and_constraint(log=log_objective, bounds=bounds)
    # opt.check_gradients()
    
    best_var, best_alloc = float('inf'), None
    for _ in range(n_restarts):
        opt.solve()
        if opt.result.success:
            var = np.exp(opt.result.fun) if log_objective else opt.result.fun
            if 0 <= var < best_var:
                best_var, best_alloc = var, opt.result.x
    
    print(f"{allocation_type} at budget {budget}: variance={best_var:.6f}, alloc={best_alloc}")
    return best_var, best_alloc


def optimize_allocation(budget_list: List[float], hf_corrs: List[Callable], lf_corrs: List[Callable],
                       costs: List[Callable], bounds: List[Tuple], log_objective: bool, 
                       hybrid: bool = False) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """Run optimization for multiple budgets. Returns ((mf_vars, mf_allocs), (is_vars, is_allocs))."""
    results = {'MF': ([], []), 'IS': ([], [])}
    
    for budget in budget_list:
        for alloc_type in ['MF', 'IS']:
            var, alloc = optimize_single_allocation(
                budget, alloc_type, hf_corrs, lf_corrs, costs, bounds, log_objective, hybrid
            )
            results[alloc_type][0].append(var)
            results[alloc_type][1].append(alloc)
    
    return results['MF'], results['IS']


# ============================================================================
# ROM TRAINING
# ============================================================================

def train_optimized_rom(fom_model, rom_builder, param_space, work_dir: str, rom_basis_num: int,
                       pilot_mgr, pilot_data: PilotData, data_npz: str, 
                       overwrite: bool = False) -> Tuple[float, List[float], float]:
    """Train ROM and compute statistics. Returns (fom_rom_corr, aux_rom_corrs, normalized_rom_time)."""
    print(f"\nTraining ROM with {rom_basis_num} basis functions")
    
    # Generate additional FOM samples if needed
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
    
    # Build and evaluate ROM
    rom_dir = f'{work_dir}/pilot/rom_optimized/basis_size_{rom_basis_num}'
    create_empty_dir(rom_dir)
    rom_model = rom_builder.build_from_training_dirs(rom_dir, train_dirs[:rom_basis_num])
    
    rom_qois, rom_times = run_model_on_samples(
        rom_model, rom_dir, param_space, pilot_data.parameter_samples, overwrite
    )
    
    # Compute correlations and costs
    with np.load(data_npz) as data:
        fom_qois, fom_times = data['fom_qois_master'], data['fom_times_master']
    
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
        return PilotData(
            fom_qois=data['fom_qois_master'],
            aux_qois_list=[data[f'aux{i}_qois_master'] for i in range(n_aux)],
            fom_aux_corrs=data['fom_aux_corrs'],
            aux_aux_corrs=data.get('aux_aux_corrs', np.array([])),
            fom_rom_corrs=data['fom_rom_corrs'],
            aux_rom_corrs_list=[data[f'aux{i}_rom_corrs'] for i in range(n_aux)],
            fom_times=data['fom_times_master'],
            normalized_aux_times=data['normalized_aux_times'],
            normalized_rom_times=data['normalized_rom_times'],
            parameter_samples=data['parameter_samples'],
            training_dirs=data['training_dirs']
        )


# def build_exact_functions(hf_corrs: List[Callable], lf_corrs: List[Callable], costs: List[Callable],
#                          fom_rom_corr: float, aux_rom_corrs: List[float], rom_time: float,
#                          n_aux: int) -> Tuple[List[Callable], List[Callable], List[Callable]]:
#     """Build correlation/cost functions with exact ROM statistics."""
#     # HF: FOM-aux (surrogate) + FOM-ROM (exact)
#     exact_hf = hf_corrs[:n_aux] + [lambda s: fom_rom_corr]
    
#     # LF: aux-aux (surrogate) + aux-ROM (exact)
#     n_aux_pairs = n_aux * (n_aux - 1) // 2 if n_aux > 1 else 0
#     exact_lf = lf_corrs[:n_aux_pairs] + [lambda s, v=v: v for v in aux_rom_corrs]
    
#     # Costs: aux (surrogate) + ROM (exact)
#     exact_costs = costs[:n_aux] + [lambda s: rom_time]
    
#     return exact_hf, exact_lf, exact_costs

def build_exact_functions(hf_corrs, lf_corrs, costs,
                          fom_rom_corr, aux_rom_corrs, rom_time, n_aux):

    # HF: FOM-aux (surrogate) + FOM-ROM (exact)
    exact_hf = (
        hf_corrs[:n_aux]
        + [lambda s, v=fom_rom_corr: torch.as_tensor(v, dtype=torch.double)]
    )

    # LF: aux-aux (surrogate) + aux-ROM (exact)
    n_aux_pairs = n_aux * (n_aux - 1) // 2 if n_aux > 1 else 0
    exact_lf = (
        lf_corrs[:n_aux_pairs]
        + [lambda s, v=v: torch.as_tensor(v, dtype=torch.double)
           for v in aux_rom_corrs]
    )

    # Costs: aux (surrogate) + ROM (exact)
    exact_costs = (
        costs[:n_aux]
        + [lambda s, v=rom_time: torch.as_tensor(v, dtype=torch.double)]
    )

    return exact_hf, exact_lf, exact_costs



# def evaluate_surrogates(hf_corrs: List[Callable], lf_corrs: List[Callable], costs: List[Callable],
#                         s_vals: np.ndarray, n_aux: int) -> dict:
#     """Evaluate surrogate functions over range of basis sizes."""
#     result = {}
    
#     # FOM-aux and aux costs (constant w.r.t. s)
#     for i in range(n_aux):
#         result[f'rho_fom_aux{i}'] = np.full_like(s_vals, hf_corrs[i](0), dtype=float)
#         result[f'cost_aux{i}'] = np.full_like(s_vals, costs[i](0), dtype=float)
    
#     # FOM-ROM and ROM cost (vary with s)
#     result['rho_fom_rom'] = np.array([hf_corrs[n_aux]([0, s]) for s in s_vals])
#     result['cost_rom'] = np.array([costs[n_aux]([0, s]) for s in s_vals])
    
#     # Aux-aux (constant, only if n_aux > 1)
#     if n_aux > 1:
#         n_aux_pairs = n_aux * (n_aux - 1) // 2
#         idx = 0
#         for i in range(n_aux):
#             for j in range(i):
#                 result[f'rho_aux{j}_aux{i}'] = np.full_like(s_vals, lf_corrs[idx](0), dtype=float)
#                 idx += 1
    
#     # Aux-ROM (vary with s)
#     lf_start = n_aux * (n_aux - 1) // 2 if n_aux > 1 else 0
#     for i in range(n_aux):
#         result[f'rho_aux{i}_rom'] = np.array([lf_corrs[lf_start + i]([0, s]) for s in s_vals])
    
#     return result

def evaluate_surrogates(hf_corrs, lf_corrs, costs, s_vals, n_aux):
    result = {}
    s0 = s_vals[0]

    def val(x, s):
        if callable(x):
            x = x([0, s])
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return float(x)

    with torch.no_grad():

        # FOM–aux and aux costs (constant in s)
        for i in range(n_aux):
            rho = val(hf_corrs[i], s0)
            cst = val(costs[i], s0)

            result[f"rho_fom_aux{i}"] = np.full_like(s_vals, rho, dtype=float)
            result[f"cost_aux{i}"]    = np.full_like(s_vals, cst, dtype=float)

        # FOM–ROM and ROM cost (vary with s)
        result["rho_fom_rom"] = np.array([
            val(hf_corrs[n_aux], s) for s in s_vals
        ])

        result["cost_rom"] = np.array([
            val(costs[n_aux], s) for s in s_vals
        ])

        # Aux–aux (constant)
        lf_idx = 0
        if n_aux > 1:
            for i in range(n_aux):
                for j in range(i):
                    rho = val(lf_corrs[lf_idx], s0)
                    result[f"rho_aux{j}_aux{i}"] = np.full_like(
                        s_vals, rho, dtype=float
                    )
                    lf_idx += 1

        # Aux–ROM (vary with s)
        for i in range(n_aux):
            result[f"rho_aux{i}_rom"] = np.array([
                val(lf_corrs[lf_idx + i], s) for s in s_vals
            ])

    return result



def build_visualization_dict(pilot_data: PilotData, surrogate_vals: dict, s_star: np.ndarray,
                             budget_list: List[float], mf_vars: List, mf_vars_ex: List,
                             is_vars: List, is_vars_ex: List, is_allocs: List, is_allocs_ex: List,
                             pilot_list: List[int], s_plot: np.ndarray, n_aux: int) -> dict:
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
        'fMFs': mf_vars, 'fMFs_ex': mf_vars_ex,
        'fISs': is_vars, 'fISs_ex': is_vars_ex,
        'fISs_alloc': is_allocs, 'fISs_alloc_ex': is_allocs_ex,
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
# MAIN WORKFLOW
# ============================================================================

def run_hybrid_mfuq(fom_model, aux_models, rom_model_builder, parameter_space,
                   absolute_hybrid_MFMC_work_directory: str,
                   pilot_sample_size: int = 20, pilot_list: List[int] = None,
                   max_combinations: int = 25, tunable_range: List[int] = None,
                   budget: float = 40, allocate_based_on: str = 'min',
                   log_of_objective: bool = True, overwrite: bool = True,
                   random_seed: int = 2025, surrogate_method: str = 'sigmoid'):
    """
    Hybrid MFUQ algorithm with multiple auxiliary models.
    
    Workflow: pilot sampling → surrogate building → optimization → ROM training → validation
    
    Args:
        fom_model: Full-order model
        aux_models: List of auxiliary models (or single model for backward compatibility)
        rom_model_builder: Variable-fidelity ROM builder
        parameter_space: Parameter space for sampling
        absolute_hybrid_MFMC_work_directory: Work directory path
        pilot_sample_size: Number of pilot samples (default: 20)
        pilot_list: ROM basis sizes to test (default: [1,3,5,7,9])
        max_combinations: Max training combinations (default: 25)
        tunable_range: Tunable ROM basis range (default: [1,20])
        budget: Computational budget (default: 40)
        allocate_based_on: 'min' or 'max' for allocation selection (default: 'min')
        log_of_objective: Use log transform in optimization (default: True)
        overwrite: Overwrite existing results (default: True)
        random_seed: Random seed (default: 2025)
        surrogate_method: 'neural_network' or 'sigmoid' (default: 'neural_network')
    """
    if allocate_based_on not in ['min', 'max']:
        raise ValueError("allocate_based_on must be 'min' or 'max'")
    
    # Setup
    pilot_list = pilot_list or [1, 3, 5, 7, 9]
    tunable_range = tunable_range or [1, 20]
    aux_models = aux_models if isinstance(aux_models, list) else [aux_models]
    n_aux = len(aux_models)
    
    np.random.seed(random_seed)
    work_dir = absolute_hybrid_MFMC_work_directory
    create_empty_dir(work_dir)
    
    log_path = os.path.join(work_dir, "hybrid_status.log")
    log = open(log_path, "w", encoding="utf-8")
    log.write(f"Hybrid MFUQ: n_aux={n_aux}, surrogate={surrogate_method}, seed={random_seed}\n\n")
    
    try:
        # Step 1: Pilot sampling
        print("\n" + "="*70)
        print("STEP 1: Pilot Sampling")
        print("="*70)
        log.write("STEP 1: Pilot Sampling\n")
        
        pilot_mgr = Pilot(pilot_list, pilot_sample_size, random_seed=random_seed)
        data_npz = f"{work_dir}/pilot_results.npz"
        
        if os.path.exists(data_npz):
            pilot_data = load_pilot_data(data_npz, n_aux)
            print("Loaded existing pilot data")
            log.write("Loaded existing pilot data\n")
        else:
            sampler = PilotSampler(fom_model, aux_models, rom_model_builder,
                                  parameter_space, pilot_mgr, work_dir)
            pilot_data = sampler.run(max_combinations, overwrite)
            print("Completed pilot sampling")
            log.write("Completed pilot sampling\n")
        
        # Step 2: Build surrogates and optimize
        print("\n" + "="*70)
        print("STEP 2: Surrogate Building and Optimization")
        print("="*70)
        log.write("\nSTEP 2: Surrogate Building and Optimization\n")
        
        builder = SurrogateBuilder(pilot_list, n_active=1, n_aux=n_aux,
                                 work_dir=work_dir, method=surrogate_method)
        hf_corrs, lf_corrs, costs = builder.build(data_npz)
        
        budget_list = [budget * (i + 1) for i in range(6)]
        bounds = [(1, None)] + [(1.001, None)] * (n_aux + 1) + [tuple(tunable_range)]
        
        print(f"\nOptimizing with bounds: {bounds}")
        log.write(f"Bounds: {bounds}\n")
        
        (mf_vars, mf_allocs), (is_vars, is_allocs) = optimize_allocation(
            budget_list, hf_corrs, lf_corrs, costs, bounds, log_of_objective, hybrid=False
        )
        
        # Step 3: Train optimized ROM
        print("\n" + "="*70)
        print("STEP 3: Training Optimized ROM")
        print("="*70)
        log.write("\nSTEP 3: Training Optimized ROM\n")
        
        allocation_idx = 0 if allocate_based_on == 'min' else -1
        s_star = is_allocs[allocation_idx]
        rom_basis_num = int(round(s_star[-1]))
        
        print(f"Optimal allocation: {s_star}")
        print(f"ROM basis size: {rom_basis_num}")
        log.write(f"s*={s_star}, basis={rom_basis_num}\n")
        
        rom_npz = f"{work_dir}/trained_{rom_basis_num}_sample_rom_results.npz"
        
        if os.path.exists(rom_npz):
            print("Using previously trained ROM")
            log.write("Using existing ROM\n")
        else:
            fom_rom_corr, aux_rom_corrs, norm_rom_time = train_optimized_rom(
                fom_model, rom_model_builder, parameter_space, work_dir,
                rom_basis_num, pilot_mgr, pilot_data, data_npz, overwrite
            )
            
            save_dict = {'fom_rom_corr': fom_rom_corr, 'normalized_rom_time': norm_rom_time}
            for i, corr in enumerate(aux_rom_corrs):
                save_dict[f'aux{i}_rom_corr'] = corr
            np.savez(rom_npz, **save_dict)
            log.write("Trained and saved ROM\n")
        
        # Step 4: Validate with exact statistics
        print("\n" + "="*70)
        print("STEP 4: Validation with Exact Statistics")
        print("="*70)
        log.write("\nSTEP 4: Validation\n")
        
        with np.load(rom_npz) as data:
            fom_rom_corr_val = float(data['fom_rom_corr'])
            aux_rom_corr_vals = [float(data[f'aux{i}_rom_corr']) for i in range(n_aux)]
            normalized_rom_time_val = float(data['normalized_rom_time'])
        
        print(f"FOM-ROM correlation: {fom_rom_corr_val:.4f}")
        print(f"Normalized ROM time: {normalized_rom_time_val:.4f}")
        log.write(f"FOM-ROM corr={fom_rom_corr_val:.4f}, time={normalized_rom_time_val:.4f}\n")
        
        exact_hf, exact_lf, exact_costs = build_exact_functions(
            hf_corrs, lf_corrs, costs, fom_rom_corr_val, aux_rom_corr_vals,
            normalized_rom_time_val, n_aux
        )
        
        bounds_exact = [(1, None)] + [(1.001, None)] * (n_aux + 1) + [(rom_basis_num, rom_basis_num)]
        (mf_vars_ex, mf_allocs_ex), (is_vars_ex, is_allocs_ex) = optimize_allocation(
            budget_list, exact_hf, exact_lf, exact_costs, bounds_exact, log_of_objective, hybrid=True
        )
        
        # Step 5: Save visualization data
        print("\n" + "="*70)
        print("STEP 5: Generating Visualization Data")
        print("="*70)
        log.write("\nSTEP 5: Visualization Data\n")
        
        # s_plot = np.arange(1, tunable_range[-1] + 1)
        s_min, s_max = 1, tunable_range[-1] + 1
        n_vis = 200   # or 500 if you want it very smooth
        s_plot = np.linspace(s_min, s_max, n_vis)
    
        surrogate_vals = evaluate_surrogates(hf_corrs, lf_corrs, costs, s_plot, n_aux)
        
        vis_data = build_visualization_dict(
            pilot_data, surrogate_vals, s_star, budget_list,
            mf_vars, mf_vars_ex, is_vars, is_vars_ex, is_allocs, is_allocs_ex,
            pilot_list, s_plot, n_aux
        )
        
        vis_path = f"{work_dir}/visualization_data.npz"
        np.savez(vis_path, **vis_data)
        print(f"\nSaved visualization data to {vis_path}")
        log.write(f"Saved to {vis_path}\n")
        
        print("\n" + "="*70)
        print("HYBRID MFUQ COMPLETE")
        print("="*70 + "\n")
        log.write("\nWorkflow completed successfully\n")
        
    finally:
        log.close()