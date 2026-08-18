"""
Hybrid Multi-Fidelity Uncertainty Quantification workflow (writeup Sec. 4).

Runs the four stages of Algorithm 1 over a FOM, any number of fixed auxiliary
models, and any number of trainable ROMs:

    Step 1  pilot sampling (Sec. 4.2)          -> pilot_methods.py
    Step 2  cost/correlation surrogates and
            the surrogate ACV solve (Sec. 4.3, 4.4)
    Step 3  train the ROMs at the selected basis sizes
    Step 4  re-solve with exact statistics (Sec. 4.5)
    Step 5  sample the surrogates for the diagnostic figures

Model ordering and pair flattening come from `model_indices.py`, which also
carries the writeup-to-code notation map.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict, Optional

import numpy as np
import torch

from romtools.workflows.hybrid_mfuq.mfuq_methods import MFMC
from romtools.workflows.hybrid_mfuq.model_indices import (
    aux_model_slot,
    aux_pairs,
    lf_pairs,
    rom_lf_slot,
    rom_model_slot,
    tril_position,
)
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.hybrid_mfuq.pilot_methods import (
    PilotData,
    PilotSampler,
    Pilot,
    run_model_sample,
)
from romtools.workflows.hybrid_mfuq.surrogate_methods import SurrogateBuilder

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ============================================================================
# MODEL EXECUTION
# ============================================================================

def _to_float(x):
    """Convert a possibly-tensor scalar to a plain Python float."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return float(x)


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

def optimize_single_allocation(
    budget: float,
    allocation_type: str,
    hf_corrs: List[Callable],
    lf_corrs: List[Callable],
    costs: List[Callable],
    bounds: List[Tuple],
    log_objective: bool,
    hybrid: bool,
    use_torch: bool = True,
    n_restarts: int = 10,
    corr_matrix_fn: Optional[Callable] = None,
    n_active: int = 1,
) -> Tuple[float, np.ndarray]:
    """
    Optimize allocation for given budget and type.

    Returns (variance, allocation) at the *discrete* solution of writeup
    Eq. (16): the best feasible continuous optimum found over n_restarts
    multistarts is floored/rounded by MFMC.discretize, and the objective is
    then re-evaluated there. Reporting the relaxed optimum instead would be
    optimistic relative to the strategy actually realized, since N is
    floored downward.

    Raises RuntimeError if no restart produced a usable solution, rather
    than returning None for the allocation and failing later and further
    away (previously this surfaced as a None basis size in Step 3).
    """
    opt = MFMC(budget, allocation_type, hybrid=hybrid, n_active=n_active, use_torch=use_torch)
    opt.set_corrs_and_costs(
        hf_corrs,
        lf_corrs,
        costs,
        corr_matrix_fn=corr_matrix_fn,
    )
    opt.set_objective_and_constraint(log=log_objective, bounds=bounds)

    best_var, best_alloc = float("inf"), None
    n_success = 0

    for _ in range(n_restarts):
        opt.solve()

        if opt.result.success:
            n_success += 1
            var = np.exp(opt.result.fun) if log_objective else opt.result.fun

            if 0 <= var < best_var:
                best_var, best_alloc = var, opt.result.x

    if best_alloc is None:
        raise RuntimeError(
            f"{allocation_type} allocation at budget {budget} failed: "
            f"{n_success}/{n_restarts} restarts converged and none produced a "
            f"finite non-negative objective. Check the surrogate correlation "
            f"matrix (an inadmissible P(s) makes F*C indefinite), the cost "
            f"surrogates, and whether the budget can cover smin."
        )

    alloc = opt.discretize(best_alloc)
    value = opt.objective(alloc)
    var = float(np.exp(value)) if log_objective else float(value)

    print(
        f"{allocation_type} at budget {budget}: variance={var:.6f} "
        f"(discrete; relaxed={best_var:.6f}, "
        f"{n_success}/{n_restarts} restarts converged)"
    )

    return var, alloc


@dataclass
class AllocationResults:
    """
    One allocation scheme's budget-sweep results.

    `vars`/`allocs` come from the Step 2 surrogate solve, `vars_ex`/`allocs_ex`
    from the Step 4 solve with exact trained-ROM statistics. All four are
    indexed by position in budget_list.
    """
    vars: List = field(default_factory=list)
    vars_ex: List = field(default_factory=list)
    allocs: List = field(default_factory=list)
    allocs_ex: List = field(default_factory=list)


def sweep_budgets(
    budget_list: List[float],
    allocation_type_pair: Tuple[str, str],
    hf_corrs: List[Callable],
    lf_corrs: List[Callable],
    costs: List[Callable],
    bounds: List[Tuple],
    log_objective: bool,
    use_torch: bool,
    corr_matrix_fn: Optional[Callable],
    n_active: int = 1,
) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """
    Run the paired budget sweep shared by Step 2 (surrogate-based
    optimization) and Step 4 (validation with exact statistics): for each
    budget in budget_list, call optimize_single_allocation once per type
    in allocation_type_pair (in order), always with hybrid=True and the
    same hf_corrs/lf_corrs/costs/bounds/corr_matrix_fn.

    n_active is the number of trainable ROM dimensions in s, and thus in
    corr_matrix_fn's raw input and the tail of bounds; it is forwarded to
    MFMC so the optimizer's state vector is sized correctly.

    Returns ((vars_a, allocs_a), (vars_b, allocs_b)) for the two types in
    allocation_type_pair, in that order.
    """
    type_a, type_b = allocation_type_pair
    vars_a, allocs_a = [], []
    vars_b, allocs_b = [], []

    for bgt in budget_list:
        var, alloc = optimize_single_allocation(
            bgt,
            type_a,
            hf_corrs,
            lf_corrs,
            costs,
            bounds,
            log_objective,
            hybrid=True,
            use_torch=use_torch,
            corr_matrix_fn=corr_matrix_fn,
            n_active=n_active,
        )
        print(f"  allocation={alloc}")
        vars_a.append(var)
        allocs_a.append(alloc)

        var, alloc = optimize_single_allocation(
            bgt,
            type_b,
            hf_corrs,
            lf_corrs,
            costs,
            bounds,
            log_objective,
            hybrid=True,
            use_torch=use_torch,
            corr_matrix_fn=corr_matrix_fn,
            n_active=n_active,
        )
        print(f"  allocation={alloc}")
        vars_b.append(var)
        allocs_b.append(alloc)

    return (vars_a, allocs_a), (vars_b, allocs_b)


# ============================================================================
# ROM TRAINING
# ============================================================================

def train_optimized_roms(fom_model, rom_model_builders, param_space, work_dir: str,
                        rom_basis_nums: List[int], pilot_mgr, pilot_data: PilotData,
                        data_npz: str, overwrite: bool = False):
    """
    Train every trainable ROM at its own selected optimal basis size and
    compute the statistics Step 4 needs to validate the online solution:
    FOM-ROM_t and aux_i-ROM_t correlations and normalized cost for every t
    (as train_optimized_rom did for the k=1 case), plus ROM_t-ROM_q
    correlations for every pair t < q (new for k > 1 -- required to fill
    the LF-LF correlation matrix's ROM-ROM block during validation).

    All trainable ROMs are evaluated on the same pilot_data.parameter_samples
    set so the ROM-ROM correlations are directly comparable (same sample
    index i lines up across ROMs).

    Training sets: each ROM t is built from `train_dirs[:s*_t]`, i.e. the
    deterministic prefix of the pilot training directories, extended with
    fresh FOM runs only if s* exceeds the pilot sample count. This is
    intentional and is the ROM a user would actually obtain, but note the
    asymmetry with Step 1: the pilot procedure (writeup §4.2) characterizes
    "a ROM of basis size s" by averaging over resampled training sets of that
    size, whereas validation realizes one specific training set. Part of any
    predicted-vs-validated variance gap is therefore this single draw rather
    than surrogate error. It also means nested basis sizes share snapshots
    across trainable ROMs, so the validated ROM-ROM correlations correspond
    to the high-overlap end of the training-set distribution.

    Statistics here are computed over all Np pilot samples, including those
    used to train the ROMs (writeup §4.5, footnote 5).

    Returns (fom_rom_corrs, aux_rom_corrs, rom_rom_corrs, normalized_rom_times):
      fom_rom_corrs: List[float], one per ROM t.
      aux_rom_corrs: List[List[float]], aux_rom_corrs[i][t].
      rom_rom_corrs: Dict[(t, q), float] for every t < q.
      normalized_rom_times: List[float], one per ROM t.
    """
    print(f"\nTraining {len(rom_model_builders)} optimized ROM(s) at basis sizes {rom_basis_nums}")

    # Ensure we have enough FOM training samples for the largest basis size.
    train_dirs = list(pilot_data.training_dirs)
    max_basis = max(rom_basis_nums)
    if len(train_dirs) < max_basis:
        num_extra = max_basis - len(train_dirs)
        extra_samples = param_space.generate_samples(num_extra)
        param_names = param_space.get_names()

        for i, sample in enumerate(extra_samples):
            fom_dir = f'{work_dir}/pilot/fom/run_{len(train_dirs)}'
            params = dict(zip(param_names, sample))
            run_model_sample(fom_model, fom_dir, params, overwrite)
            train_dirs.append(fom_dir)

    # Build and evaluate every trainable ROM.
    rom_qois_list, rom_times_list = [], []
    for t, (rom_builder, rom_basis_num) in enumerate(zip(rom_model_builders, rom_basis_nums)):
        rom_subdir = f'basis_size_{rom_basis_num}' if t == 0 else f'rom{t}_basis_size_{rom_basis_num}'
        rom_dir = f'{work_dir}/pilot/rom_optimized/{rom_subdir}'
        create_empty_dir(rom_dir)
        rom_model = rom_builder.build_from_training_dirs(rom_dir, train_dirs[:rom_basis_num])

        rom_qois, rom_times = run_model_on_samples(
            rom_model, rom_dir, param_space, pilot_data.parameter_samples, overwrite
        )
        rom_qois_list.append(rom_qois)
        rom_times_list.append(rom_times)

    with np.load(data_npz) as data:
        fom_qois = data['fom_qois_master']
        fom_times = data['fom_times_master']

    fom_rom_corrs = [
        pilot_mgr.estimate_pairwise_correlations([fom_qois[None, :]], [rom_qois[None, :]])[0]
        for rom_qois in rom_qois_list
    ]
    aux_rom_corrs = [
        [
            pilot_mgr.estimate_pairwise_correlations([aux_qois[None, :]], [rom_qois[None, :]])[0]
            for rom_qois in rom_qois_list
        ]
        for aux_qois in pilot_data.aux_qois_list
    ]
    rom_rom_corrs = {}
    for t in range(len(rom_qois_list)):
        for q in range(t + 1, len(rom_qois_list)):
            rom_rom_corrs[(t, q)] = pilot_mgr.estimate_pairwise_correlations(
                [rom_qois_list[t][None, :]], [rom_qois_list[q][None, :]]
            )[0]

    normalized_rom_times = [
        float(np.mean(rom_times / fom_times)) for rom_times in rom_times_list
    ]

    return fom_rom_corrs, aux_rom_corrs, rom_rom_corrs, normalized_rom_times


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

def load_pilot_data(data_npz: str, n_aux: int, n_active: int) -> PilotData:
    """Load pilot data from an NPZ file written by PilotData.to_npz_dict."""
    with np.load(data_npz) as data:
        return PilotData.from_npz(data, n_aux, n_active)


def _plot_state_builders(rom_reference, n_aux, n_active):
    """
    Build the two state-vector constructors the plotting sweeps need.

    Each sweep profiles one trainable ROM against its own basis size while
    every other ROM stays at its Step-3 selected size, so both constructors
    take (t, rom_val) and vary only ROM t.

    expanded: length n_lofi, the convention MFMC.expand_s produces and the
        scalar cost/correlation surrogates expect (zeros in the aux slots).
    raw_active: length n_active, which corr_matrix_fn takes unexpanded.
    """
    def raw_active(t, rom_val):
        tail = list(rom_reference)
        tail[t] = rom_val
        return tail

    def expanded(t, rom_val):
        return [0.0] * n_aux + raw_active(t, rom_val)

    return expanded, raw_active


def _evaluate_matrix_surrogates(
    costs, corr_matrix_fn, expanded, raw_active, s_plots, n_aux, n_active
):
    """Sample the matrix-valued AH surrogate for the diagnostic plots."""
    values = {}
    n_vis = len(s_plots[0])

    # Fixed-fixed correlations and auxiliary costs do not depend on any basis
    # size, so one reference evaluation supplies all of them.
    s_ref = torch.tensor(raw_active(0, s_plots[0][0]), dtype=torch.float64)
    P_ref = corr_matrix_fn(s_ref)
    if torch.is_tensor(P_ref):
        P_ref = P_ref.detach().cpu().numpy()

    for i in range(n_aux):
        values[f"rho_fom_aux{i}"] = np.full(
            n_vis, float(P_ref[aux_model_slot(i), 0])
        )
        values[f"cost_aux{i}"] = np.full(
            n_vis, _to_float(costs[i](expanded(0, s_plots[0][0])))
        )

    for _, i, j in aux_pairs(n_aux):
        values[f"rho_aux{j}_aux{i}"] = np.full(
            n_vis, float(P_ref[aux_model_slot(i), aux_model_slot(j)])
        )

    for t in range(n_active):
        rom_idx = rom_model_slot(n_aux, t)

        P_vals = []
        cost_vals = []
        for s in s_plots[t]:
            P = corr_matrix_fn(
                torch.tensor(raw_active(t, float(s)), dtype=torch.float64)
            )
            if torch.is_tensor(P):
                P = P.detach().cpu().numpy()

            P_vals.append(np.asarray(P, dtype=float))
            cost_vals.append(_to_float(costs[n_aux + t](expanded(t, s))))

        P_vals = np.array(P_vals)

        values[f"rho_fom_rom{t}"] = P_vals[:, rom_idx, 0]
        values[f"cost_rom{t}"] = np.array(cost_vals)

        for i in range(n_aux):
            values[f"rho_aux{i}_rom{t}"] = P_vals[:, rom_idx, aux_model_slot(i)]

    return values


def _evaluate_scalar_surrogates(
    hf_corrs, lf_corrs, costs, expanded, s_plots, n_aux, n_active
):
    """Sample the componentwise scalar surrogates for the diagnostic plots."""
    values = {}
    n_vis = len(s_plots[0])
    ref_state = expanded(0, s_plots[0][0])

    for i in range(n_aux):
        values[f"rho_fom_aux{i}"] = np.full(
            n_vis, _to_float(hf_corrs[i](ref_state))
        )
        values[f"cost_aux{i}"] = np.full(
            n_vis, _to_float(costs[i](ref_state))
        )

    for pair, i, j in aux_pairs(n_aux):
        values[f"rho_aux{j}_aux{i}"] = np.full(
            n_vis, _to_float(lf_corrs[pair](ref_state))
        )

    for t in range(n_active):
        # Flat position of ROM t's row in the lower-triangular LF ordering;
        # column i within that row is auxiliary model i.
        row_start = tril_position(rom_lf_slot(n_aux, t), 0)

        states = [expanded(t, s) for s in s_plots[t]]

        values[f"rho_fom_rom{t}"] = np.array(
            [_to_float(hf_corrs[n_aux + t](state)) for state in states]
        )
        values[f"cost_rom{t}"] = np.array(
            [_to_float(costs[n_aux + t](state)) for state in states]
        )

        for i in range(n_aux):
            values[f"rho_aux{i}_rom{t}"] = np.array(
                [_to_float(lf_corrs[row_start + i](state)) for state in states]
            )

    return values


def evaluate_surrogates_for_plots(
    hf_corrs, lf_corrs, costs, corr_matrix_fn,
    rom_basis_nums, s_plots, n_aux, n_active,
):
    """
    Sample every cost and correlation surrogate over each trainable ROM's
    basis-size range, for the diagnostic figures.

    Each ROM is profiled against its own basis size with the others held at
    their Step-3 selected sizes. Joint ROM_t-ROM_q surfaces are a separate
    presentation concern and are not sampled here.
    """
    expanded, raw_active = _plot_state_builders(
        list(rom_basis_nums), n_aux, n_active
    )

    with torch.no_grad():
        if corr_matrix_fn is not None:
            return _evaluate_matrix_surrogates(
                costs, corr_matrix_fn, expanded, raw_active,
                s_plots, n_aux, n_active,
            )

        return _evaluate_scalar_surrogates(
            hf_corrs, lf_corrs, costs, expanded, s_plots, n_aux, n_active,
        )


def build_visualization_dict(pilot_data: PilotData, surrogate_vals: Dict[str, np.ndarray],
                             s_star: np.ndarray, budget_list: List[float],
                             alloc_results: Dict[str, AllocationResults],
                             pilot_basis_grids: List[List[int]],
                             s_plots: List[np.ndarray], n_aux: int, n_active: int,
                             validation_budget_idx: int) -> Dict:
    """Assemble complete visualization data dictionary.

    Per-ROM entries (one basis-size sweep per trainable ROM) are stored
    under a `{t}`-suffixed key, e.g. `rho_fom_rom0_vals`, `ss1`. ROM 0 is
    additionally aliased to the original unsuffixed key names (`ss`,
    `rho_fom_rom_vals`, ...) so single-ROM tooling keeps working unchanged.

    validation_budget_idx records which entry of budget_list ('xx') was
    actually used to fix the trained ROM basis size(s) (see Step 3), so
    postprocessing can flag that budget's bar in the surrogate-optimized
    allocation chart as the one behind the validation panel.
    """
    mf = alloc_results["mf"]
    is_ = alloc_results["is"]

    vis_data = {
        's_star': s_star,
        'xx': budget_list,
        'validation_budget_idx': validation_budget_idx,
        # Optimization results
        'fMFs': mf.vars,
        'fMFs_ex': mf.vars_ex,
        'fMFs_alloc': mf.allocs,
        'fMFs_alloc_ex': mf.allocs_ex,
        'fISs': is_.vars,
        'fISs_ex': is_.vars_ex,
        'fISs_alloc': is_.allocs,
        'fISs_alloc_ex': is_.allocs_ex,
        'n_aux': n_aux,
        'n_active': n_active,
    }

    # Add aux-specific data (independent of any ROM's basis size)
    for i in range(n_aux):
        vis_data[f'rho_fom_aux{i}_vals'] = surrogate_vals[f'rho_fom_aux{i}']
        vis_data[f'rho_fom_aux{i}_pilot'] = pilot_data.fom_aux_corrs[i]
        vis_data[f'cost_aux{i}_vals'] = surrogate_vals[f'cost_aux{i}']
        vis_data[f'cost_aux{i}_pilot'] = pilot_data.normalized_aux_times[i]

    # Add aux-aux data if multiple aux models
    for pair, i, j in aux_pairs(n_aux):
        vis_data[f'rho_aux{j}_aux{i}_vals'] = surrogate_vals[f'rho_aux{j}_aux{i}']
        vis_data[f'rho_aux{j}_aux{i}_pilot'] = pilot_data.aux_aux_corrs[pair]

    # Add one basis-size sweep per trainable ROM
    for t in range(n_active):
        vis_data[f'ss{t}'] = np.tile(s_plots[t], (2, 1))
        vis_data[f'pp{t}'] = np.tile(pilot_basis_grids[t], (2, 1))
        vis_data[f'fom_rom_corrs_pilot{t}'] = pilot_data.fom_rom_corrs_list[t]
        vis_data[f'normalized_rom_times_pilot{t}'] = pilot_data.normalized_rom_times_list[t]
        vis_data[f'rho_fom_rom{t}_vals'] = surrogate_vals[f'rho_fom_rom{t}']
        vis_data[f'cost_rom{t}_vals'] = surrogate_vals[f'cost_rom{t}']
        for i in range(n_aux):
            vis_data[f'rho_aux{i}_rom{t}_vals'] = surrogate_vals[f'rho_aux{i}_rom{t}']
            vis_data[f'rho_aux{i}_rom{t}_pilot'] = pilot_data.aux_rom_corrs_list[i][t]

    # Legacy unsuffixed aliases (ROM 0), for single-ROM tooling.
    for suffixed, legacy in [
        ('ss0', 'ss'), ('pp0', 'pp'),
        ('fom_rom_corrs_pilot0', 'fom_rom_corrs_pilot'),
        ('normalized_rom_times_pilot0', 'normalized_rom_times_pilot'),
        ('rho_fom_rom0_vals', 'rho_fom_rom_vals'), ('cost_rom0_vals', 'cost_rom_vals'),
    ]:
        vis_data[legacy] = vis_data[suffixed]
    for i in range(n_aux):
        vis_data[f'rho_aux{i}_rom_vals'] = vis_data[f'rho_aux{i}_rom0_vals']
        vis_data[f'rho_aux{i}_rom_pilot'] = vis_data[f'rho_aux{i}_rom0_pilot']

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

def _as_per_rom_list(value, n_active, default):
    """
    Normalize a per-ROM configuration argument to a length-`n_active` list
    of lists, one entry per trainable ROM.

    Accepts either:
      - the new nested form, e.g. [[1, 3, 5], [2, 4, 6]] (one list per ROM), or
      - the legacy flat form, e.g. [1, 3, 5] or [1, 20], which is broadcast
        to every ROM for backward compatibility with single-ROM call sites.
    """
    if value is None:
        value = default

    if not (isinstance(value, list) and value and isinstance(value[0], list)):
        value = [value] * n_active  # legacy flat form: broadcast to every ROM

    return value


def run_hybrid_mfuq(fom_model, aux_models, rom_model_builders, parameter_space,
                   absolute_hybrid_MFMC_work_directory: str,
                   pilot_sample_size: int = 20,
                   pilot_basis_grids: List[List[int]] = None,
                   max_combinations: int = 5,
                   tunable_ranges: List[List[int]] = None,
                   min_pair_validation_size: int = 1,
                   budget_list: List[float] = None,
                   validation_budget_idx: int = 0,
                   log_of_objective: bool = True,
                   overwrite: bool = True,
                   random_seed: int = 2025,
                   surrogate_method: str = 'ah_matrix_bandaid',
                   validation_type: str = 'IS',
                   use_torch: bool = True):
    """
    Hybrid MFUQ algorithm with multiple auxiliary models and multiple
    trainable ROMs.
    
    Workflow: pilot sampling → surrogate building → optimization → ROM training → validation
    
    Args:
        fom_model: Full-order model
        aux_models: List of auxiliary models (or single model for backward compatibility)
        rom_model_builders: List of variable-fidelity ROM builders, one per
            trainable ROM (or a single builder for backward compatibility).
            `n_active := len(rom_model_builders)`.
        parameter_space: Parameter space for sampling
        absolute_hybrid_MFMC_work_directory: Work directory path
        pilot_sample_size: Number of pilot samples
        pilot_basis_grids: ROM basis sizes to test, one grid per trainable ROM
            (List[List[int]]), or a single flat list broadcast to every ROM
            for backward compatibility.
        max_combinations: Max training combinations
        tunable_ranges: Tunable ROM basis range, one [smin, smax] per
            trainable ROM (List[List[int]]), or a single flat [smin, smax]
            broadcast to every ROM for backward compatibility.
        min_pair_validation_size: Minimum size of the held-out validation
            set for a trainable-trainable pairwise correlation combo before
            it is dropped (see Pilot.set_ROM_correlation_labels). No-op when
            n_active == 1, since no ROM pairs exist.
        budget_list: Computational budgets to sweep (list of floats, FOM
            equivalents), used exactly in the order given -- not
            reordered. Every budget is solved independently in Step 2
            (surrogate-optimized) and Step 4 (validated-with-trained-ROM);
            validation_budget_idx below selects which one actually
            determines the trained ROM basis size(s). Pass budgets in
            increasing order if you want the bar charts and budget-sweep
            curves in postprocessing to render sensibly; this is not
            enforced. Defaults to [40, 80, 120, 160, 200, 240].
        validation_budget_idx: Index into budget_list (as passed, in that
            same order) identifying which budget's surrogate-optimized
            ROM basis size(s) are the ones actually trained and carried
            into Step 3/4 validation. Supports Python-style negative
            indexing (e.g. -1 for the last entry of budget_list). Defaults
            to 0 (the first entry), matching the previous
            allocate_based_on='min' default.
        log_of_objective: Use log transform in optimization
        overwrite: Overwrite existing results
        random_seed: Random seed
        surrogate_method: 'ah_matrix_bandaid' (default), 'ah_matrix',
            'ah_componentwise_sigmoid', or 'componentwise_sigmoid'. See
            SurrogateBuilder's docstring; the default pins the fixed-fixed
            correlations to their exact pilot values per writeup Algorithm 6
            line 20, at the cost of the global admissibility guarantee.
        validation_type: which ACV scheme's Step-2 solution supplies the ROM
            basis size(s) s* that are actually trained and validated in
            Steps 3-4 -- 'IS' (default, matching prior behavior) or 'MF'.
            Both schemes are still solved and reported at every budget; this
            only selects which one's s* is realized. Note the asymmetry that
            follows: the *other* scheme's validated variance is then reported
            at basis sizes chosen by this one.
        use_torch: Use PyTorch for optimization gradients

    Note on Step 4 ROM training: the validated ROMs are built from a
    deterministic prefix of the pilot training samples (train_dirs[:s*], see
    train_optimized_roms), i.e. one specific training set rather than an
    average over training sets of that size. This is a deliberate choice --
    it is the ROM a user would actually get -- but it differs from the pilot
    heuristic of writeup §4.2, which characterizes "a ROM of basis size s" by
    averaging over training sets. Some of the prediction-vs-validation gap is
    therefore attributable to that single draw rather than to surrogate error.
    """
    # Validate inputs
    if budget_list is None:
        budget_list = [40.0, 80.0, 120.0, 160.0, 200.0, 240.0]
    budget_list = list(budget_list)
    if len(budget_list) == 0:
        raise ValueError("budget_list must contain at least one budget")
    if not (-len(budget_list) <= validation_budget_idx < len(budget_list)):
        raise ValueError(
            f"validation_budget_idx={validation_budget_idx} is out of range "
            f"for budget_list of length {len(budget_list)}"
        )
    validation_budget_idx = validation_budget_idx % len(budget_list)

    validation_type = str(validation_type).upper().replace("ACV-", "")
    if validation_type not in ("MF", "IS"):
        raise ValueError(
            f"validation_type must be 'MF' or 'IS', got {validation_type!r}"
        )

    # Setup defaults and configuration
    rom_model_builders = rom_model_builders if isinstance(rom_model_builders, list) else [rom_model_builders]
    n_active = len(rom_model_builders)
    pilot_basis_grids = _as_per_rom_list(pilot_basis_grids, n_active, [1, 3, 5, 7, 9])
    tunable_ranges = _as_per_rom_list(tunable_ranges, n_active, [1, 20])
    aux_models = aux_models if isinstance(aux_models, list) else [aux_models]
    n_aux = len(aux_models)
    
    np.random.seed(random_seed)
    work_dir = absolute_hybrid_MFMC_work_directory
    create_empty_dir(work_dir)
    
    with WorkflowLogger(os.path.join(work_dir, "hybrid_status.log")) as logger:
        logger.write("Hybrid MFUQ Configuration:")
        logger.write(
            f"  n_aux={n_aux}, n_active={n_active}, surrogate={surrogate_method}, "
            f"validation_type={validation_type}, seed={random_seed}, use_torch={use_torch}"
        )
        
        # ====================================================================
        # STEP 1: Pilot Sampling
        # ====================================================================
        logger.section("Pilot Sampling", step=1)
        
        pilot_mgr = Pilot(pilot_basis_grids, pilot_sample_size, random_seed=random_seed,
                         min_pair_validation_size=min_pair_validation_size)
        data_npz = f"{work_dir}/pilot_results.npz"
        
        if os.path.exists(data_npz):
            pilot_data = load_pilot_data(data_npz, n_aux, n_active)
            logger.write("Loaded existing pilot data")
        else:
            sampler = PilotSampler(fom_model, aux_models, rom_model_builders,
                                  parameter_space, pilot_mgr, work_dir)
            pilot_data = sampler.run(max_combinations, overwrite)
            logger.write("Completed pilot sampling")
        
        # ====================================================================
        # STEP 2: Surrogate Building and Optimization (Surrogates Only)
        # ====================================================================
        logger.section("Surrogate Building and Optimization", step=2)
        
        builder = SurrogateBuilder(pilot_basis_grids, n_active=n_active, n_aux=n_aux,
                                 work_dir=work_dir, method=surrogate_method,
                                 tunable_ranges=tunable_ranges, use_torch=use_torch)
        hf_corrs, lf_corrs, costs, corr_matrix_fn = builder.build(data_npz)
        
        # x = [N, r_1..r_{n_aux+n_active}, s_1..s_n_active]; one oversampling
        # bound per LF model (aux + all trainable ROMs) plus one basis-size
        # bound per trainable ROM.
        bounds = (
            [(1, None)]
            + [(1.001, None)] * (n_aux + n_active)
            + [tuple(tr) for tr in tunable_ranges]
        )
        
        logger.write(f"Optimization bounds: {bounds}")
        
        # Run optimization with surrogate models only (hybrid=True)
        (mf_vars, mf_allocs), (is_vars, is_allocs) = sweep_budgets(
            budget_list,
            ("MF", "IS"),
            hf_corrs,
            lf_corrs,
            costs,
            bounds,
            log_of_objective,
            use_torch,
            corr_matrix_fn,
            n_active,
        )
        
        # ====================================================================
        # STEP 3: Train Optimized ROM(s)
        # ====================================================================
        logger.section("Training Optimized ROM(s)", step=3)
        
        # Which scheme's optimizer decides the basis sizes that get built.
        # Both are still solved and reported at every budget.
        selected_allocs = {"MF": mf_allocs, "IS": is_allocs}[validation_type]
        s_star = selected_allocs[validation_budget_idx]
        rom_basis_nums = [int(round(v)) for v in s_star[-n_active:]]

        logger.write(
            f"Validating against budget_list[{validation_budget_idx}] = "
            f"{budget_list[validation_budget_idx]:g} "
            f"using the ACV-{validation_type} solution"
        )
        logger.write(f"Optimal allocation: {s_star}")
        logger.write(f"ROM basis sizes: {rom_basis_nums}")
        
        basis_tag = "-".join(str(b) for b in rom_basis_nums)
        rom_npz = f"{work_dir}/trained_{basis_tag}_sample_rom_results.npz"
        
        if os.path.exists(rom_npz):
            logger.write("Using previously trained ROM(s)")
        else:
            fom_rom_corrs, aux_rom_corrs, rom_rom_corrs, norm_rom_times = train_optimized_roms(
                fom_model, rom_model_builders, parameter_space, work_dir,
                rom_basis_nums, pilot_mgr, pilot_data, data_npz, overwrite
            )
            
            save_dict = {}
            for t in range(n_active):
                save_dict[f'rom{t}_fom_corr'] = fom_rom_corrs[t]
                save_dict[f'rom{t}_normalized_time'] = norm_rom_times[t]
                for i in range(n_aux):
                    save_dict[f'aux{i}_rom{t}_corr'] = aux_rom_corrs[i][t]
            for (t, q), corr in rom_rom_corrs.items():
                save_dict[f'rom{t}_rom{q}_corr'] = corr

            # Flat ROM-0 aliases, read by postprocess_hybrid_mfuq.py's
            # _trained_rom_statistics, _trained_rom_aux_correlations and
            # _rom_validation_cost, which are deliberately ROM-0-only.
            save_dict['fom_rom_corr'] = fom_rom_corrs[0]
            save_dict['normalized_rom_time'] = norm_rom_times[0]
            for i in range(n_aux):
                save_dict[f'aux{i}_rom_corr'] = aux_rom_corrs[i][0]
            
            np.savez(rom_npz, **save_dict)
            logger.write("Trained and saved ROM(s)")
        
        # ====================================================================
        # STEP 4: Validation with Exact Statistics
        # ====================================================================
        logger.section("Validation with Exact Statistics", step=4)
        
        with np.load(rom_npz) as data:
            fom_rom_corr_vals = [float(data[f'rom{t}_fom_corr']) for t in range(n_active)]
            aux_rom_corr_vals = [
                [float(data[f'aux{i}_rom{t}_corr']) for t in range(n_active)]
                for i in range(n_aux)
            ]
            normalized_rom_time_vals = [
                float(data[f'rom{t}_normalized_time']) for t in range(n_active)
            ]
            rom_rom_corr_vals = {}
            for t in range(n_active):
                for q in range(t + 1, n_active):
                    key = f'rom{t}_rom{q}_corr'
                    if key in data:
                        rom_rom_corr_vals[(t, q)] = float(data[key])
        
        for t in range(n_active):
            logger.write(f"FOM-ROM{t} correlation: {fom_rom_corr_vals[t]:.4f}")
            logger.write(f"Normalized ROM{t} time: {normalized_rom_time_vals[t]:.4f}")
        
        # Build exact scalar functions for validation.
        # Do not rely on hf_corrs/lf_corrs here, because the ah_matrix
        # surrogate backend returns hf_corrs=None and lf_corrs=None.
        def make_constant(value):
            value = float(np.asarray(value).squeeze())

            if use_torch:
                def const_fn(s, v=value):
                    if torch.is_tensor(s):
                        return torch.tensor(v, dtype=torch.float64, device=s.device)
                    return v
                return const_fn

            return lambda s, v=value: v

        # HF correlations:
        #   Corr[FOM, aux_i] from pilot data,
        #   Corr[FOM, trained ROM_t] from trained ROM validation, one per t.
        exact_hf = [
            make_constant(corr) for corr in np.asarray(pilot_data.fom_aux_corrs).ravel()
        ]
        exact_hf.extend(make_constant(c) for c in fom_rom_corr_vals)

        # LF-LF correlations, assembled over the n_lofi = n_aux + n_active
        # LF-LF correlations, in the flat order MFMC.build_C reads back.
        aux_aux_vals = np.asarray(pilot_data.aux_aux_corrs).ravel()
        n_lofi = n_aux + n_active

        exact_lf = []
        for i, j, kind, t, q in lf_pairs(n_aux, n_active):
            if kind == "aux_aux":
                exact_lf.append(
                    make_constant(float(aux_aux_vals[tril_position(i, j)]))
                )
            elif kind == "aux_rom":
                exact_lf.append(make_constant(aux_rom_corr_vals[j][t]))
            else:
                exact_lf.append(
                    make_constant(rom_rom_corr_vals.get((t, q), 0.0))
                )

        # Costs:
        #   auxiliary model costs from pilot data,
        #   trained ROM_t cost from validation, one per t.
        exact_costs = [
            make_constant(cost) for cost in np.asarray(pilot_data.normalized_aux_times).ravel()
        ]
        exact_costs.extend(make_constant(c) for c in normalized_rom_time_vals)
        
        bounds_exact = (
            [(1, None)]
            + [(1.001, None)] * n_lofi
            + [(b, b) for b in rom_basis_nums]
        )
        
        # Run optimization with exact ROM statistics (hybrid=True)
        (mf_vars_ex, mf_allocs_ex), (is_vars_ex, is_allocs_ex) = sweep_budgets(
            budget_list,
            ("MF", "IS"),
            exact_hf,
            exact_lf,
            exact_costs,
            bounds_exact,
            log_of_objective,
            use_torch,
            None,
            n_active,
        )
        
        # ====================================================================
        # STEP 5: Generate Visualization Data
        # ====================================================================
        n_vis = 200
        s_plots = [
            np.linspace(1, tunable_ranges[t][-1] + 1, n_vis) for t in range(n_active)
        ]

        surrogate_vals = evaluate_surrogates_for_plots(
            hf_corrs, lf_corrs, costs, corr_matrix_fn,
            rom_basis_nums, s_plots, n_aux, n_active,
        )
        
        alloc_results = {
            "mf": AllocationResults(
                vars=mf_vars, vars_ex=mf_vars_ex,
                allocs=mf_allocs, allocs_ex=mf_allocs_ex,
            ),
            "is": AllocationResults(
                vars=is_vars, vars_ex=is_vars_ex,
                allocs=is_allocs, allocs_ex=is_allocs_ex,
            ),
        }

        vis_data = build_visualization_dict(
            pilot_data, surrogate_vals, s_star, budget_list,
            alloc_results, pilot_basis_grids, s_plots, n_aux, n_active,
            validation_budget_idx,
        )
        
        vis_path = f"{work_dir}/visualization_data.npz"
        np.savez(vis_path, **vis_data)
        logger.write(f"Saved visualization data to {vis_path}")
        
        # ====================================================================
        # Complete
        # ====================================================================
        logger.section("HYBRID MFUQ COMPLETE")
        logger.write("Workflow completed successfully")