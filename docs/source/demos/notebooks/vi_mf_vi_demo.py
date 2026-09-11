"""Compare FOM VI and GP auto-ROM MF-VI for the CDR model."""
import shutil
from pathlib import Path
import numpy as np
from romtools.workflows.inverse.vi_drivers import run_vi
from romtools.workflows.inverse.mf_vi_drivers import mf_vi_with_auto_rom
from romtools.workflows.inverse.vi_optimization_methods import VINewtonOptimizerConfig, VIStochasticNonmonotoneLineSearchConfig
from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler
from eki_mf_eki_demo import CdrFomQoiModel, cdr
from vi_demo_helpers import collect_vi_history, write_vi_plots


def main():
    system, b_vec = cdr.AdvectionDiffusionSystem(Nx=25, Ny=25), np.array([1.0, 1.0])
    truth = {"nu": 0.04, "sigma": 0.3}
    model = CdrFomQoiModel(system, b_vec)
    truth_dir = Path(__file__).resolve().parent / "vi_mf_vi_work" / "truth"
    shutil.rmtree(truth_dir.parent, ignore_errors=True); truth_dir.mkdir(parents=True)
    model.populate_run_directory(str(truth_dir), truth); model.run_model(str(truth_dir), truth)
    observations = model.compute_qoi(str(truth_dir), truth)
    mins, maxes = np.array([0.01, 0.1]), np.array([0.08, 0.6])
    prior = GaussianParameterSpace(["nu", "sigma"], (mins + maxes) / 2, (maxes - mins) / 4, MonteCarloSampler)
    optimizer = VINewtonOptimizerConfig(max_iterations=50, newton_hessian_type="full", newton_curvature_strategy="lagged", newton_hessian_averaging_factor=0.5)
    line_search = VIStochasticNonmonotoneLineSearchConfig()
    common = dict(model=model, prior_parameter_space=prior, observations=observations, observations_covariance=np.eye(observations.size) * 1e-5, parameter_mins=mins, parameter_maxes=maxes, optimizer_method="newton", optimizer_config=optimizer, line_search_method="stochastic_nonmonotone", line_search_config=line_search, restart_files_to_keep=50)
    root = truth_dir.parent
    vi_dir, mf_dir = root / "vi", root / "mf_vi"
    run_vi(**common, absolute_vi_directory=str(vi_dir), sample_size=8, evaluation_concurrency=1)
    mf_vi_with_auto_rom(**common, absolute_vi_directory=str(mf_dir), fom_sample_size=8, rom_extra_sample_size=64, fom_evaluation_concurrency=1, rom_type="gp", rom_args={"normalize_parameters": True, "normalize_targets": True})
    write_vi_plots(Path(__file__).resolve().parent, "cdr_vi_mf_vi", prior.get_names(), np.array([truth[name] for name in prior.get_names()]), collect_vi_history(vi_dir), collect_vi_history(mf_dir))


if __name__ == "__main__": main()
