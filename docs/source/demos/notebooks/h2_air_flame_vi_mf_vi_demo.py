"""Compare FOM VI and GP auto-ROM MF-VI for the H2-air flame model."""
import shutil
import sys
from pathlib import Path
import numpy as np
from romtools.workflows.inverse.vi_drivers import run_vi
from romtools.workflows.inverse.mf_vi_drivers import mf_vi_with_auto_rom
from romtools.workflows.inverse.vi_optimization_methods import VINewtonOptimizerConfig, VIStochasticNonmonotoneLineSearchConfig
from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler
from vi_demo_helpers import collect_vi_history, write_vi_plots

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "models"))
from h2_air_flame_model import H2AirFlameQoiModel  # noqa: E402


def main():
    model = H2AirFlameQoiModel(nx=64, ny=32, dt=1e-3, t_end=6e-2, snapshot_stride=10)
    truth = {"kappa": 2.0, "scaled_activation_energy": 8.0, "beta_x": 40.0, "beta_y": 7.0}
    mins, maxes = np.array([0.5, 4.0, 20.0, 1.0]), np.array([4.0, 12.0, 60.0, 20.0])
    names = list(truth)
    prior = GaussianParameterSpace(names, (mins + maxes) / 2, (maxes - mins) / 4, MonteCarloSampler)
    root = Path(__file__).resolve().parent / "h2_air_flame_vi_mf_vi_work"
    shutil.rmtree(root, ignore_errors=True); truth_dir = root / "truth"; truth_dir.mkdir(parents=True)
    model.populate_run_directory(str(truth_dir), truth); model.run_model(str(truth_dir), truth)
    observations = model.compute_qoi(str(truth_dir), truth)
    optimizer = VINewtonOptimizerConfig(max_iterations=50, newton_hessian_type="full", newton_curvature_strategy="lagged", newton_hessian_averaging_factor=0.5)
    line_search = VIStochasticNonmonotoneLineSearchConfig()
    common = dict(model=model, prior_parameter_space=prior, observations=observations, observations_covariance=np.eye(observations.size) * 1e-4, parameter_mins=mins, parameter_maxes=maxes, optimizer_method="newton", optimizer_config=optimizer, line_search_method="stochastic_nonmonotone", line_search_config=line_search, restart_files_to_keep=50)
    vi_dir, mf_dir = root / "vi", root / "mf_vi"
    run_vi(**common, absolute_vi_directory=str(vi_dir), sample_size=8, evaluation_concurrency=4)
    mf_vi_with_auto_rom(**common, absolute_vi_directory=str(mf_dir), fom_sample_size=8, rom_extra_sample_size=64, fom_evaluation_concurrency=4, rom_type="gp", rom_args={"normalize_parameters": True, "normalize_targets": True})
    write_vi_plots(Path(__file__).resolve().parent, "h2_air_flame_vi_mf_vi", names, np.array([truth[name] for name in names]), collect_vi_history(vi_dir), collect_vi_history(mf_dir))


if __name__ == "__main__": main()
