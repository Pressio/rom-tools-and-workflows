from pathlib import Path
import importlib.util
import numpy as np
HERE = Path(__file__).resolve().parents[1]

def solution():
    path = HERE / "solution.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("cdr_benchmark_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_model_protocol_and_qoi(tmp_path):
    from benchmarks.agent_skills.tasks.cdr_model_adapter.legacy_cdr import LegacyCDR
    model = solution().CDRModel(nx=8, ny=8)
    params = {"nu": .025, "sigma": .2}
    run = str(tmp_path / "single")
    Path(run).mkdir()
    model.populate_run_directory(run, params)
    assert model.run_model(run, params) == 0
    state = np.load(Path(run) / "solution.npy")
    ref = LegacyCDR()
    np.testing.assert_allclose(state, ref.solve(params["nu"], params["sigma"]), rtol=1e-11)
    np.testing.assert_allclose(np.asarray(model.compute_qoi(run, params)).reshape(-1), [ref.boundary_flux(state)], rtol=1e-11)

def test_sampling_workflow(tmp_path):
    from benchmarks.agent_skills.tasks.cdr_model_adapter.legacy_cdr import LegacyCDR
    work = tmp_path / "samples"
    directories = solution().run_experiment(str(work), number_of_samples=4)
    assert len(directories) == 4
    samples = np.loadtxt(work / "sample_parameters.txt")
    assert samples.shape == (4, 2)
    assert np.all((samples[:, 0] >= .01) & (samples[:, 0] <= .05))
    assert np.all((samples[:, 1] >= .1) & (samples[:, 1] <= .5))
    ref = LegacyCDR()
    for i, d in enumerate(directories):
        directory = Path(d)
        assert (directory / "passed.txt").is_file()
        np.testing.assert_allclose(np.load(directory / "solution.npy"), ref.solve(*samples[i]), rtol=1e-10)
    qois = np.load(work / "sampling_stats.npz")["qoi_values"]
    assert qois.shape == (4, 1)
    np.testing.assert_allclose(qois[:, 0], [ref.boundary_flux(ref.solve(*s)) for s in samples], rtol=1e-10)
