"""Reference-adapter self-check and intentional legacy API failure."""
from pathlib import Path
import numpy as np
from benchmarks.agent_skills.tasks.cdr_model_adapter.legacy_cdr import LegacyCDR
from benchmarks.agent_skills.reference.cdr_model_adapter import CDRModel, run_experiment


def test_legacy_is_not_model():
    legacy = LegacyCDR()
    assert not hasattr(legacy, "populate_run_directory")
    assert not hasattr(legacy, "run_model")
    assert not hasattr(legacy, "compute_qoi")


def test_reference_adapter(tmp_path):
    model = CDRModel()
    params = {"nu": .025, "sigma": .2}
    run = tmp_path / "single"
    model.populate_run_directory(str(run), params)
    assert model.run_model(str(run), params) == 0
    ref = LegacyCDR()
    state = np.load(run / "solution.npy")
    np.testing.assert_allclose(state, ref.solve(.025, .2), rtol=1e-11)
    np.testing.assert_allclose(model.compute_qoi(str(run), params),
                               [ref.boundary_flux(state)], rtol=1e-11)


def test_reference_sampling(tmp_path):
    work = tmp_path / "sampling"
    dirs = run_experiment(str(work), number_of_samples=4)
    assert len(dirs) == 4
    samples = np.loadtxt(work / "sample_parameters.txt")
    assert samples.shape == (4, 2)
    qois = np.load(work / "sampling_stats.npz")["qoi_values"]
    ref = LegacyCDR()
    for i, d in enumerate(dirs):
        assert (Path(d) / "passed.txt").exists()
        state = np.load(Path(d) / "solution.npy")
        # Samples are written with limited decimal precision: compare to
        # the reference at tolerance consistent with that serialization.
        np.testing.assert_allclose(state, ref.solve(*samples[i]), rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(qois[i, 0], ref.boundary_flux(state), rtol=1e-10)
