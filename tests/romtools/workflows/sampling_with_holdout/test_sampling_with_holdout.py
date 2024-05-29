import pytest
import os
import numpy as np
from romtools.workflows.models import *
from romtools.workflows.model_builders import *
from romtools.workflows.sampling_with_holdout import run_sampling_with_holdout
from romtools.workflows.parameter_spaces import MonteCarloSampler, UniformParameterSpace


class MockFOMQoiModel:
    def __init__(self):
        self.my_qois_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.counter_ = 0

    def populate_run_directory(self, run_dir, parameter_sample):
        os.chdir(run_dir)
        parameter_values = np.zeros(0)
        for parameter_name in list(parameter_sample.keys()):
            parameter_values = np.append(
                parameter_values, parameter_sample[parameter_name]
            )
        np.savez("parameter_values.npz", parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample):
        os.chdir(run_dir)
        params_input = np.load("parameter_values.npz")["parameter_values"]
        for i in range(0, len(parameter_sample)):
            parameter_name = list(parameter_sample.keys())[i]
            assert params_input[i] == parameter_sample[parameter_name]
        np.savetxt("fom_succesful.dat", np.array([0]), "%i")
        return 0

    def compute_qoi(self, run_dir, parameter_sample):
        self.counter_ += 1
        return self.my_qois_[self.counter_ - 1]


class MockQoiModelBuilder:
    def __init__(self):
        self.counter_ = 0
        self.my_qois_ = np.array([1.4, 1.5, 1.3, 0.7, 1.1, 1.01, 1.0 + 1e-7])

    def build_from_training_dirs(self, offline_data_dir, training_data_dirs):
        rom_model = MockROMQoiModel(self.my_qois_[self.counter_])
        np.savetxt(f"{offline_data_dir}/offline_data.dat", np.array([0]), "%i")
        self.counter_ += 1
        return rom_model

    def reset_counter(self):
        self.counter_ = 0


class MockROMQoiModel:
    def __init__(self, my_qoi):
        self.counter = 0
        self.my_qoi_ = my_qoi

    def populate_run_directory(self, run_dir, parameter_sample):
        os.chdir(run_dir)
        parameter_values = np.zeros(0)
        for parameter_name in list(parameter_sample.keys()):
            parameter_values = np.append(
                parameter_values, parameter_sample[parameter_name]
            )
        np.savez("parameter_values.npz", parameter_values=parameter_values)

    def run_model(self, run_dir, parameter_sample):
        os.chdir(run_dir)
        params_input = np.load("parameter_values.npz")["parameter_values"]
        for i in range(0, len(parameter_sample)):
            parameter_name = list(parameter_sample.keys())[i]
            assert params_input[i] == parameter_sample[parameter_name]
        np.savetxt("passed.txt", np.array([0]), "%i")
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> float:
        return self.my_qoi_


@pytest.mark.mpi_skip
def test_sampling_with_holdout(tmp_path):
    # see https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html for more info
    #   about tmp_path
    wdir = str(tmp_path)  # does not like posixpaths
    print("\n", wdir)

    my_dir = os.path.realpath(os.path.dirname(__file__))

    QoiModel = MockFOMQoiModel()
    RomModelBuilder = MockQoiModelBuilder()

    my_parameter_space = UniformParameterSpace(
        ["u", "v", "w"],
        np.array([0, 1, 2]),
        np.array([1, 2, 3]),
        sampler=MonteCarloSampler,
    )

    ## First, test that the routine will terminate if it hits the max iters
    run_sampling_with_holdout(
        QoiModel, RomModelBuilder, my_parameter_space, wdir, 4, 3, 1e-5, 1
    )

    # Ensure correct number of holdout set and training set FOMs are run
    for sample in range(0, 4):
        assert os.path.isfile(
            f"{wdir}/fom/holdout_set/run_{sample}/fom_succesful.dat"
        ), sample
    for sample in range(0, 2):
        assert os.path.isfile(
            f"{wdir}/fom/training_set/run_{sample}/fom_succesful.dat"
        ), sample
    for sample in range(3, 7):
        assert not os.path.isfile(
            f"{wdir}/fom/training_set/run_{sample}/fom_succesful.dat"
        ), sample
    # Check that right number of ROMs were created
    for i in range(0, 3):
        assert os.path.isfile(f"{wdir}/rom_iteration_{i}/offline_data/offline_data.dat")

    holdout_output = np.load(f"{wdir}/holdout_stats.npz")
    assert np.allclose(holdout_output["holdout_set_errs"], np.array([0.4, 0.5, 0.3]))
    assert np.allclose(holdout_output["trained_samples"], np.array([0, 1, 2]))

    ## Then, test that the method will terminate after reaching the error tolerance on the 7th iter
    RomModelBuilder.reset_counter()  # Reset mock ROM model QoI's
    run_sampling_with_holdout(
        QoiModel, RomModelBuilder, my_parameter_space, wdir, 5, 20, 1e-5, 1
    )

    # Ensure correct number of holdout set and training set FOMs are run
    for sample in range(0, 5):
        assert os.path.isfile(
            f"{wdir}/fom/holdout_set/run_{sample}/fom_succesful.dat"
        ), sample
    for sample in range(0, 7):
        assert os.path.isfile(
            f"{wdir}/fom/training_set/run_{sample}/fom_succesful.dat"
        ), sample
    for sample in range(7, 8):
        assert not os.path.isfile(
            f"{wdir}/fom/training_set/run_{sample}/fom_succesful.dat"
        ), sample
    # Check that right number of ROMs were created
    for i in range(0, 7):
        assert os.path.isfile(f"{wdir}/rom_iteration_{i}/offline_data/offline_data.dat")

    holdout_output = np.load(f"{wdir}/holdout_stats.npz")
    assert np.allclose(
        holdout_output["holdout_set_errs"],
        np.array([0.4, 0.5, 0.3, 0.3, 0.1, 0.01, 1e-7]),
    )
    assert np.allclose(
        holdout_output["trained_samples"], np.array([0, 1, 2, 3, 4, 5, 6])
    )


if __name__ == "__main__":
    test_sampling_with_holdout(os.getcwd() + "/sampling_with_holdout_test_tmp/")
