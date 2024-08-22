import sys
import subprocess
import os
import numpy as np
import yaml
import romtools as rt
from romtools.workflows.models import QoiModel
import pickle
import pathlib
from romtools.workflows.dakota.dakota_coupler import run_model_for_dakota

path_to_file = str(pathlib.Path(__file__).parent.resolve())


class PDA_Model(rt.workflows.models.QoiModel):
    def __init__(self):
        self.path = str(pathlib.Path(__file__).parent.resolve())
        self.template_directory = self.path + "/templates"
        self.template_file = "parameters.yaml"
        # print(self.template_file)
        return None

    def populate_run_directory(self, run_directory, parameter_sample):
        # Grab template yaml
        print("Is this running?")
        with open(self.template_directory + "/" + self.template_file) as f:
            input_yaml = yaml.safe_load(f)

        input_yaml["D"] = float(parameter_sample["D"])
        input_yaml["k"] = float(parameter_sample["k"])

        # Write yaml into run_directory
        with open(run_directory + "/" + self.template_file, "w") as f:
            yaml.dump(input_yaml, f, default_flow_style=False)

        return 0

    def run_model(self, run_directory, parameter_sample):
        print("Running model in directory " + str(run_directory))
        os.system("python " + run_directory + "/../../model.py")
        return 0  # NOTE: This function must return 0

    def compute_qoi(self, run_directory, parameter_sample):
        data = np.load("solution.npz")
        return np.array([np.amax(data["u"])])


# Define FOM
fom_model = PDA_Model()

# Run coupler
run_model_for_dakota(fom_model, multifidelity_flag=True, add_core_time_metadata=True)
