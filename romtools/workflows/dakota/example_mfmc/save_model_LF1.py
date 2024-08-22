import sys
import subprocess
import os
import numpy as np
import yaml
import romtools as rt
from romtools.workflows.models import QoiModel
import pickle
import pathlib

path_to_file = str(pathlib.Path(__file__).parent.resolve())

class PDA_Model_LF1(rt.workflows.models.QoiModel):
    def __init__(self):
        print("Initializing LF1")
        self.path = str(pathlib.Path(__file__).parent.resolve())
        self.template_directory = self.path + '/templates'
        self.template_file = "parameters.yaml"
        return None

    def populate_run_directory(self, run_directory, parameter_sample):
        # Grab template yaml
        with open(self.template_directory+"/"+self.template_file) as f:
            input_yaml = yaml.safe_load(f)
        
        input_yaml['D'] = float(parameter_sample['D'])
        input_yaml['k'] = float(parameter_sample['k'])

        # Write yaml into run_directory 
        with open(run_directory + "/" + self.template_file,"w") as f:
            yaml.dump(input_yaml,f,default_flow_style=False)

        return 0

    def run_model(self, run_directory, parameter_sample):
        print("Running model in directory " + str(run_directory))
        os.system('python ' + run_directory + '/../../model_LF1.py')
        return 0 # NOTE: This function must return 0

    def compute_qoi(self, run_directory, parameter_sample):
        data = np.load('solution.npz')
        return np.array([np.amax(data['u'])])

if __name__ == "__main__":
    # Define FOM
    fom_model = PDA_Model_LF1()

    with open('LF1.pickle', 'wb') as handle:
        pickle.dump(fom_model, handle, protocol=pickle.HIGHEST_PROTOCOL)