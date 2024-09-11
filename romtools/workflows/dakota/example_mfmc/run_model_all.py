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
from save_model_HF import PDA_Model_HF
from save_model_LF1 import PDA_Model_LF1
from save_model_LF2 import PDA_Model_LF2

# Use Dakota input file "ModelForm" variable to determine model
data = np.genfromtxt(sys.argv[1], encoding=None, delimiter=" ")
data_col = data[:, 0]
num_vars = int(data_col[0])
model_flag = int(data_col[num_vars])

if model_flag == 0:
    model_path = "../../HF.pickle"
elif model_flag == 1:
    model_path = "../../LF1.pickle"
elif model_flag == 2:
    model_path = "../../LF2.pickle"

with open(model_path, "rb") as handle:
    qoi_model = pickle.load(handle)

run_model_for_dakota(qoi_model, multifidelity_flag=True, add_core_time_metadata=True)
