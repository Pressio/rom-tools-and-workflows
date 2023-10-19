'''
There are a number of ways to construct a trial space, including affine offsets, scaling, splitting, etc.
The trial_space_utils module provides these functionalities
'''
from romtools.trial_space_utils.shifter import *
from romtools.trial_space_utils.scaler import *
from romtools.trial_space_utils.orthogonalizer import *
from romtools.trial_space_utils.splitter import *
from romtools.trial_space_utils.truncater import *
from romtools.trial_space_utils.svd_method_of_snapshots import *
