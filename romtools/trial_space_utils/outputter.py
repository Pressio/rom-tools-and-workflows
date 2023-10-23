import numpy as np
import os
from romtools.trial_space import AbstractTrialSpace

try:
  import exodus
except:
  pass

import math
import h5py

def npz_output(filename: str, trial_space: AbstractTrialSpace, compress=True) -> None:
    '''
    Save trial space information to a compressed or uncompressed NumPy .npz file.

    Args:
        filename (str): The name of the output file.
        trial_space (AbstractTrialSpace): The trial space containing shift and basis information.
        compress (bool, optional): Whether to compress the output file (default is True).

    Example:
        npz_output("trial_space.npz", my_trial_space)
    '''
    if compress:
        np.savez_compressed(filename, shift=trial_space.getShiftVector(),
            basis=trial_space.getBasis())
    else:
        np.savez(filename, shift=trial_space.getShiftVector(),
            basis=trial_space.getBasis())

def hdf5_output(output_filename: str, trial_space: AbstractTrialSpace) -> None:
    '''
    Save trial space information to an HDF5 file.

    Args:
        output_filename (str): The name of the output HDF5 file.
        trial_space (AbstractTrialSpace): The trial space containing shift and basis information.

    Example:
        hdf5_output("trial_space.h5", my_trial_space)
    '''
    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('shift', data=trial_space.getShiftVector())
        f.create_dataset('basis', data=trial_space.getBasis())

def exodus_ouput(output_filename: str, mesh_filename: str, trial_space: AbstractTrialSpace, var_names: list = None) -> None:
    '''
    Save trial space information to an Exodus file.

    Args:
        output_filename (str): The name of the output Exodus file.
        mesh_filename (str): The name of the mesh file.
        trial_space (AbstractTrialSpace): The trial space containing shift and basis information.
        var_names (list, optional): A list of variable names (default is None).

    Example:
        exodus_output("trial_space.e", "mesh.exo", my_trial_space, var_names=["var1", "var2"])
    '''
    if os.path.isfile(output_filename):
      os.system(f'rm {output_filename}')

    e = exodus.copy_mesh(mesh_filename, output_filename)
    e.close()
    e = exodus.exodus(output_filename, mode='a')
    num_nodes = e.num_nodes()
    num_vars = int(len(trial_space.getShiftVector())/num_nodes)
    num_modes = trial_space.getBasis().shape[1]
    num_modes_str_len = int(math.log10(num_modes))+1

    assert var_names is None or len(var_names)==num_vars, f"len(variable_names), {len(var_names)} != number of variables in basis, {num_vars}"

    if var_names is None:
        var_names = [f"{i}" for i in range(num_vars)]

    field_names = []
    for var_name in var_names:
        field_names.append(f"u0_{var_name}")
        for j in range(num_modes):
            mode_str = str.zfill(str(j+1), num_modes_str_len)
            field_names.append(f"phi_{var_name}_{mode_str}")
    exodus.add_variables(e, nodal_vars=field_names)

    for i in range(num_vars):
        values = trial_space.getShiftVector()[i*num_nodes:(i+1)*num_nodes]
        field_name = field_names[i*(num_modes+1)]
        e.put_node_variable_values(field_name, 1, values)

        for j in range(num_modes):
            field_name = field_names[i*(num_modes+1) + j]
            values = trial_space.getBasis()[i*num_nodes:(i+1)*num_nodes,j]
            e.put_node_variable_values(field_name, 1, values)

    e.close()

