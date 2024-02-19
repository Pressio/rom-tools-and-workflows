#
# ************************************************************************
#
#                         ROM Tools and Workflows
# Copyright 2019 National Technology & Engineering Solutions of Sandia,LLC
#                              (NTESS)
#
# Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
#
# ROM Tools and Workflows is licensed under BSD-3-Clause terms of use:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact Eric Parish (ejparis@sandia.gov)
#
# ************************************************************************
#

import os
import math
import numpy as np
from romtools.vector_space import VectorSpace

try:
    import exodus
except ImportError:
    pass

try:
    import h5py
except ImportError:
    pass


def npz_output(output_filename: str, vector_space: VectorSpace, compress=True) -> None:
    '''
    Save vector space information to a compressed or uncompressed NumPy .npz file.

    Args:
        filename (str): The name of the output file.
        vector_space (VectorSpace): The vector space containing shift and basis information.
        compress (bool, optional): Whether to compress the output file (default is True).

    Example:
        npz_output("vector_space.npz", my_vector_space)
    '''
    if compress:
        np.savez_compressed(output_filename,
                            shift=vector_space.get_shift_vector(),
                            basis=vector_space.get_basis())
    else:
        np.savez(output_filename,
                 shift=vector_space.get_shift_vector(),
                 basis=vector_space.get_basis())


def hdf5_output(output_filename: str, vector_space: VectorSpace) -> None:
    '''
    Save vector space information to an HDF5 file.

    Args:
        output_filename (str): The name of the output HDF5 file.
        vector_space (VectorSpace): The vector space containing shift and basis information.

    Example:
        hdf5_output("vector_space.h5", my_vector_space)
    '''
    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('shift', data=vector_space.get_shift_vector())
        f.create_dataset('basis', data=vector_space.get_basis())


def exodus_ouput(output_filename: str, mesh_filename: str, vector_space: VectorSpace, var_names: list = None) -> None:
    '''
    Save vector space information to an Exodus file.

    Args:
        output_filename (str): The name of the output Exodus file.
        mesh_filename (str): The name of the mesh file.
        vector_space (VectorSpace): The vector space containing shift and basis information.
        var_names (list, optional): A list of variable names (default is None).

    Example:
        exodus_output("vector_space.e", "mesh.exo", my_vector_space, var_names=["var1", "var2"])
    '''
    if os.path.isfile(output_filename):
        os.system(f'rm {output_filename}')

    e = exodus.copy_mesh(mesh_filename, output_filename)
    e.close()
    e = exodus.exodus(output_filename, mode='a')

    num_vars = vector_space.get_shift_vector().shape[0]
    num_modes = vector_space.get_basis().shape[2]
    num_modes_str_len = int(math.log10(num_modes))+1

    if var_names is None:
        var_names = [f"{i}" for i in range(num_vars)]

    assert (len(var_names) == num_vars), (
            f"len(variable_names), {len(var_names)} "
            f"!= number of variables in basis, {num_vars}"
    )

    field_names = []
    for var_name in var_names:
        field_names.append(f"u0_{var_name}")
        for j in range(num_modes):
            mode_str = str.zfill(str(j+1), num_modes_str_len)
            field_names.append(f"phi_{var_name}_{mode_str}")
    exodus.add_variables(e, nodal_vars=field_names)

    for i in range(num_vars):
        shift = vector_space.get_shift_vector()[i, :]
        field_name = field_names[i*(num_modes+1)]
        e.put_node_variable_values(field_name, 1, shift)

        basis = vector_space.get_basis()
        for j in range(num_modes):
            field_name = field_names[i*(num_modes+1) + j + 1]
            e.put_node_variable_values(field_name, 1, basis[i, :, j])

    e.close()
