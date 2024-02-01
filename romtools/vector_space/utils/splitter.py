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

'''
The Splitter class is used to "split" a snapshot matrix before constructing the vector space.

*What do we mean by splitting a vector space, and why do it?*
Vector space splitting can be important for vector-valued equations where there are multiple state variables. As an
example, we could consider the compressible Euler equations in one dimension in which case the spatially continuous
(conserved) variable set is
$$\\boldsymbol u(x) = \\begin{bmatrix} \\rho(x) &  \\rho u (x)& \\rho E(x) \\end{bmatrix}^T.$$
There are several ways of constructing ROMs of vector-valued systems. In one approach, we simply stack all of our
degrees of freedom and create a global basis. Mathematically, this comprises constructing a basis function that looks
as follows
$$
\\boldsymbol \\Phi(x) = \\begin{bmatrix}
\\Phi_{1}^{\\rho}(x) & \\Phi_{2}^{\\rho }(x) & \\cdots & \\Phi_{K}^{\\rho}(x) \\\\\\\\
\\Phi_{1}^{\\rho u}(x) & \\Phi_{2}^{\\rho u }(x) & \\cdots & \\Phi_{K}^{\\rho u}(x) \\\\\\
\\Phi_{1}^{\\rho E}(x) & \\Phi_{2}^{\\rho E }(x) & \\cdots & \\Phi_{K}^{\\rho E}(x) \\
\\end{bmatrix},
$$
and the state is approximated by
$$\\tilde{\\boldsymbol u}(x) = \\sum_{i=1}^{K} \\Phi_i(x) \\hat{\\boldsymbol u}(t).$$
With this representation, the different state variables share generalized coordinates $\\hat{\\boldsymbol u}(t)$.

An alternative approach is to ensure that each state variable is linearly
independent of the other by employing a block basis. With this approach, this
comprises a basis function that looks like
$$
\\boldsymbol \\Phi(x) = \\begin{bmatrix}
\\Phi_{1}^{\\rho}(x) &  0                     &  0  & \\Phi_{2}^{\\rho }(x) & \\cdots \\\\\\\\
 0                   & \\Phi_{1}^{\\rho u }(x) &  0 & 0 & \\cdots \\\\\\
 0                   & 0                      & \\Phi_{1}^{\\rho E}(x) & 0 & \\cdots \\
\\end{bmatrix}.
$$
This "split" representation can be particularly advantageous for problems
where different state variables span numerous orders of magnitude.

The splitting class encapsulates this functionality. We note that the splitter is **applied to the snapshot matrix
before POD is performed**.

'''
import abc
import numpy as np


class Splitter(abc.ABC):
    '''Abstract class for a splitter'''

    @abc.abstractmethod
    def __call__(self, my_array: np.ndarray) -> np.ndarray:
        pass


class NoOpSplitter(Splitter):
    '''Concrete no-op implementation'''
    def __init__(self) -> None:
        pass

    def __call__(self, my_array: np.ndarray):
        return my_array


class BlockSplitter(Splitter):
    '''
    Splits a data matrix into blocks defined by a list, e.g., for our Euler
    equation example above, we could set
    blocks = [[0,1],[2]] which would result in
    $$
    \\boldsymbol \\Phi(x) = \\begin{bmatrix}
    \\Phi_{1}^{\\rho}(x)   &  0 & \\Phi_{2}^{\\rho}(x) &  0 & \\cdots  \\\\\\\\
    \\Phi_{1}^{\\rho u}(x) &  0 & \\Phi_{2}^{\\rho u}(x) & 0 & \\cdots \\\\\\
    0                      & \\Phi_{1}^{\\rho E }(x) & 0 & \\Phi_{2}^{\\rho E }(x) & \\cdots \\
    \\end{bmatrix}.
    $$

    **Notes on variable ordering**

    **Order F variable ordering: $[u_1,v_1,w_1,u_2,v_2,w_2,...,u_n,v_n,w_n]$**

    **Order C variable ordering: $[u_1,u_2,...,u_n,v_1,...]$**
    '''
    def __init__(self, blocks: list, n_var: int, variable_ordering: str = 'C'):
        '''
        Constructor for BlockSplitter.

        Args:
            blocks (list): A list of lists, where each inner list contains variable indices to form a block.
            n_var (int): The total number of variables in the data matrix.
            variable_ordering (str): The variable ordering, either 'C' or 'F'.
        '''
        assert variable_ordering in ('C', 'F'), "Invalid variable ordering, options are F and C"
        self.__variable_ordering = variable_ordering
        self.__n_var = n_var
        self.__n_blocks = len(blocks)
        self.__blocks = blocks
        # Check that the block list is valid
        my_vars = np.zeros(0, dtype='int')
        for block in blocks:
            for var in block:
                my_vars = np.append(my_vars, var)
        my_vars.sort()
        assert np.allclose(my_vars, np.arange(0, self.__n_var, dtype='int')), "Invalid block input"

    def __call__(self, my_array: np.ndarray):
        '''
        Splits the input data matrix into smaller blocks as defined by the
        block list.

        Args:
            my_array (np.ndarray): The input data array.

        Returns:
            np.ndarray: A data matrix split into blocks as defined by the input blocks.
        '''
        d1 = my_array.shape[0]
        d2 = my_array.shape[1]
        n = int(d1/self.__n_var)
        new_d2 = d2*self.__n_blocks
        my_split_array = np.zeros((d1, new_d2))
        for block_counter, block in enumerate(self.__blocks):
            start_col = block_counter*d2
            end_col = (block_counter+1)*d2
            for var in block:
                if self.__variable_ordering == 'F':
                    my_split_array[var::self.__n_var, start_col:end_col] = get_data_matrix_for_ith_var(var, self.__n_var, my_array, self.__variable_ordering)
                elif self.__variable_ordering == 'C':
                    start_index = var*n
                    end_index = (var+1)*n
                    my_split_array[start_index:end_index, start_col:end_col] = get_data_matrix_for_ith_var(var, self.__n_var, my_array, self.__variable_ordering)
        return my_split_array


def get_data_matrix_for_ith_var(i, n_var, data_matrix, variable_ordering='C'):
    '''
    Helper function to split a data matrix based on variable ordering.

    This function takes an input data matrix and splits it based on the
    variable ordering ('C' or 'F') and the variable index 'i'.

    Args:
        i (int): The index of the variable to extract.
        n_var (int): The total number of variables in the data matrix.
        data_matrix (np.ndarray): The input data matrix to split.
        variable_ordering (str): The variable ordering, either 'C' or 'F'.

    Returns:
        np.ndarray: A subset of the input data matrix corresponding to the
        specified variable index 'i'.
    '''
    if variable_ordering == 'F':
        return data_matrix[i::n_var]
    if variable_ordering == 'C':
        n = int(data_matrix.shape[0] / n_var)
        start_index = i*n
        end_index = (i+1)*n
        return data_matrix[start_index:end_index]
    raise ValueError(f'Invalid variable Ordering {variable_ordering}')