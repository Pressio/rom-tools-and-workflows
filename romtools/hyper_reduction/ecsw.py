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
Implementation of Energy-conserving sampling and weighting (ECSW)hyper-reduction

Energy-conserving sampling and weighting (ECSW) is a hyper-reduction approach
originally developed specifically for solid mechanics problems, but it has
since been generalized. It is a project-then-approximate hyper-reduction
approach, similar in spirit and implementation to empirical cubature
approaches. The name comes from the energy conservation properties the method
has for solid mechanics applications; note that this property is not provable
for general systems.

Given a set of residual snapshots ECSW computes sampling indices $GID_i$ and
weights $\\xi_i$. Specifically, the residual snapshots must be computed for
reconstructed full-order model snapshots,

$$\\boldsymbol r_i = \\boldsymbol r( \\boldsymbol \\Phi \\boldsymbol \\Phi^T (\\mathbf{u}_i - \\mathbf{u}_0))$$

where $\\boldsymbol r_i$ is the residual $i$th residual snapshot,
$\\mathbf{u}_i$ is the $i$th state snapshot, and $\\boldsymbol \\Phi$ is the
trial basis.

The goal of ECSW is to find a sparse set of weights to approximate the reduced
residual with a subset of local test-basis/residual products

$$\\sum_{e \\in \mathcal{E}} \\xi_e \\boldsymbol \\Psi_e^T \\boldsymbol r_e \\approx \\Psi^T \\boldsymbol r$$

For more details, consult Chapman et al. 2016 DOI: 10.1002/nme.5332.

The ECSW class contains the methods needed to compute sampling indices and
weights given a set of residual snapshot and trial basis data.
'''

import sys
import abc
from typing import Tuple
import numpy as np


class AbstractECSWsolver(abc.ABC):
    '''
    Abstract base class for ECSW solvers

    ECSW solvers should take in a linear system constructed from projected residual vector snapshots and the contributions at each mesh degree of freedom to the projected snapshot. The solvers should return arrays with sample mesh indices and weights. 

    Methods:
    '''
    @abc.abstractmethod
    def __init__(self, solver_param_dict: dict = None):
        '''
        Set solver parameters to non-default values

        Args: 
            (optional) solver_param_dict: dictionary, with some of the following keys:
            max_iters: int, maximum overall iterations
            max_non_neg_iters: int, maximum inner iterations to enforce non-negativity
            max_iters_res_unchanged: int, maximum number of iterations without any change in the residual norm before terminating
            zero_tol: int, tolerance used to check if weights or residual norm changes are near zero
        '''
        pass

    @abc.abstractmethod
    def __call__(self, full_mesh_lhs: np.ndarray, full_mesh_rhs: np.array, tolerance: np.double) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute the sample mesh DoF indices and corresponding weights

        Args:
            full_mesh_lhs: (n_snap*n_rom, n_dof) numpy ndarray, where n_snap is the number of residual snapshots, n_rom is the ROM dimension, and n_dof is the number of mesh degrees of freedom (DoFs) (nodes, volumes, or elements)
            full_mesh_rhs: (n_snap*n_rom,) numpy array
            tolerance: Double, the ECSW tolerance parameter. Lower values of tolerance will result in more mesh DoF samples

        Returns:
            Tuple of numpy ndarrays. 
            First array: (n_dof_sample_mesh,) numpy ndarray of ints, the mesh indices in the sample mesh. 
            Second array: (n_dof_sample_mesh,) numpy ndarray of doubles, the corresponding sample mesh weights. 
        
        '''
        pass

# ecsw non-negative least-squares class
# TODO: scipy loss and enhancement method for testing? Won't work in place of
#       this function because tau can't be specified


class ECSWsolverNNLS(AbstractECSWsolver):
    '''
    Given a linear system with left-hand side full_mesh_lhs and right-hand side full_mesh_rhs compute sample mesh indices and weights for ECSW using the non-negative least squares algorithm from Chapman et al. 2016
    DOI: 10.1002/nme.5332.
    '''

    def __init__(self, solver_param_dict: dict = None):
        # Set default solver parameter values
        self.max_iters = 10000  # maximum overall iterations
        self.max_non_neg_iters = 100  # maximum inner iterations to enforce non-negativity
        self.max_iters_res_unchanged = 10  # maximum number of iterations without any change in residual before terminating
        self.zero_tol = 1e-12  # tolerance used to check if weights or residual norm changes are near zero

        if solver_param_dict is not None:
            if 'max_iters' in solver_param_dict.keys():
                self.max_iters = solver_param_dict['max_iters']
            if 'max_non_neg_iters' in solver_param_dict.keys():
                self.max_non_neg_iters = solver_param_dict['max_non_neg_iters']
            if 'max_iters_res_unchanged' in solver_param_dict.keys():
                self.max_iters_res_unchanged = solver_param_dict['max_iters_res_unchanged']
            if 'zero_tol' in solver_param_dict.keys():
                self.zero_tol = solver_param_dict['zero_tol']

    def __call__(self, full_mesh_lhs: np.ndarray, full_mesh_rhs: np.array, tolerance: np.double):
        '''
        Compute the sample mesh DoF indices and corresponding weights using the non-negative least squares algorithm from Chapman et al. 2016
        DOI: 10.1002/nme.5332.

        min || full_mesh_lhs*full_mesh_weights-full_mesh_rhs ||_2^2 s.t. full_mesh_weights >=0, || full_mesh_lhs*full_mesh_weights-full_mesh_rhs ||_2 < tolerance ||full_mesh_rhs||_2

        Args:
            full_mesh_lhs: (n_snap*n_rom, n_dof) numpy ndarray, where n_snap is the number of residual snapshots, n_rom is the ROM dimension, and n_dof is the number of mesh degrees of freedom (DoFs) (nodes, volumes, or elements)
            full_mesh_rhs: (n_snap*n_rom,) numpy array
            tolerance: Double, the ECSW tolerance parameter. Lower values of tolerance will result in more mesh DoF samples

        Returns:
            Tuple of numpy ndarrays. 
            First array: (n_dof_sample_mesh,) numpy ndarray of ints, the mesh indices in the sample mesh. 
            Second array: (n_dof_sample_mesh,) numpy ndarray of doubles, the corresponding sample mesh weights. 
        
        '''

        n_dof = full_mesh_lhs.shape[1]

        # initialize
        iters = 0
        sample_mesh_indicies = []
        residual = full_mesh_rhs.copy()
        residual_norm = np.linalg.norm(residual)
        target_norm = tolerance*residual_norm

        full_mesh_weights = np.zeros(n_dof)
        full_mesh_candidate_weights = np.zeros(n_dof)

        residual_norm_unchanged = 0

        # add nodes to sample mesh until tolerance is met
        while (residual_norm > target_norm) and (residual_norm_unchanged < self.max_iters_res_unchanged) and (iters < self.max_iters):
            # determine new node to add to sample mesh
            weighted_residual = np.dot(full_mesh_lhs.T, residual)

            # make sure mesh entity hasn't already been selected
            still_searching = True
            while still_searching:
                mesh_index = np.argmax(weighted_residual)
                still_searching = False
                if mesh_index in sample_mesh_indicies:
                    still_searching = True
                    weighted_residual[mesh_index] = np.min(weighted_residual)

            # add new mesh entity index
            sample_mesh_indicies.append(mesh_index)

            print("iteration={}  sample mesh size={}  residual norm={:.8e}  ratio to target={:.8e} ".format(iters, len(sample_mesh_indicies), residual_norm, residual_norm/target_norm))
            iters += 1

            # compute corresponding weights

            for j in range(self.max_non_neg_iters):
                sample_mesh_lhs = full_mesh_lhs[:, sample_mesh_indicies]
                sample_mesh_candidate_weights = np.dot(np.linalg.pinv(sample_mesh_lhs), full_mesh_rhs)
                full_mesh_candidate_weights *= 0
                full_mesh_candidate_weights[sample_mesh_indicies] = sample_mesh_candidate_weights
                if np.all(sample_mesh_candidate_weights > 0):
                    full_mesh_weights = full_mesh_candidate_weights.copy()
                    break
                # line search to enforce non-negativity
                max_step = self.__max_feasible_step(full_mesh_weights[sample_mesh_indicies], sample_mesh_candidate_weights)
                full_mesh_weights_new = full_mesh_weights + max_step * (full_mesh_candidate_weights - full_mesh_weights)

                # remove zero valued indices
                near_zero_inds = np.nonzero(full_mesh_weights_new[sample_mesh_indicies] < self.zero_tol)[0]
                samp_inds_for_removal = [sample_mesh_indicies[i] for i in near_zero_inds]
                for samp_ind in samp_inds_for_removal:
                    sample_mesh_indicies.remove(samp_ind)

                # increment iteration count
                print("iteration={}  sample mesh size={}  residual norm={:.8e}  ratio to target={:.8e} ".format(iters, len(sample_mesh_indicies), residual_norm, residual_norm/target_norm))
                iters += 1
                full_mesh_weights = 1*full_mesh_weights_new


            if j == self.max_non_neg_iters-1:
                sys.exit("Error: NNLS algorithm failed to compute weights")

            # update least-squares residual
            sample_mesh_weights = full_mesh_weights[sample_mesh_indicies]
            residual = full_mesh_rhs - np.dot(sample_mesh_lhs, sample_mesh_weights)
            residul_old_norm = 1*residual_norm
            residual_norm = np.linalg.norm(residual)

            if np.abs(residual_norm - residul_old_norm) < self.zero_tol:
                residual_norm_unchanged += 1
            else:
                residual_norm_unchanged = 0

            if (residual_norm_unchanged >= self.max_iters_res_unchanged):
                print("WARNING: Norm has not changed more than {} in {} steps, exiting NNLS".format(self.zero_tol, self.max_iters_res_unchanged))

        print("NNLS complete! Final stats:")
        print("iteration={}  sample mesh size={}  residual norm={:.8e}  ratio to target={:.8e} ".format(iters, len(sample_mesh_indicies), residual_norm, residual_norm/target_norm))

        return np.array(sample_mesh_indicies,dtype=int), sample_mesh_weights

    def __max_feasible_step(self, weights, candidate_weights):
        '''
        determine maximum update step size such that:
        weights + step_size * (candidate_weights-weights) >=0

        Args: 
            weights: (n,) array
            candidate_weights: (n, array)

        Returns:
            step_size: double
        '''
        inds = np.argwhere(candidate_weights <= 0)
        step_size = 1.0
        for i in inds:
            if (weights[i] == 0.0):
                step_size = 0
            else:
                step_size_i = weights[i] / (weights[i] - candidate_weights[i])
                step_size = min([step_size, step_size_i])
        return step_size


# ESCW helper functions for specific test basis types

def _construct_linear_system(residual_snapshots: np.ndarray,
                             test_basis: np.ndarray,
                             n_var: int,
                             variable_ordering: str):
    '''
    Construct the linear system required for ECSW with a fixed test basis, such as POD-Galerkin projection. 

    Args:
        residual_snapshots: (n_dof*n_var, n_snap) numpy ndarray, where n_dof is the number of mesh degrees of freedom (DoFs) (nodes, volumes, or elements), n_var is the number of residual variables, and n_snap is the number of snapshots
        test_basis: (n_dof*n_var, n_mode) numpy ndarray, where n_mode is the number of modes in the basis.
        n_var: int, the number of residual variables (e.g. for fluid flow, residual variable could be mass, x-momentum, y-momentum, z-momentum, and energy)
        variable_ordering (str): The variable ordering, either 'C' or 'F'. C: variables are the fastest index F: mesh DoFs are the fastest index

    Returns:
        full_mesh_lhs: (n_snap*n_mode, n_dof) numpy ndarray, the left-hand side of the linear system required by the ECSW solver
        full_mesh_rhs: (n_snap*n_rom,) numpy array, the right-hand side of the linear system required by the ECSW solver
    '''

    (n_row, n_snap) = residual_snapshots.shape
    n_dof = int(n_row / n_var)
    n_mode = test_basis.shape[1]
    # construct ECSW system
    full_mesh_lhs = np.zeros((n_snap*n_mode, n_dof))

    # left-hand side
    for i in range(n_dof):
        # should be projection of all variables for a given mesh DoF
        if variable_ordering == 'C':
            Phi_block = test_basis[(i*n_var):((i+1)*n_var),:]  # n_var x n_mode
            resSnaps_block = residual_snapshots[(i*n_var):((i+1)*n_var), :]  # n_var x n_snap
        elif variable_ordering == 'F':
            Phi_block = test_basis[i::n_dof, :]  # n_var x n_mode
            resSnaps_block = residual_snapshots[i::n_dof, :]  # n_var x n_snap

        full_mesh_lhs_block = np.dot(Phi_block.T, resSnaps_block)  # Nmodes x Nsnaps matrix
        full_mesh_lhs[:, i] = np.ravel(full_mesh_lhs_block, order='F')

    # right-hand-side
    full_mesh_rhs = np.sum(full_mesh_lhs, axis=1)

    return full_mesh_lhs, full_mesh_rhs


def ecsw_fixed_test_basis(ecsw_solver: AbstractECSWsolver,
                          residual_snapshots: np.ndarray,
                          test_basis: np.ndarray,
                          n_var: int,
                          variable_ordering: str,
                          tolerance: np.double):
    '''
    ECSW implementation for a fixed test basis, such as POD-Galerkin projection

    Args:
        ecsw_solver: AbstractECSWsolver object corresponding to a child class with concrete implementations such as ECSWsolverNNLS.
        residual_snapshots: (n_dof*n_var, n_snap) numpy ndarray, where n_dof is the number of mesh degrees of freedom (DoFs) (nodes, volumes, or elements), n_var is the number of residual variables, and n_snap is the number of snapshots
        test_basis: (n_dof*n_var, n_mode) numpy ndarray, where n_mode is the number of modes in the basis.
        n_var: int, the number of residual variables (e.g. for fluid flow, residual variable could be mass, x-momentum, y-momentum, z-momentum, and energy)
        variable_ordering (str): The variable ordering, either 'C' or 'F'. C: variables are the fastest index F: mesh DoFs are the fastest index
        tolerance: Double, the ECSW tolerance parameter. Lower values of tolerance will result in more mesh DoF samples

    Returns:
        Tuple of numpy ndarrays. 
        First array: (n_dof_sample_mesh,) numpy ndarray of ints, the mesh indices in the sample mesh. 
        Second array: (n_dof_sample_mesh,) numpy ndarray of doubles, the corresponding sample mesh weights. 
    '''

    # TODO need to incorporate residual scales here too, perhaps using scaler.py
    full_mesh_lhs, full_mesh_rhs = _construct_linear_system(residual_snapshots,
                                                            test_basis,
                                                            n_var,
                                                            variable_ordering)

    return ecsw_solver(full_mesh_lhs, full_mesh_rhs, tolerance)

def ecsw_varying_test_basis(ecsw_solver: AbstractECSWsolver,
                            full_mesh_lhs: np.ndarray,
                            full_mesh_rhs: np.ndarray,
                            tolerance: np.double):
    '''
    ECSW implementation for a varying test basis, such as Least-Squares Petrov-Galerkin projection

    Args:
        ecsw_solver: AbstractECSWsolver object corresponding to a child class with concrete implementations such as ECSWsolverNNLS.
        full_mesh_lhs: (n_snap*n_rom, n_dof) numpy ndarray, where n_snap is the number of residual snapshots, n_rom is the ROM dimension, and n_dof is the number of mesh degrees of freedom (DoFs) (nodes, volumes, or elements)
        full_mesh_rhs: (n_snap*n_rom,) numpy array
        tolerance: Double, the ECSW tolerance parameter. Lower values of tolerance will result in more mesh DoF samples

    Returns:
        Tuple of numpy ndarrays. 
        First array: (n_dof_sample_mesh,) numpy ndarray of ints, the mesh indices in the sample mesh. 
        Second array: (n_dof_sample_mesh,) numpy ndarray of doubles, the corresponding sample mesh weights. 
    '''

    return ecsw_solver(full_mesh_lhs, full_mesh_rhs, tolerance)

