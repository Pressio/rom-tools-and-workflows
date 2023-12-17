'''
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
    '''
    @abc.abstractmethod
    def __init__(self, solver_param_dict: dict = None):
        pass

    @abc.abstractmethod
    def __call__(self, G: np.ndarray, b: np.array, tau: np.double) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Overload to compute mesh DoF indices and weights
        '''
        pass

# ecsw non-negative least-squares class
# TODO: scipy loss and enhancement method for testing? Won't work in place of
#       this function because tau can't be specified


class ECSWsolverNNLS(AbstractECSWsolver):
    '''
    Takes linears system G,b, returns mesh indices and weights for ECSW
    computed via non-negative least squares algorithm from Chapman et al. 2016
    DOI: 10.1002/nme.5332.

    full_mesh_lhs is a [n_snap*n_rom by n_dof] numpy ndarray, where n_snap is the
    number of residual snapshots, n_rom is the ROM dimension, and n_dof is
    the number of mesh degrees of freedom (DoFs) (nodes, volumes, or elements)

    full_mesh_rhs is a [n_snap*n_rom] length numpy array

    tolerance is the ECSW tolerance parameter. Lower values of tolerance will result in
    more mesh DoF samples
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
        # Solve min || full_mesh_lhs*full_mesh_weights-full_mesh_rhs ||_2^2 s.t. full_mesh_weights >=0, || full_mesh_lhs*full_mesh_weights-full_mesh_rhs ||_2 < tolerance ||full_mesh_rhs||_2

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

        return sample_mesh_indicies, full_mesh_weights

    def __max_feasible_step(self, weights, candidate_weights):
        '''
        determine maximum step size such that:
        weights + step_size * (candidate_weights-weights) >=0
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
    Make linear system for ECSW with a fixed test basis, such as POD-Galerkin
    projection

    Read in residual snapshots, test basis, and the number of variables, then
    construct the ECSW linear ystem for the entire mesh, including the
    left-hand-side matrix and right-hand-side vector. Pieces of the matrix and
    vector will be used to construct the corresponding linear systems for
    candidate sample meshes.

    Return the matrix and right-hand-side

    residual_snapshots is a n_dof*n_var by n_snap array, where n_dof is
    the number of mesh degrees of freedom, n_var is the number of residual
    variables (e.g. for fluid flow, residual variable could be mass,
    x-momentum, y-momentum, z-momentum, and energy), and n_snap is the number
    of snapshots

    test_basis is a n_dof*n_var by n_mode array, where n_mode is the
    number of modes in the basis.

    variable_ordering is a character specifying the ordering of variables and
    DoFs in residual snapshots and test_basis.
    C: variables are the fastest index
    F: mesh DoFs are the fastest index
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

    Read in residual snapshots, test basis, and the number of variables, then
    construct the ECSW linear system for the entire mesh, including the
    left-hand-side matrix and right-hand-side vector. Pieces of the matrix and
    vector will be used to construct the corresponding linear systems for
    candidate sample meshes.

    Return mesh DoF weights and indices

    residual_snapshots is a n_dof*n_var by n_snap array, where n_dof is
    the number of mesh degrees of freedom, n_var is the number of residual
    variables (e.g. for fluid flow, residual variable could be mass,
    x-momentum, y-momentum, z-momentum, and energy), and n_snap is the number
    of snapshots

    test_basis is a n_dof*n_var by n_mode array, where n_mode is the
    number of modes in the basis.

    variable_ordering is a character specifying the ordering of variables and
    DoFs in residual snapshots and test_basis.
    C: variables are the fastest index
    F: mesh DoFs are the fastest index

    tolerance is the tolerance for the ecsw_solver. It is a small (<<1) positive
    number
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
    ECSW implementation for a fixed test basis, such as POD-Galerkin projection

    Read in the ECSW linear system for the entire mesh, including the
    left-hand-side matrix and right-hand-side vector. Pieces of the matrix and
    vector will be used to construct the corresponding linear systems for
    candidate sample meshes.

    See ecsw_fixed_test_basis for an example of how to construct full_mesh_lhs
    and full_mesh_rhs for a fixed test basis.

    full_mesh_lhs is a Nsnaps*Nrom by Ndofs numpy ndarray, where Nsnaps is the
    number of residual snapshots, Nrom is the ROM dimension, and Ndofs is the
    number of mesh degrees of freedom (DoFs) (nodes, volumes, or elements)

    full_mesh_rhs is a Nsnaps*Nrom length numpy array

    tolerance is the tolerance for the ecsw_solver. It is a small (<<1) positive
    number
    '''

    return ecsw_solver(full_mesh_lhs, full_mesh_rhs, tolerance)

