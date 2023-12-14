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

    G is a [n_snaps*n_rom by n_dofs] numpy ndarray, where n_snaps is the
    number of residual snapshots, n_rom is the ROM dimension, and n_dofs is
    the number of mesh degrees of freedom (DoFs) (nodes, volumes, or elements)

    b is a [n_snaps*n_rom] length numpy array

    tau is the ECSW tolerance parameter. Lower values of tau will result in
    more mesh DoF samples
    '''

    def __init__(self, solver_param_dict: dict = None):
        # Set default solver parameter values
        self.k_max = 10000  # maximum overall iterations
        self.j_max = 100  # maximum inner iterations to enforce non-negativity
        self.max_iters_res_unchanged = 10  # maximum number of iterations without any change in residual before terminating
        self.zero_tol = 1e-12  # tolerance used to check if weights or residual norm changes are near zero

        if solver_param_dict is not None:
            if 'max_iters' in solver_param_dict.keys():
                self.k_max = solver_param_dict['max_iters']
            if 'max_non_neg_iters' in solver_param_dict.keys():
                self.j_max = solver_param_dict['max_non_neg_iters']
            if 'max_iters_res_unchanged' in solver_param_dict.keys():
                self.max_iters_res_unchanged = solver_param_dict['max_iters_res_unchanged']
            if 'zero_tol' in solver_param_dict.keys():
                self.zero_tol = solver_param_dict['zero_tol']

    def __call__(self, G: np.ndarray, b: np.array, tau: np.double):
        # min || G*xi-b||_2^2 s.t. xi >=0, || G*xi-b||_2 < tau ||b||_2
        n_dofs = G.shape[1]

        # initialize
        k = 0
        I_k = []
        r_k = b.copy()
        r_k_norm = np.linalg.norm(r_k)
        tau_b_norm = tau*r_k_norm

        xi_k = np.zeros(n_dofs)
        zeta_k = np.zeros(n_dofs)

        r_k_norm_unchanged = 0

        # add nodes to sample mesh until tolerance is met
        while (r_k_norm > tau_b_norm) and (r_k_norm_unchanged < self.max_iters_res_unchanged) and (k < self.k_max):
            # determine new node to add to sample mesh
            mu_k = np.dot(G.T, r_k)

            # make sure mesh entity hasn't already been selected
            still_searching = True
            while still_searching:
                i_k = np.argmax(mu_k)
                still_searching = False
                if i_k in I_k:
                    still_searching = True
                    mu_k[i_k] = np.min(mu_k)

            # add new mesh entity index
            I_k.append(i_k)

            print("k={}  n_samps={}  ||r_k||={}  tau*||b||={} ".format(k, len(I_k), r_k_norm, tau_b_norm))
            k += 1

            # compute corresponding weights

            for j in range(self.j_max):
                G_I_k = G[:, I_k]
                zeta_I_k = np.dot(np.linalg.pinv(G_I_k), b)
                zeta_k *= 0
                zeta_k[I_k] = zeta_I_k
                if np.all(zeta_I_k > 0):
                    xi_k = zeta_k.copy()
                    break
                # line search to enforce non-negativity
                max_step = self.__max_feasible_step(xi_k[I_k], zeta_I_k)
                xi_kp1 = xi_k + max_step * (zeta_k - xi_k)

                # remove zero valued indices
                inds = np.nonzero(xi_kp1[I_k] < self.zero_tol)[0]
                samp_inds_for_removal = [I_k[i] for i in inds]
                for samp_ind in samp_inds_for_removal:
                    I_k.remove(samp_ind)

                # increment iteration count
                k += 1
                xi_k = 1*xi_kp1
                print("k={}  n_samps={}  ||r_k||={}  tau*||b||={} ".format(k, len(I_k), r_k_norm, tau_b_norm))

            if j == self.j_max-1:
                sys.exit("Error: NNLS algorithm failed to compute weights")

            # update least-squares residual r_k
            xi_I_k = xi_k[I_k]
            r_k = b - np.dot(G_I_k, xi_I_k)
            r_km1_norm = 1*r_k_norm
            r_k_norm = np.linalg.norm(r_k)

            if np.abs(r_k_norm - r_km1_norm) < self.zero_tol:
                r_k_norm_unchanged += 1
            else:
                r_k_norm_unchanged = 0

            if (r_k_norm_unchanged >= self.max_iters_res_unchanged):
                print("WARNING: Norm has not changed more than {} in {} steps, exiting NNLS".format(self.zero_tol, self.max_iters_res_unchanged))

        print("NNLS complete! Final stats:")
        print("k={}  n_samps={}  ||r_k||={}  tau*||b||={} ".format(k, len(I_k), r_k_norm, tau_b_norm))

        return I_k, xi_k

    def __max_feasible_step(self, xi, zeta):
        '''
        determine maximum step size such that:
        xi + step_size * (zeta-xi) >=0
        '''
        inds = np.argwhere(zeta <= 0)
        step_size = 1.0
        for i in inds:
            if (xi[i] == 0.0):
                step_size = 0
            else:
                step_size_i = xi[i] / (xi[i] - zeta[i])
                step_size = min([step_size, step_size_i])
        return step_size


# ESCW helper functions for specific test basis types

def _construct_linear_system(residual_snapshots: np.ndarray,
                             test_basis: np.ndarray,
                             n_vars: int,
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

    residual_snapshots is a n_dofs*n_vars by n_snaps array, where n_dofs is
    the number of mesh degrees of freedom, n_vars is the number of residual
    variables (e.g. for fluid flow, residual variable could be mass,
    x-momentum, y-momentum, z-momentum, and energy), and n_snaps is the number
    of snapshots

    test_basis is a n_dofs*n_vars by n_modes array, where n_modes is the
    number of modes in the basis.

    variable_ordering is a character specifying the ordering of variables and
    DoFs in residual snapshots and test_basis.
    C: variables are the fastest index
    F: mesh DoFs are the fastest index
    '''

    (n_rows, n_snaps) = residual_snapshots.shape
    n_dofs = int(n_rows / n_vars)
    n_modes = test_basis.shape[1]
    # construct ECSW system
    full_mesh_lhs = np.zeros((n_snaps*n_modes, n_dofs))

    # left-hand side
    for i in range(n_dofs):
        # should be projection of all variables for a given mesh DoF
        if variable_ordering == 'C':
            Phi_block = test_basis[(i*n_vars):((i+1)*n_vars),:]  # n_vars x n_modes
            resSnaps_block = residual_snapshots[(i*n_vars):((i+1)*n_vars), :]  # n_vars x n_snaps
        elif variable_ordering == 'F':
            Phi_block = test_basis[i::n_dofs, :]  # n_vars x n_modes
            resSnaps_block = residual_snapshots[i::n_dofs, :]  # n_vars x n_snaps

        full_mesh_lhs_block = np.dot(Phi_block.T, resSnaps_block)  # Nmodes x Nsnaps matrix
        full_mesh_lhs[:, i] = np.ravel(full_mesh_lhs_block, order='F')

    # right-hand-side
    full_mesh_rhs = np.sum(full_mesh_lhs, axis=1)

    return full_mesh_lhs, full_mesh_rhs


def ecsw_fixed_test_basis(ecsw_solver: AbstractECSWsolver,
                          residual_snapshots: np.ndarray,
                          test_basis: np.ndarray,
                          n_vars: int,
                          variable_ordering: str,
                          tau: np.double):
    '''
    ECSW implementation for a fixed test basis, such as POD-Galerkin projection

    Read in residual snapshots, test basis, and the number of variables, then
    construct the ECSW linear system for the entire mesh, including the
    left-hand-side matrix and right-hand-side vector. Pieces of the matrix and
    vector will be used to construct the corresponding linear systems for
    candidate sample meshes.

    Return mesh DoF weights and indices

    residual_snapshots is a n_dofs*n_vars by n_snaps array, where n_dofs is
    the number of mesh degrees of freedom, n_vars is the number of residual
    variables (e.g. for fluid flow, residual variable could be mass,
    x-momentum, y-momentum, z-momentum, and energy), and n_snaps is the number
    of snapshots

    test_basis is a n_dofs*n_vars by n_modes array, where n_modes is the
    number of modes in the basis.

    variable_ordering is a character specifying the ordering of variables and
    DoFs in residual snapshots and test_basis.
    C: variables are the fastest index
    F: mesh DoFs are the fastest index

    tau is the tolerance for the ecsw_solver. It is a small (<<1) positive
    number
    '''

    # TODO need to incorporate residual scales here too, perhaps using scaler.py
    full_mesh_lhs, full_mesh_rhs = _construct_linear_system(residual_snapshots,
                                                            test_basis,
                                                            n_vars,
                                                            variable_ordering)

    return ecsw_solver(full_mesh_lhs, full_mesh_rhs, tau)

def ecsw_varying_test_basis(ecsw_solver: AbstractECSWsolver,
                            full_mesh_lhs: np.ndarray,
                            full_mesh_rhs: np.ndarray,
                            tau: np.double):
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

    tau is the tolerance for the ecsw_solver. It is a small (<<1) positive
    number
    '''

    return ecsw_solver(full_mesh_lhs, full_mesh_rhs, tau)

