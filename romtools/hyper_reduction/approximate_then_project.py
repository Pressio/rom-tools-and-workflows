import numpy as np
import romtools
from romtools.vector_space.utils import *
from romtools.hyper_reduction.deim import *
def StaticPetrovGalerkinApproximateThenProjectHyperReducer(function_snapshots : np.ndarray, test_basis: np.ndarray, target_sample_mesh_fraction: float , method='DEIM'):
        '''
        For a function f approximated as Phi_f  pinv(P \Phi_f) P f, the purpose is to return 
        Psi^T Phi_f  pinv(P \Phi_f), where Psi^T is the "test" basis 
        along with the sample mesh indices
    
        Inputs: (n_vars, nx, n_snaps) np.ndarray, residual snapshots
                (n_vars, nx, K) np.ndarray, test basis
                     float, target sample mesh fraction
        Outputs: 
             (n_vars, ns , K) np.ndarray, hyper_reduced_test_basis, where ns is the number of sample mesh points 
             ns, np.ndarray: sample mesh points
        '''

        # first, create a basis for the right hand size
        n_vars = function_snapshots.shape[0]
        n_mesh_points = function_snapshots.shape[1]
        n_snaps = function_snapshots.shape[2]
        target_mesh_points = target_sample_mesh_fraction*n_mesh_points
        if target_mesh_points > n_snaps:
            print("Warning, you requested more target mesh points than available snapshots")
            print("Number of requested mesh points = " , target_mesh_points)
            print("Number of snapshots = " , n_snaps)

        target_mesh_points = min(target_mesh_points,n_snaps)
        assert(target_mesh_points > test_basis.shape[-1])

        # Truncate the basis to the number of target mesh points  
        my_truncater = BasisSizeTruncater(target_mesh_points)
        function_vector_space = romtools.VectorSpaceFromPOD(function_snapshots)
        function_basis = function_vector_space.get_basis()

        ## Now reshape into a 2D matrix to do basic DEIM algorithm
        function_basis = np.reshape(function_basis,(function_basis.shape[0]*function_basis.shape[1],function_basis.shape[-1]))

        ## get indices of selected DOFs
        deim_indices = deim_get_indices(function_basis)

        ## Translate DOF indices to mesh indices  
        sample_mesh_indices = deim_indices % n_mesh_points
        sample_mesh_indices = np.int_(np.unique(sample_mesh_indices))

        ## Now send back to include all variables at a mesh index
        dof_indices = sample_mesh_indices*1
        for i in range(1,n_vars):
            dof_indices = np.append(dof_indices , sample_mesh_indices + i*n_mesh_points)

        ## Now get DEIM test basis
        test_basis = np.reshape(test_basis,(test_basis.shape[0]*test_basis.shape[1],test_basis.shape[-1]))
        deim_test_basis = deim_get_test_basis(test_basis, function_basis, dof_indices)
        deim_test_basis = np.reshape(deim_test_basis,(n_vars,sample_mesh_indices.size,deim_test_basis.shape[-1]))
        print('DEIM SUMMARY')
        print('=============================')
        print('Number of sample mesh points: ' + str(sample_mesh_indices.size))
        print('Function basis dimension: ' + str(function_basis.shape[-1]))
        print('DEIM test basis shape: ' + str(deim_test_basis.shape))
        print('=============================')
        return deim_test_basis,sample_mesh_indices
    

