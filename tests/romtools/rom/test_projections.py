import romtools
import pytest
import numpy as np
import scipy.optimize

@pytest.mark.mpi_skip
def test_optimal_l2_projection_single_vector():
    np.random.seed(1)
    data = np.random.normal(size=(3,10,5))
    data_to_project = np.random.normal(size=(3,10))

    rom_dim = 4
    my_truncater = romtools.vector_space.utils.BasisSizeTruncater(rom_dim)
    trial_space = romtools.VectorSpaceFromPOD(snapshots=data,
                                          truncater=my_truncater,
                                          shifter = None,
                                          orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                          scaler = romtools.vector_space.utils.NoOpScaler())

    reduced_state = romtools.rom.optimal_l2_projection(data_to_project, trial_space)

    phi = trial_space.get_basis()
    def residual_for_ls_solver(xhat):
      x = np.einsum('ijk,k...->ij...',phi,xhat)
      error  = x.flatten() - data_to_project.flatten()
      return error

    scipy_reduced_state = scipy.optimize.least_squares(residual_for_ls_solver,np.zeros(rom_dim),jac='cs').x 
    assert np.allclose(scipy_reduced_state,reduced_state)


    ## Now test w/ an offset
    np.random.seed(1)
    data = np.random.normal(size=(3,10,5))
    data_to_project = np.random.normal(size=(3,10))

    my_shifter = romtools.utils.create_average_shifter(data)
    rom_dim = 4
    my_truncater = romtools.vector_space.utils.BasisSizeTruncater(rom_dim)
    trial_space = romtools.VectorSpaceFromPOD(snapshots=data,
                                          truncater=my_truncater,
                                          shifter = my_shifter,
                                          orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                          scaler = romtools.vector_space.utils.NoOpScaler())

    reduced_state = romtools.rom.optimal_l2_projection(data_to_project, trial_space)

    phi = trial_space.get_basis()
    shift_vec = trial_space.get_shift_vector()
    def residual_for_ls_solver(xhat):
      x = np.einsum('ijk,k...->ij...',phi,xhat) + shift_vec
      error  = x.flatten() - data_to_project.flatten()
      return error

    scipy_reduced_state = scipy.optimize.least_squares(residual_for_ls_solver,np.zeros(rom_dim),jac='cs').x 
    assert np.allclose(scipy_reduced_state,reduced_state)

    ## Now test w/ a weighted inner product 
    np.random.seed(1)
    data = np.random.normal(size=(3,10,5))
    data_to_project = np.random.normal(size=(3,10))
    M = np.random.normal(size=((30,30)))
    M = M @ M.transpose()
    Mchol = np.linalg.cholesky(M)
    my_shifter = romtools.utils.create_average_shifter(data)
    rom_dim = 4
    my_truncater = romtools.vector_space.utils.BasisSizeTruncater(rom_dim)
    trial_space = romtools.VectorSpaceFromPOD(snapshots=data,
                                          truncater=my_truncater,
                                          shifter = my_shifter,
                                          orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                          scaler = romtools.vector_space.utils.NoOpScaler())

    reduced_state = romtools.rom.optimal_l2_projection(data_to_project, trial_space, weighting_matrix=M)

    phi = trial_space.get_basis()
    shift_vec = trial_space.get_shift_vector()
    def residual_for_ls_solver(xhat):
      x = np.einsum('ijk,k...->ij...',phi,xhat) + shift_vec
      error  = x.flatten() - data_to_project.flatten()
      return Mchol.transpose() @ error

    scipy_reduced_state = scipy.optimize.least_squares(residual_for_ls_solver,np.zeros(rom_dim),jac='cs').x 
    assert np.allclose(scipy_reduced_state,reduced_state)


@pytest.mark.mpi_skip
def test_optimal_l2_projection_multiple_vectors():
    np.random.seed(1)
    data = np.random.normal(size=(3,10,5))
    n_data = 3
    data_to_project = np.random.normal(size=(3,10,n_data))

    rom_dim = 4
    my_truncater = romtools.vector_space.utils.BasisSizeTruncater(rom_dim)
    trial_space = romtools.VectorSpaceFromPOD(snapshots=data,
                                          truncater=my_truncater,
                                          shifter = None,
                                          orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                          scaler = romtools.vector_space.utils.NoOpScaler())

    reduced_state = romtools.rom.optimal_l2_projection(data_to_project, trial_space)

    phi = trial_space.get_basis()
    def residual_for_ls_solver(xhat):
      xhat = np.reshape(xhat,(rom_dim,n_data))
      x = np.einsum('ijk,k...->ij...',phi,xhat)
      error  = x.flatten() - data_to_project.flatten()
      return error


    scipy_reduced_state = scipy.optimize.least_squares(residual_for_ls_solver,np.zeros(rom_dim*n_data),jac='cs').x 
    assert np.allclose(scipy_reduced_state,reduced_state.flatten())


    ## Now test w/ an offset
    np.random.seed(1)
    data = np.random.normal(size=(3,10,5))
    data_to_project = np.random.normal(size=(3,10,n_data))

    my_shifter = romtools.utils.create_average_shifter(data)
    rom_dim = 4
    my_truncater = romtools.vector_space.utils.BasisSizeTruncater(rom_dim)
    trial_space = romtools.VectorSpaceFromPOD(snapshots=data,
                                          truncater=my_truncater,
                                          shifter = my_shifter,
                                          orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                          scaler = romtools.vector_space.utils.NoOpScaler())

    reduced_state = romtools.rom.optimal_l2_projection(data_to_project, trial_space)

    phi = trial_space.get_basis()
    shift_vec = trial_space.get_shift_vector()

    def residual_for_ls_solver(xhat):
      xhat = np.reshape(xhat,(rom_dim,n_data))
      x = np.einsum('ijk,k...->ij...',phi,xhat) + shift_vec[...,None]
      error  = x.flatten() - data_to_project.flatten()
      return error


    scipy_reduced_state = scipy.optimize.least_squares(residual_for_ls_solver,np.zeros(rom_dim*n_data),jac='cs').x 
    assert np.allclose(scipy_reduced_state,reduced_state.flatten())

    ## Now test w/ a weighted inner product 
    np.random.seed(1)
    data = np.random.normal(size=(3,10,5))
    data_to_project = np.random.normal(size=(3,10,n_data))
    M = np.random.normal(size=((30,30)))
    M = M @ M.transpose()
    Mchol = np.linalg.cholesky(M)
    my_shifter = romtools.utils.create_average_shifter(data)
    rom_dim = 4
    my_truncater = romtools.vector_space.utils.BasisSizeTruncater(rom_dim)
    trial_space = romtools.VectorSpaceFromPOD(snapshots=data,
                                          truncater=my_truncater,
                                          shifter = my_shifter,
                                          orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                          scaler = romtools.vector_space.utils.NoOpScaler())

    reduced_state = romtools.rom.optimal_l2_projection(data_to_project, trial_space, weighting_matrix=M)

    phi = trial_space.get_basis()
    shift_vec = trial_space.get_shift_vector()

    def residual_for_ls_solver(xhat):
      xhat = np.reshape(xhat,(rom_dim,n_data))
      x = np.einsum('ijk,k...->ij...',phi,xhat) + shift_vec[...,None]
      error  = x - data_to_project
      error = np.reshape(error,((error.shape[0]*error.shape[1]),error.shape[2]))
      return ( Mchol.transpose() @ error ).flatten()

    scipy_reduced_state = scipy.optimize.least_squares(residual_for_ls_solver,np.zeros(rom_dim*n_data),jac='cs').x 
    assert np.allclose(scipy_reduced_state,reduced_state.flatten())



@pytest.mark.mpi_skip
def test_optimal_l2_projection_full_return():
    np.random.seed(1)
    data = np.random.normal(size=(3,10,5))
    data_to_project = data[...,0]

    rom_dim = 5
    my_truncater = romtools.vector_space.utils.BasisSizeTruncater(rom_dim)
    trial_space = romtools.VectorSpaceFromPOD(snapshots=data,
                                          truncater=my_truncater,
                                          shifter = None,
                                          orthogonalizer=romtools.vector_space.utils.EuclideanL2Orthogonalizer(),
                                          scaler = romtools.vector_space.utils.NoOpScaler())

    reduced_state,projected_data = romtools.rom.optimal_l2_projection(data_to_project, trial_space,return_full_state=True)
   
    assert np.allclose(projected_data,data_to_project)

if __name__ == '__main__':
    test_optimal_l2_projection_single_vector() 
    test_optimal_l2_projection_multiple_vectors() 
    test_optimal_l2_projection_full_return() 
