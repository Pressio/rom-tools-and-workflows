# PRESSIO Demo Apps, 2D Single-Species Reaction Diffusion Problem
#
# This demo will show usages of scripts from the ROM tools workflow including:
# (1) Scripts to make a basis with greedy training
# (2) Scripts to run the full-order model (FOM)
# (3) Scripts to sample the reduced-order model (ROM)
# (4) Scripts to post-process results

# Pressio Modules
import pressiodemoapps as pda
import romtools as rt

# Python Modules
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import sys

from autograd import jacobian

def my_source(x, y, time):
    return 4.*np.sin(4*math.pi*x*y)*np.sin(math.pi*x*(y - 0.2)) 
    # return np.sin(math.pi*x) + y * x + time

class FOM_Model(rt.workflows.models.Model):
    def __init__(self):
        return None

    def populate_run_directory(self, run_directory, parameter_sample):
        os.system('cp input.json ' + run_directory + '/.')
        return 0

    def run_model(self, run_directory, parameter_sample):
        # Swap to run directory
        cdir = os.getcwd()
        os.chdir(run_directory)

        # Open and Read Input File
        f = open(file_name)
        input = json.load(f)
        f.close()

        # Parse Input File
        mesh_path = input['mesh_path']
        # dt = input['dt']
        # n_steps = input['n_steps']

        # Load mesh
        mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

        # Define Scheme
        # A. set scheme
        scheme  = pda.ViscousFluxReconstruction.FirstOrder

        # B. constructor for problem using default values
        prob_ID  = pda.DiffusionReaction2d.ProblemA
        self._problem = pda.create_problem(mesh_obj, prob_ID, scheme)

        # C. setting custom coefficients and custom source function
        self._problem = pda.create_diffusion_reaction_2d_problem_A(mesh_obj, scheme, my_source, parameter_sample['D'], parameter_sample['K'])

        # (For Unsteady ROM)
        # Run FOM
        # # A. get initial condition
        # yn = problem.initialCondition()

        # # B. solve
        # max_y = []
        # for i in range(0, n_steps):
        #     pda.advanceRK4(problem, yn, dt, 1) #n_steps)
        #     max_y.append(np.max(yn))

        # C. Plot convergence
        # plt.figure(figsize=(10,7))
        # plt.plot(max_y)
        # plt.savefig('y_convergence.jpeg')
        # plt.close()
        
        # (For Steady ROM)
        # D. Define residual function
        F = self._problem.createRightHandSide()
        def residual(x):
            self._problem.rightHandSide(x,0.,F)
            return F
        
        # Solve FOM
        yn = scipy.optimize.newton_krylov(residual, self._problem.initialCondition(), verbose=False)

        # Save solution
        np.savez('results.npz', y=yn, parameters=[parameter_sample['D'], parameter_sample['K']])

        # Swap back to base directory
        os.chdir(cdir)

        return 0 # NOTE: This function must return 0

class ROM_Model(rt.workflows.models.QoiModelWithErrorEstimate):
    def __init__(self, hyperreduction=False):
        # Open and Read Input File
        f = open(file_name)
        input = json.load(f)
        f.close()

        # Parse Input File
        mesh_path = input['mesh_path']

        # Load mesh
        self._mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

        # Load basis
        self._basis = np.load('basis.npz')['basis']

        # Hyperreduction
        self._hyperreduction = hyperreduction
        if self._hyperreduction:
            self._sample_indices = rt.hyper_reduction.deim_get_indices(self._basis)
            self._test_basis = rt.hyper_reduction.deim_get_test_basis(self._basis, self._basis, self._sample_indices)
            self._approx_mat = rt.hyper_reduction.deim_get_approximation_matrix(self._basis, self._sample_indices)
            np.savez('hyperreduction.npz', sample_indices=self._sample_indices, test_basis=self._test_basis, approx_mat=self._approx_mat)
        else:
            self._sample_indices = range(0,np.shape(self._basis)[0])
            self._test_basis = self._basis
            self._approx_mat = np.eye(np.shape(self._basis)[0])
        return None

    def populate_run_directory(self, run_directory, parameter_sample):
        # NOTE: This is needed when using myRK4
        # os.system('cp input.json ' + run_directory + '/.')
        return 0

    def run_model(self, run_directory, parameter_sample):
        # Swap to run directory
        cdir = os.getcwd()
        os.chdir(run_directory)

        # NOTE: This is needed when using myRK4
        # Open and Read Input File
        # f = open(file_name)
        # input = json.load(f)
        # f.close()

        # NOTE: This is needed when using myRK4
        # Parse Input File
        # dt = input['dt']
        # n_steps = input['n_steps']

        # Define Scheme
        # A. set scheme
        scheme  = pda.ViscousFluxReconstruction.FirstOrder

        # B. constructor for problem using default values
        prob_ID  = pda.DiffusionReaction2d.ProblemA
        problem = pda.create_problem(self._mesh_obj, prob_ID, scheme)

        # C. setting custom coefficients and custom source function
        problem = pda.create_diffusion_reaction_2d_problem_A(self._mesh_obj, scheme, my_source, parameter_sample['D'], parameter_sample['K'])

        # Run ROM
        # A. get initial condition
        yn = problem.initialCondition()
        qn = np.matmul(self._test_basis.transpose(), yn[self._sample_indices])

        # B. solve ROM
        rom = ROM(basis=self._basis, problem=problem, hyperreduction=self._hyperreduction, sample_indices=self._sample_indices, test_basis=self._test_basis, approx_mat=self._approx_mat)
        # pda.advanceRK4(rom, qn, dt, n_steps)
        # qn = myRK4(rom, yn, dt, n_steps)
        # qn = myRK4(rom, qn, dt, n_steps)

        # D. Define residual function
        F = rom.createRightHandSide()
        def residual(x):
            _, v = rom.rightHandSide(np.matmul(self._test_basis, x),0.,np.matmul(self._test_basis, F))
            return v
    
        # Test residual
        test = residual(np.matmul(self._test_basis.transpose(), problem.initialCondition()[self._sample_indices]))
        print(test)
        print(np.matmul(self._test_basis.transpose(), problem.initialCondition()[self._sample_indices]))
        print(problem.initialCondition())
        print(np.max(np.matmul(self._test_basis, np.matmul(self._test_basis.transpose(), problem.initialCondition()[self._sample_indices]))), np.min(np.matmul(self._test_basis, np.matmul(self._test_basis.transpose(), problem.initialCondition()[self._sample_indices]))))
        print(np.max(np.matmul(self._test_basis, F)), np.min(np.matmul(self._test_basis, F)))
        sys.exit()

        # Solve FOM
        qn = scipy.optimize.newton_krylov(residual, np.matmul(self._test_basis.transpose(), problem.initialCondition()[self._sample_indices]), verbose=False, f_tol=1e-8)

        # Reconstruct solution
        yn = np.matmul(self._test_basis, qn)

        # Compute inverse of diagonal of Jacobian
        J = problem.createApplyJacobianResult(np.eye(np.shape(yn)[0]))
        problem.applyJacobian(yn, np.eye(np.shape(yn)[0]), 0., J)
        invJ = np.zeros(np.shape(J))
        count = 0
        for x in np.diag(J):
            invJ[count,count] = 1./x
            count += 1

        # Save results
        np.savez('results.npz', y=yn, parameters=[parameter_sample['D'], parameter_sample['K']], res=np.matmul(basis,residual(qn)), invJ=invJ)

        # Swap back to base directory
        os.chdir(cdir)

        return 0 # NOTE: This function must return 0

    def compute_qoi(self, run_directory, parameter_sample):
        # Load from npz file and return y
        return np.load(run_directory + '/results.npz')['y']

    def compute_error_estimate(self, run_directory, parameter_sample):
        # Run model
        y = self.run_model(run_directory, parameter_sample)

        # Read in results
        dat = np.load(run_directory + '/results.npz')
        invJ = dat['invJ']
        res = dat['res']

        # Calculate error estimate
        return np.linalg.norm(np.matmul(invJ,res))

class ROM():
    def __init__(self, basis, problem, hyperreduction, sample_indices, test_basis, approx_mat):
        self._basis = basis
        self._problem = problem
        self._hyperreduction = hyperreduction
        self._sample_indices = sample_indices
        self._test_basis = test_basis
        self._approx_mat = approx_mat
    
    def initializeRightHandSide(self):
        self._problem.createRightHandSide()
    
    def createRightHandSide(self):
        self._problem.createRightHandSide()
        return np.zeros(self._test_basis.shape[1])

    def rightHandSide(self, state, time, v):
        if self._hyperreduction == True:
            # NOTE: v is overwritten on call to rightHandSide
            v = np.matmul(self._approx_mat, v)
            self._problem.rightHandSide(state, time, v)
        else:
            self._problem.rightHandSide(state, time, v)
        # if self._hyperreduction == True:
        #     state = np.matmul(self._test_basis.transpose(), state[self._sample_indices])
        #     v = np.matmul(self._test_basis.transpose(), v[self._sample_indices])
        # else:
        #     state = np.matmul(self._basis.transpose(), state)
        #     v = np.matmul(self._basis.transpose(), v)
        state = np.matmul(self._test_basis.transpose(), state)
        v = np.matmul(self._test_basis.transpose(), v)

        return state, v

class ParameterSpace():
    def __init__(self, parameter_name, num_parameters, bounds):
        self._parameter_name = parameter_name
        self._dimension = num_parameters
        self._bounds = np.array(bounds)

    def get_names(self):
        return self._parameter_name
    
    def get_dimensionality(self):
        return self._dimension
    
    def generate_samples(self, samples):
        # samples is inputted as a uniform distribution. Need to scale to bounds.
        # NOTE: Look at Box-Muller for future.
        scale =  self._bounds[:,1::] - self._bounds[:,0:1]
        samples = samples*scale.transpose() + self._bounds[:,0:1].transpose()

        return np.array(samples)
    
def generate_mesh(pressio_file_path, mesh_path, figure_path, n_x, n_y):
    # generate mesh
    print('python3 ' + pressio_file_path + '/meshing_scripts/create_full_mesh_for.py --problem diffreac2d -n ' 
                + str(n_x) + ' ' + str(n_y) + ' --outdir ' + mesh_path)
    os.system('python3 ' + pressio_file_path + '/meshing_scripts/create_full_mesh_for.py --problem diffreac2d -n ' 
                + str(n_x) + ' ' + str(n_y) + ' --outdir ' + mesh_path)

    # load mesh
    mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

    # plot mesh
    x = mesh_obj.viewX()
    y = mesh_obj.viewY()
    unique_x = list(set(x))
    unique_y = list(set(y))
    plt.figure(figsize=(10,7))
    for i in range(0,len(unique_x)):
        plt.vlines(unique_x[i], np.max(unique_y), np.min(unique_y), colors = 'k', linewidth = 0.5)
    for i in range(0,len(unique_y)):
        plt.hlines(unique_y[i], np.min(unique_x), np.max(unique_x), colors = 'k', linewidth = 0.5)
    plt.savefig(figure_path + '/mesh.jpeg')
    plt.close()

    return mesh_obj

def plot_single_result(figure_path, mesh_obj, yn, x_label, y_label, suffix):
    x = mesh_obj.viewX()
    y = mesh_obj.viewY()
    unique_x = list(set(x))
    unique_y = list(set(y))
    plt.figure(figsize=(10,7))
    for i in range(0,len(unique_x)):
        plt.vlines(unique_x[i], np.max(unique_y), np.min(unique_y), colors = 'k', linewidth = 0.5, alpha=0.2)
    for i in range(0,len(unique_y)):
        plt.hlines(unique_y[i], np.min(unique_x), np.max(unique_x), colors = 'k', linewidth = 0.5, alpha=0.2)
    plt.tricontourf(x, y, yn, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(figure_path + '/contour_yn_with_mesh' + suffix + '.jpeg')
    plt.close()

    plt.figure(figsize=(10,7))
    plt.tricontourf(x, y, yn, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(figure_path + '/contour_yn_without_mesh' + suffix + '.jpeg')
    plt.close()

def plot_results(figure_path, mesh_obj, y_fom, y_rom, x_label, y_label, suffix):
    x = mesh_obj.viewX()
    y = mesh_obj.viewY()
    unique_x = list(set(x))
    unique_y = list(set(y))
    vmin_ = np.min(y_fom)
    vmax_ = np.max(y_fom)
    
    fig, ax = plt.subplots(2,1,figsize=(9,12))
    plt.sca(ax[0])
    for i in range(0,len(unique_x)):
        plt.vlines(unique_x[i], np.max(unique_y), np.min(unique_y), colors = 'k', linewidth = 0.5, alpha=0.2)
    for i in range(0,len(unique_y)):
        plt.hlines(unique_y[i], np.min(unique_x), np.max(unique_x), colors = 'k', linewidth = 0.5, alpha=0.2)
    im = plt.tricontourf(x, y, y_fom, cmap='coolwarm', vmin=vmin_, vmax=vmax_)
    plt.sca(ax[1])
    for i in range(0,len(unique_x)):
        plt.vlines(unique_x[i], np.max(unique_y), np.min(unique_y), colors = 'k', linewidth = 0.5, alpha=0.2)
    for i in range(0,len(unique_y)):
        plt.hlines(unique_y[i], np.min(unique_x), np.max(unique_x), colors = 'k', linewidth = 0.5, alpha=0.2)
    im = plt.tricontourf(x, y, y_rom, cmap='coolwarm', vmin=vmin_, vmax=vmax_)
    fig.colorbar(im, ax=ax.ravel().tolist())

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(figure_path + '/contour_fom_vs_rom' + suffix + '.jpeg')
    plt.close()

# Open and read input file
file_name = 'input.json'
f = open(file_name)
input = json.load(f)
f.close()

# Parse input file
pressio_file_path = input['pressio_file_path']
mesh_path = input['mesh_path']
n_x = input['n_x']
n_y = input['n_y']
n_snapshots = input['n_snapshots']
figure_path = input['figure_path']
n_test = input['n_test']

# Generate and plot the mesh
mesh_obj = generate_mesh(pressio_file_path=pressio_file_path, mesh_path=mesh_path, figure_path=figure_path, n_x=n_x, n_y=n_y)

# Define FOM and ROM Model
fom_model = FOM_Model()
rom_model = ROM_Model(hyperreduction=False)

# Define parameter space
param_space = ParameterSpace(parameter_name=['K', 'D'], num_parameters=2, bounds=[[0.005, 0.015], [0.005, 0.015]])

# Example: Monte Carlo Sampling of FOM
# A. Run FOM at train/test points using montecarlo sampling
rt.workflows.sampling.run_sampling(model=fom_model, parameter_space=param_space, run_directory_prefix='random/fom_', number_of_samples=n_snapshots, random_seed=1)
rt.workflows.sampling.run_sampling(model=fom_model, parameter_space=param_space, run_directory_prefix='test/fom_', number_of_samples=n_test, random_seed=1)

# B. Read in train/test snapshots (NOTE: snapshots should be a tensor)
n_vars = 1 # number of PDE variables
n = n_x * n_y # number of spatial DOFs
snapshots_train = np.zeros((n_vars, n, n_snapshots))
parameters_train = np.zeros((n_snapshots, 2))
for i in range(0,n_snapshots):
    results = np.load('random/fom_' + str(i) + '/results.npz')
    snapshots_train[:,:,i] = results['y']
    parameters_train[i,:] = results['parameters']
snapshots_test = np.zeros((n_vars, n, n_test))
parameters_test = np.zeros((n_test, 2))
for i in range(0,n_test):
    results = np.load('test/fom_' + str(i) + '/results.npz')
    snapshots_test[:,:,i] = results['y']
    parameters_test[i,:] = results['parameters']

# C. Plot results for one snapshot
plot_single_result(figure_path, mesh_obj, snapshots_train[0,:,0], x_label=f'$K={parameters_train[0,0]:.3f}$', y_label=f'$D={parameters_train[0,1]:.3f}$', suffix='_random')

# ROM
# A. Calculate trial space from snapshots
truncater = rt.vector_space.utils.truncater.NoOpTruncater()
orthogonalizer = rt.vector_space.utils.orthogonalizer.EuclideanL2Orthogonalizer()
pod_space = rt.vector_space.VectorSpaceFromPOD(snapshots=snapshots_train, truncater=truncater, orthogonalizer=orthogonalizer)
basis = pod_space.get_basis()[0]
np.savez('basis.npz',basis=basis) # NOTE: This is read in by run_model

# B. Plot zeroth mode
plot_single_result(figure_path, mesh_obj, basis[:,0], x_label=f'$K={parameters_train[0,0]:.3f}$', y_label=f'$D={parameters_train[0,1]:.3f}$', suffix='_basis0_random')

# C. Run ROM at test points
# NOTE: As long as the same seed is chosen, the ROM will run at the same points as the FOM.
rt.workflows.sampling.run_sampling(model=rom_model, parameter_space=param_space, run_directory_prefix='test/rom_', number_of_samples=n_test, random_seed=1)

# D. Read in ROM results
rom_snapshots_test = np.zeros((n_vars, n, n_test))
rom_parameters_test = np.zeros((n_test, 2))
for i in range(0,n_test):
    results = np.load('test/rom_' + str(i) + '/results.npz')
    rom_snapshots_test[:,:,i] = results['y']
    rom_parameters_test[i,:] = results['parameters']

# E. Plot ROM result at test point
plot_single_result(figure_path, mesh_obj, rom_snapshots_test[0,:,0], x_label=f'$K={rom_parameters_test[0,0]:.3f}$', y_label=f'$D={rom_parameters_test[0,1]:.3f}$', suffix='_rom_random')

# F. Plot ROM/FOM results at test point
plot_results(figure_path, mesh_obj, snapshots_test[0,:,0], rom_snapshots_test[0,:,0], x_label=f'$K={parameters_test[0,0]:.3f}$', y_label=f'$D={parameters_test[0,1]:.3f}$', suffix='_random')

# G. Calculate error
print('L2 Norm of Error between ROM and FOM at test points using MC sampling: ', np.linalg.norm(snapshots_test - rom_snapshots_test), '\n')

# Example: DEIM Hyperreduction with Monte Carlo Sampling
# A. Define ROM model with hyperreduction
hyper_rom_model = ROM_Model(hyperreduction=True)

# B. Run ROM at test points
rt.workflows.sampling.run_sampling(model=hyper_rom_model, parameter_space=param_space, run_directory_prefix='test/hyper_rom_', number_of_samples=n_test, random_seed=1)

# C. Read in ROM results
approx_mat = np.load('hyperreduction.npz')['approx_mat']
hyper_rom_snapshots_test = np.zeros((n_vars, n, n_test))
hyper_rom_parameters_test = np.zeros((n_test, 2))
for i in range(0,n_test):
    results = np.load('test/hyper_rom_' + str(i) + '/results.npz')
    hyper_rom_snapshots_test[:,:,i] = np.matmul(approx_mat, results['y'])
    hyper_rom_parameters_test[i,:] = results['parameters']

# D. Plot ROM result at test point
plot_single_result(figure_path, mesh_obj, hyper_rom_snapshots_test[0,:,0], x_label=f'$K={hyper_rom_parameters_test[0,0]:.3f}$', y_label=f'$D={hyper_rom_parameters_test[0,1]:.3f}$', suffix='_rom_random_hyperreduction')

# E. Plot ROM/FOM results at test point
plot_results(figure_path, mesh_obj, snapshots_test[0,:,0], hyper_rom_snapshots_test[0,:,0], x_label=f'$K={parameters_test[0,0]:.3f}$', y_label=f'$D={parameters_test[0,1]:.3f}$', suffix='_random_hyperreduction')

# F. Calculate error
print('L2 Norm of Error between ROM and FOM at test points using MC sampling with DEIM hyperreduction: ', np.linalg.norm(snapshots_test - hyper_rom_snapshots_test))
print('L2 Norm of Error between hyperreduced ROM and full ROM at test points using MC sampling: ', np.linalg.norm(rom_snapshots_test - hyper_rom_snapshots_test), '\n')

# G. Compare runtime
fom_time = []
rom_time = []
hyper_rom_time = []
for i in range(0, n_test):
    st = time.time(); fom_model.run_model(run_directory=os.getcwd(), parameter_sample={'D': parameters_test[i,0], 'K': parameters_test[i,1]})
    fom_time.append(time.time()-st)
    st = time.time(); rom_model.run_model(run_directory=os.getcwd(), parameter_sample={'D': parameters_test[i,0], 'K': parameters_test[i,1]})
    rom_time.append(time.time()-st)
    st = time.time(); hyper_rom_model.run_model(run_directory=os.getcwd(), parameter_sample={'D': parameters_test[i,0], 'K': parameters_test[i,1]})
    hyper_rom_time.append(time.time()-st)
print('Runtime of FOM: ', np.mean(fom_time))
print('Runtime of full ROM: ', np.mean(rom_time))
print('Runtime of DEIM hyperreduced ROM: ', np.mean(hyper_rom_time))
sys.exit()

# Example: Greedy sampling for ROM
# A. Define greedy sampler
greedy_sampler = GreedyCouplerBase(rom_coupler=rom_sampler_greedy, fom_coupler=fom_sampler_greedy, base_directory=os.getcwd())

# B. Run greedy
rt.workflows.greedy.run_greedy(greedy_coupler=greedy_sampler, tolerance=1e-4, testing_sample_size=3)

# C. Run greedy-trained ROM at test points
st = time.time()
y_rom = np.zeros(np.shape(snapshots_test))
for i in range(0, n_test):
    y_rom[:,:,i] = rom_sampler_greedy.run_model(file_name=file_name, parameter_values=parameters_test[i,:])
print('Time to run ROM: ', (time.time()-st) / n_test)

# D. Plot results
plot_single_result(figure_path, mesh_obj, y_rom[0,:,0], x_label=f'$K={parameters_test[0,0]:.3f}$', y_label=f'$D={parameters_test[0,1]:.3f}$', suffix='_greedy_rom')
plot_results(figure_path, mesh_obj, snapshots_test[0,:,0], y_rom[0,:,0], x_label=f'$K={parameters_test[0,0]:.3f}$', y_label=f'$D={parameters_test[0,1]:.3f}$', suffix='_greedy')
plot_single_result(figure_path, mesh_obj, snapshots_test[0,:,0] - y_rom[0,:,0], x_label=f'$K={parameters_test[0,0]:.3f}$', y_label=f'$D={parameters_test[0,1]:.3f}$', suffix='_greedy_romfom_error')

# E. Calculate error
err = np.linalg.norm(snapshots_test - y_rom)
print('L2 Norm of Error between ROM and FOM at test points using greedy sampling: ', err)











# class FOMSamplingCouplerBase(rt.workflows.sampling.SamplingCouplerBase):
#     def __init__(self, template_directory, template_input_file, base_directory, sol_directory_basename, parameter_name, num_parameters):
#         self._template_directory = template_directory
#         self._template_input_file = template_input_file
#         self._base_directory = base_directory
#         self._sol_directory_basename = sol_directory_basename
#         self._parameter_name = parameter_name
#         self._dimension = num_parameters

#     def set_parameters_in_input(self):
#         return 0

#     def run_model(self, file_name, parameter_values):
#         # Open and Read Input File
#         f = open(file_name)
#         input = json.load(f)
#         f.close()

#         # Parse Input File
#         mesh_path = input['mesh_path']
#         dt = input['dt']
#         n_steps = input['n_steps']

#         # Load mesh
#         mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

#         # Define Scheme
#         # A. set scheme
#         scheme  = pda.ViscousFluxReconstruction.FirstOrder

#         # B. constructor for problem using default values
#         prob_ID  = pda.DiffusionReaction2d.ProblemA
#         self._problem = pda.create_problem(mesh_obj, prob_ID, scheme)

#         # C. setting custom coefficients and custom source function
#         D = parameter_values[0]
#         K = parameter_values[1]
#         self._problem = pda.create_diffusion_reaction_2d_problem_A(mesh_obj, scheme, my_source, D, K)

#         # (For Unsteady ROM)
#         # Run FOM
#         # # A. get initial condition
#         # yn = problem.initialCondition()

#         # # B. solve
#         # max_y = []
#         # for i in range(0, n_steps):
#         #     pda.advanceRK4(problem, yn, dt, 1) #n_steps)
#         #     max_y.append(np.max(yn))

#         # C. Plot convergence
#         # plt.figure(figsize=(10,7))
#         # plt.plot(max_y)
#         # plt.savefig('y_convergence.jpeg')
#         # plt.close()
        
#         # (For Steady ROM)
#         # D. Define residual function
#         F = self._problem.createRightHandSide()
#         def residual(x):
#             self._problem.rightHandSide(x,0.,F)
#             return F
        
#         # Solve FOM
#         yn = scipy.optimize.newton_krylov(residual, self._problem.initialCondition(), verbose=False)

#         # Save inverse of diagonal of Jacobian
#         J = self._problem.createApplyJacobianResult(np.eye(np.shape(yn)[0]))
#         self._problem.applyJacobian(yn, np.eye(np.shape(yn)[0]), 0., J)
#         invJ = np.zeros(np.shape(J))
#         count = 0
#         for x in np.diag(J):
#             invJ[count,count] = 1./x
#             count += 1

#         # Save solution
#         np.savez('results.npz', y=yn, parameters=[D, K], invJ = invJ)

#         return 0
    
#     def get_parameter_space(self):
#         param_space = ParameterSpace(parameter_name=self._parameter_name, num_parameters=self._dimension, bounds=[[0.005, 0.015], [0.005, 0.015]])
#         return param_space

#     def create_cases(self, starting_sample_no, parameter_samples):
#         if np.shape(parameter_samples)[0] == 1:
#             if not os.path.exists(self._sol_directory_basename + str(starting_sample_no)):
#                 os.mkdir(self._sol_directory_basename + str(starting_sample_no))
#             os.system('cp ' + self._template_input_file + ' ' + self._sol_directory_basename + str(starting_sample_no) + '/.')
#         else:
#             for i in range(starting_sample_no, np.shape(parameter_samples)[0]):
#                 if not os.path.exists(self._sol_directory_basename + str(i)):
#                     os.mkdir(self._sol_directory_basename + str(i))
#                 os.system('cp ' + self._template_input_file + ' ' + self._sol_directory_basename + str(i) + '/.')
    
#     def get_sol_directory(self, idx):
#         return self._sol_directory_basename + str(idx)

#     def get_input_filename(self):
#         return self._template_input_file
    
#     def get_base_directory(self):
#         return self._base_directory

# class ROMSamplingCouplerBase(rt.workflows.sampling.SamplingCouplerBase):
#     def __init__(self, template_directory, template_input_file, base_directory, sol_directory_basename, parameter_name, num_parameters):
#         self._template_directory = template_directory
#         self._template_input_file = template_input_file
#         self._base_directory = base_directory
#         self._sol_directory_basename = sol_directory_basename

#     def set_parameters_in_input(self):
#         return 0

#     def run_model(self, file_name, parameter_values):
#         # Open and Read Input File
#         f = open(file_name)
#         input = json.load(f)
#         f.close()

#         # Parse Input File
#         mesh_path = input['mesh_path']
#         dt = input['dt']
#         n_steps = input['n_steps']

#         # Load mesh
#         mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

#         # Load basis
#         self._basis = np.load(self.get_base_directory() + '/basis.npz')['basis']

#         # Define Scheme
#         # A. set scheme
#         scheme  = pda.ViscousFluxReconstruction.FirstOrder

#         # B. constructor for problem using default values
#         prob_ID  = pda.DiffusionReaction2d.ProblemA
#         problem = pda.create_problem(mesh_obj, prob_ID, scheme)

#         # C. setting custom coefficients and custom source function
#         D = parameter_values[0]
#         K = parameter_values[1]
#         problem = pda.create_diffusion_reaction_2d_problem_A(mesh_obj, scheme, my_source, D, K)

#         # Run ROM
#         # A. get initial condition
#         yn = problem.initialCondition()
#         qn = np.matmul(self._basis.transpose(), yn)

#         # B. solve ROM
#         rom = ROM(basis=self._basis, problem=problem) # NOTE: Defined ROM class so that I can use pda.advanceRK4
#         # pda.advanceRK4(rom, qn, dt, n_steps)
#         # qn = myRK4(rom, yn, dt, n_steps)
#         # qn = myRK4(rom, qn, dt, n_steps)

#         # D. Define residual function
#         F = rom.createRightHandSide()
#         def residual(x):
#             _, v = rom.rightHandSide(np.matmul(self._basis, x),0.,np.matmul(self._basis, F))
#             return v
        
#         # Solve FOM
#         qn = scipy.optimize.newton_krylov(residual, np.matmul(self._basis.transpose(), problem.initialCondition()), verbose=False, f_tol=1e-8)

#         # Save reconstructed solution
#         yn = np.matmul(self._basis, qn)
#         np.savez('results.npz', y=yn, parameters=[D, K], res=np.matmul(self._basis,residual(qn)))

#         return yn
    
#     def create_cases(self, starting_sample_no, parameter_samples):
#         if np.shape(parameter_samples)[0] == 1:
#             if not os.path.exists(self._sol_directory_basename + str(starting_sample_no)):
#                 os.mkdir(self._sol_directory_basename + str(starting_sample_no))
#             os.system('cp ' + self._template_input_file + ' ' + self._sol_directory_basename + str(starting_sample_no) + '/.')
#         else:
#             for i in range(starting_sample_no, np.shape(parameter_samples)[0]):
#                 if not os.path.exists(self._sol_directory_basename + str(i)):
#                     os.mkdir(self._sol_directory_basename + str(i))
#                 os.system('cp ' + self._template_input_file + ' ' + self._sol_directory_basename + str(i) + '/.')

#     def get_sol_directory(self, idx):
#         return self._sol_directory_basename + str(idx)

#     def get_parameter_space(self, parameter_name, num_parameters):
#         param_space = ParameterSpace(parameter_name=parameter_name, num_parameters=num_parameters)
#         return param_space
    
#     def get_input_filename(self):
#         return self._template_input_file
    
#     def get_base_directory(self):
#         return self._base_directory

# class GreedyCouplerBase(rt.workflows.greedy.GreedyCouplerBase):
#     def compute_qoi(self):
#         # Read response from file
#         output = np.load('results.npz')
#         y = output['y']
#         qoi = np.max(y)
#         return qoi
    
#     def compute_error_indicator(self):
#         rom = np.load('results.npz')
#         ind = os.getcwd()[-1]
#         fom = np.load('../fom_0/results.npz')
#         invJ = fom['invJ']
#         res = rom['res']

#         return np.linalg.norm(np.matmul(invJ, res))

#     def get_parameter_space(self):
#         param_space = self.fom_coupler.get_parameter_space()
#         return param_space
    
#     def create_trial_space(self, training_sample_indices):
#         # Save training sample indices
#         self._training_sample_indices = training_sample_indices

#         # Read in input file
#         f = open(self.fom_coupler.get_input_filename())
#         input = json.load(f)
#         f.close()

#         # Read in FOM snapshots
#         n_vars = 1 # number of PDE variables
#         n = input['n_x'] * input['n_y'] # number of spatial DOFs
#         snapshots = np.zeros((n_vars, n, np.shape(training_sample_indices)[0]))
#         count = 0
#         for sample in training_sample_indices:
#             results = np.load(sample + '/results.npz')
#             snapshots[:,:,count] = results['y']
#             count += 1

#         # Define options
#         truncater = rt.vector_space.utils.truncater.NoOpTruncater()
#         orthogonalizer = rt.vector_space.utils.orthogonalizer.EuclideanL2Orthogonalizer()

#         # Calculate trial space
#         pod_space = rt.vector_space.VectorSpaceFromPOD(snapshots=snapshots, truncater=truncater, orthogonalizer=orthogonalizer)
#         basis = pod_space.get_basis()[0]

#         # Save basis
#         np.savez('basis.npz', basis=basis)