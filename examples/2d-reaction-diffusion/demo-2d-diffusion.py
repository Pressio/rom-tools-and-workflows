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

def my_source(x, y, time):
    return 4.*np.sin(4*math.pi*x*y)*np.sin(math.pi*x*(y - 0.2)) 
    # return np.sin(math.pi*x) + y * x + time
    

class FOMSamplingCouplerBase():
    def __init__(self, parameter_name, num_parameters, base_directory):
        self._parameter_name = parameter_name
        self._dimension = num_parameters
        self._base_directory = base_directory

    def set_parameters_in_input(self):
        return 0

    def run_model(self, file_name, parameter_values):
        # Open and Read Input File
        f = open(file_name)
        input = json.load(f)
        f.close()

        # Parse Input File
        mesh_path = input['mesh_path']
        dt = input['dt']
        n_steps = input['n_steps']

        # Load mesh
        mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

        # Define Scheme
        # A. set scheme
        scheme  = pda.ViscousFluxReconstruction.FirstOrder

        # B. constructor for problem using default values
        prob_ID  = pda.DiffusionReaction2d.ProblemA
        problem = pda.create_problem(mesh_obj, prob_ID, scheme)

        # C. setting custom coefficients and custom source function
        D = parameter_values[0]
        K = parameter_values[1]
        problem = pda.create_diffusion_reaction_2d_problem_A(mesh_obj, scheme, my_source, D, K)

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
        F = problem.createRightHandSide()
        def residual(x):
            problem.rightHandSide(x,0.,F)
            return F
        
        # Solve FOM
        yn = scipy.optimize.newton_krylov(residual, problem.initialCondition(), verbose=True)

        # Save solution
        np.savez('results.npz', y=yn, parameters=[D, K])

        return 0
    
    def get_parameter_space(self):
        param_space = ParameterSpace(parameter_name=self._parameter_name, num_parameters=self._dimension, bounds=[[0.005, 0.015], [0.005, 0.015]])
        return param_space

    def create_cases(self, starting_sample_no, parameter_samples):
        for i in range(starting_sample_no, np.shape(parameter_samples)[0]):
            if not os.path.exists('i' + str(i)):
                os.mkdir('i' + str(i))
            os.system('cp input.json i' + str(i) + '/.')
    
    def get_sol_directory(self, idx):
        return 'i' + str(idx)

    def get_input_filename(self):
        return 'input.json'
    
    def get_base_directory(self):
        return self._base_directory

class ROMSamplingCouplerBase():
    def set_parameters_in_input(self):
        return 0

    def run_model(self, file_name, parameter_values, basis):
        # Open and Read Input File
        f = open(file_name)
        input = json.load(f)
        f.close()

        # Parse Input File
        mesh_path = input['mesh_path']
        dt = input['dt']
        n_steps = input['n_steps']

        # Load mesh
        mesh_obj = pda.load_cellcentered_uniform_mesh(mesh_path)

        # Define Scheme
        # A. set scheme
        scheme  = pda.ViscousFluxReconstruction.FirstOrder

        # B. constructor for problem using default values
        prob_ID  = pda.DiffusionReaction2d.ProblemA
        problem = pda.create_problem(mesh_obj, prob_ID, scheme)

        # C. setting custom coefficients and custom source function
        D = parameter_values[0]
        K = parameter_values[1]
        problem = pda.create_diffusion_reaction_2d_problem_A(mesh_obj, scheme, my_source, D, K)

        # Run ROM
        # A. get initial condition
        yn = problem.initialCondition()
        qn = np.matmul(basis.transpose(), yn)

        # B. solve ROM
        rom = ROM(basis=basis, problem=problem) # NOTE: Defined ROM class so that I can use pda.advanceRK4
        # pda.advanceRK4(rom, qn, dt, n_steps)
        # qn = myRK4(rom, yn, dt, n_steps)
        # qn = myRK4(rom, qn, dt, n_steps)

        # D. Define residual function
        F = rom.createRightHandSide()
        def residual(x):
            _, v = rom.rightHandSide(np.matmul(basis, x),0.,np.matmul(basis, F))
            return v
        
        # Solve FOM
        qn = scipy.optimize.newton_krylov(residual, np.matmul(basis.transpose(), problem.initialCondition()), verbose=True, f_tol=1e-8)

        return qn
    
    def get_parameter_space(self, parameter_name, num_parameters):
        param_space = ParameterSpace(parameter_name=parameter_name, num_parameters=num_parameters)
        
        return param_space

class ROM():
    def __init__(self, basis, problem):
        self._basis = basis
        self._problem = problem
    
    def initializeRightHandSide(self):
        self._problem.createRightHandSide()
    
    def createRightHandSide(self):
        # return self._problem.createRightHandSide()
        self._problem.createRightHandSide()
        return np.zeros(self._basis.shape[1])

    def rightHandSide(self, state, time, v):
        self._problem.rightHandSide(state, time, v)
        state = np.matmul(self._basis.transpose(), state)
        v = np.matmul(self._basis.transpose(), v)

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

class GreedyCouplerBase():
    def compute_qoi(self):
        # Read response from file
        y = 0.
        qoi = np.max(y)

        return qoi
    
    def compute_error_indicator(self):
        5

    def get_parameter_space(self):
        5

def myRK4(appObj, state, dt, Nsteps, \
               startTime = 0.0, \
               observer = None, \
               showProgress=False):
    sys.settrace

    v = appObj.createRightHandSide()
    tmpState = state.copy()
    half = 0.5
    two  = 2.
    oneOverSix = 1./6.

    time = startTime
    for step in range(1, Nsteps+1):
        if showProgress:
            if step % 50 == 0: print("step = ", step, "/", Nsteps)

        state, v = appObj.rightHandSide(state, time, v)
        if observer!= None:
            observer(step-1, state, v)
        k1 = dt * v

        tmpState = state+half*k1
        state, v = appObj.rightHandSide(tmpState, time+half*dt, v)
        k2 = dt * v

        tmpState = state+half*k2
        state, v = appObj.rightHandSide(tmpState, time+half*dt, v)
        k3 = dt * v

        tmpState = state+k3
        state, v = appObj.rightHandSide(tmpState, time+dt, v)
        k4 = dt * v

        state[:] = state + (k1+two*k2+two*k3+k4)*oneOverSix
        time += dt
        
        # print(step, time, np.max(state), np.min(state))
    return state
    
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

def plot_results(figure_path, mesh_obj, y_fom, y_rom, x_label, y_label):
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
    plt.savefig(figure_path + '/contour_fom_vs_rom.jpeg')
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

# Generate and plot the mesh
mesh_obj = generate_mesh(pressio_file_path=pressio_file_path, mesh_path=mesh_path, figure_path=figure_path, n_x=n_x, n_y=n_y)

# Run FOM
# A. Set up FOM Coupler
fom_sampler = FOMSamplingCouplerBase(parameter_name=['K','D'], num_parameters=2, base_directory=os.getcwd())

# B. Run FOM for N number of samples
rt.workflows.sampling.run_sampling(sampling_coupler=fom_sampler, testing_sample_size=n_snapshots, random_seed=1)

# C. Read in snapshots (NOTE: snapshots should be a tensor)
n_vars = 1 # number of PDE variables
n = n_x * n_y # number of spatial DOFs
n_snapshots = input['n_snapshots'] # number of samples/snapshots
snapshots = np.zeros((n_vars, n, n_snapshots))
parameters = np.zeros((n_snapshots, 2))
for i in range(0,n_snapshots):
    results = np.load('i' + str(i) + '/results.npz')
    snapshots[:,:,i] = results['y']
    parameters[i,:] = results['parameters']

# D. Plot results for one snapshot
plot_single_result(figure_path, mesh_obj, snapshots[0,:,0], x_label=f'$K={parameters[0,0]:.3f}$', y_label=f'$D={parameters[0,1]:.3f}$', suffix='')

# E. Run FOM for one sample
st = time.time()
y = fom_sampler.run_model(file_name=file_name, parameter_values=parameters[0,:])
print('Time to run FOM: ', time.time()-st)

# Run ROM
# A. Define truncater, shifter, splitter, and orthogonalizer
truncater = rt.vector_space.utils.truncater.EnergyTruncater(threshold=0.99)
# truncater = rt.vector_space.utils.truncater.NoOpTruncater()
shifter = rt.vector_space.utils.shifter.NoOpShifter()
splitter = rt.vector_space.utils.splitter.NoOpSplitter()
orthogonalizer = rt.vector_space.utils.orthogonalizer.EuclideanL2Orthogonalizer()

# B. Calculate trial space
pod_space = rt.vector_space.VectorSpaceFromPOD(snapshots=snapshots, truncater=truncater, shifter=shifter, splitter=splitter, orthogonalizer=orthogonalizer)
basis = pod_space.get_basis()[0]
plot_single_result(figure_path, mesh_obj, basis[:,0], x_label=f'$K={parameters[0,0]:.3f}$', y_label=f'$D={parameters[0,1]:.3f}$', suffix='_basis0')

# C. Run ROM for a reconstructive case
rom_sampler = ROMSamplingCouplerBase()
st = time.time()
q = rom_sampler.run_model(file_name=file_name, parameter_values=parameters[0,:], basis=basis)
print('Time to run ROM: ', time.time()-st)

# D. Reconstruct state vector
y_rom_reconstruct = np.matmul(basis, q)

# E. Plot results
plot_single_result(figure_path, mesh_obj, y_rom_reconstruct, x_label=f'$K={parameters[0,0]:.3f}$', y_label=f'$D={parameters[0,1]:.3f}$', suffix='_rom')

# Plot both results together
plot_results(figure_path, mesh_obj, snapshots[0,:,0], y_rom_reconstruct, x_label=f'$K={parameters[0,0]:.3f}$', y_label=f'$D={parameters[0,1]:.3f}$')

