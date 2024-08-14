import pressiodemoapps as pda
import scipy.optimize
import numpy as np
import yaml
import pathlib

path_to_file = str(pathlib.Path(__file__).parent.resolve())

meshObj = pda.load_cellcentered_uniform_mesh(path_to_file + "/mesh_32x32")
x = meshObj.viewX()
y = meshObj.viewY()
scheme = pda.ViscousFluxReconstruction.FirstOrder

with open("parameters.yaml") as f:
    parameters_yaml = yaml.safe_load(f)

D = parameters_yaml["D"]
k = parameters_yaml["k"]

problem = pda.create_diffusion_reaction_2d_problem_A(meshObj, scheme, D, k)

u = problem.initialCondition()
f = problem.createRightHandSide()


def residual(u):
    problem.rightHandSide(u, 0.0, f)
    return f


u = scipy.optimize.newton_krylov(residual, u, verbose=4)
np.savez("solution", u=u)
