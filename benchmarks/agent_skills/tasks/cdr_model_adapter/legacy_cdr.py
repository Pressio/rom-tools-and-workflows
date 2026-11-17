"""Legacy CDR interface; intentionally does not implement ROM Tools Model."""
import numpy as np
from examples.models.steady_cdr import AdvectionDiffusionSystem, solveFom

class LegacyCDR:
    def __init__(self, nx=8, ny=8):
        self.system = AdvectionDiffusionSystem(nx, ny)
    def solve(self, diffusivity, reaction, advection=(1., .5)):
        return solveFom(self.system, np.asarray(advection), diffusivity, reaction)
    def boundary_flux(self, state):
        return float(self.system.C @ state)
