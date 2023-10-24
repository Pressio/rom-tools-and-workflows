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
# Scope and Motivation

The ROM tools and workflows Python library comprises a set of algorithms for
constructing and exploiting ROMs.

# Design and Philosophy

The library is designed internally in terms of *abstract base classes* that encapsulate
all the information needed to run a given algorithm.
The philosophy is that, for any given application, the user "simply" needs to create
a class that meets the required API of the abstract base class.
Once this class is complete, the user gains access to all of our existing algorithms.

# What does this library contain?

The Python library, called `romtools`, contains abstract interfaces and functions required for, e.g.,

- Constructing parameter spaces
- Constructing a snapshot data class
- Constructing trial spaces
- Constructing and exploiting ROMs via outer loop workflows

## Algorithms

- Trial space computation:
  - Reduced-basis methods
  - Proper orthogonal decomposition
    - Algorithms are all compatible with basis scaling, basis splitting for multistate problems, and orthogonalization in different inner products
- Workflows for ROM construction and ROM exploitation:
  - ROM construction via reduced-basis greedy (RB-Greedy)
  - ROM/FOM exploitation via sampling
  - ROM/FOM exploitation via Dakota-driven sampling


## Representative abstract base classes

- `AbstractSnapshotData`
  - This class defines the minimum API requirements for a "snapshot_data" class that will be used in the construction of a trial space.

- `AbstractTrialSpace`
  - This class defines the minimum API requirements for a trial space

  - Constructing a trial space relies on utilities like truncaters, orthogonalizers, etc. Abstract classes, and concrete implementations, exist for:

      - orthogonalizers
      - scalers
      - shifters
      - splitters
      - truncaters

- `AbstractParameterSpace`
  - This class defines the minimum API of a parameter space. These parameter spaces are used in workflows for running/building ROMs

- Abstract couplers for greedy sampling, sampling, and coupling to Dakota.

# Demos/HOWTOs

In this section, we present some demos/examples showing how to use `romtools` in practice.

## 1. Snapshot data

The first demo is a simple example of how you could use collect snapshot data for a basic problem.\
For the sake of exposition, suppose that we are solving the [1D heat equation](https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_03_Diffusion_Explicit.html)

$$\\partial_t T(x,t) = \\alpha \\frac{d^2 T} {dx^2}(x,t) + \\sigma (x,t)$$

where $x$ is space, $t$ is time, $T$ is the temperature, $\\alpha$ is the diffusivity,
and $\\sigma$ is the source term.
Instead of the numerical solution, for now let's work with the analytical solution:

$$T(x,t)=e^{-4\\pi^2\\alpha t}\\sin(2\\pi x) + \\frac{2}{\\pi^2\\alpha}(1-e^{-\\pi^2\\alpha t})\\sin(\\pi x)$$


```python
import numpy as np

def exact_solution(x,t, alpha):
    '''
    Returns the exact solution of the 1D heat equation with
    heat source term sin(np.pi*x) and initial condition sin(2*np.pi*x)

    Parameters
    ----------
    x : array of floats, grid points coordinates
    t : float, time

    Returns
    -------
    f : array of floats, exact solution
    '''
    f = (np.exp(-4*np.pi**2*alpha*t) * np.sin(2*np.pi*x)
      + 2.0*(1-np.exp(-np.pi**2*alpha*t)) * np.sin(np.pi*x)
      / (np.pi**2*alpha))

    return f

import romtools as rt
class HeatSnapshots(rt.AbstractSnapshotData):
    def __init__(self, snapshots: list):
        rt.AbstractSnapshotData.__init__(self, var_names=['T'])
        self.snapshots = snapshots

    def getMeshGids(self):
        # this method is a noop for now but needs to be defined
        # since it is an abstract method in the base class
        pass

    def getSnapshotsAsListOfArrays(self):
        return self.snapshots

if __name__=="__main__":
    numPoints, numTimes = 21, 11
    x     = np.linspace(0., 1., numPoints)
    times = np.linspace(0., 5., numTimes)

    alpha = 0.1
    data = [exact_solution(x, t, alpha) for t in times]
    snapshots = HeatSnapshots(data)
```
# License
```plaintext
.. include:: ../LICENSE
```
'''

__all__ = ['snapshot_data', 'trial_space', 'trial_space_utils', 'workflows', 'hyper_reduction']

__docformat__ = "restructuredtext" # required to generate the license

from romtools.snapshot_data import *
from romtools.trial_space import *
