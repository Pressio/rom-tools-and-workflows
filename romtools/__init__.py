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
- Constructing trial spaces
- Constructing and exploiting ROMs via outer loop workflows

## Algorithms

- Trial space computation:
  - Reduced-basis methods
  - Proper orthogonal decomposition
    - Algorithms are all compatible with basis scaling, basis splitting for multistate problems, and orthogonalization
      in different inner products
- Workflows for ROM construction and ROM exploitation:
  - ROM construction via reduced-basis greedy (RB-Greedy)
  - ROM/FOM exploitation via sampling
  - ROM/FOM exploitation via Dakota-driven sampling


## Representative abstract base classes

- `TrialSpace`
  - This class defines the minimum API requirements for a trial space

  - Constructing a trial space relies on utilities like truncaters, orthogonalizers, etc. Abstract classes, and
    concrete implementations, exist for:

      - orthogonalizers
      - scalers
      - shifters
      - splitters
      - truncaters

- `ParameterSpace`
  - This class defines the minimum API of a parameter space. These parameter spaces are used in workflows for
    running/building ROMs

- Abstract couplers for greedy sampling, sampling, and coupling to Dakota.

# Demos/HOWTOs

In this section, we present some demos/examples showing how to use `romtools` in practice.

TBD

# License
```plaintext
.. include:: ../LICENSE
```
'''

__all__ = ['trial_space', 'trial_space_utils', 'workflows', 'hyper_reduction']

__docformat__ = "restructuredtext" # required to generate the license

from romtools.trial_space import *
from romtools.hyper_reduction import *
