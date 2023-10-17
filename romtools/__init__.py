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

'''

__docformat__ = "markdown"  # explicitly disable rST processing in the examples above.

from romtools.snapshot_data import *
from romtools.trial_space import *
