# rom-tools-and-workflows
The ROM tools and workflows Python library comprises a set of algorithms for constructing and exploiting ROMs that rely on *abstract base classes* that encapsulate all the information needed to run a given algorithm. The philosophy is that, for any given application, the user simply needs to "fill out" a class that meets the required API of the abstract base class. Once this class is complete, the user gains access to all of our existing algorithms.


## What is in the rom-tools-and-workflows library?


Algorithms:

- Trial space computation:
  - Reduced-basis methods
  - Proper orthogonal decomposition
    - Algorithms are all compatible with basis scaling, basis splitting for multistate problems, and orthogonalization in different inner products
- Workflows for ROM construction and ROM exploitation:
  - ROM construction via reduced-basis greedy (RB-Greedy)
  - ROM/FOM exploitation via sampling
  - ROM/FOM exploitation via Dakota-driven sampling


Abstract base classes include:

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

## To setup environment, you need to pip install it
```bash
cd my-path/rom-tools-and-workflows
pip install .
```

## To run tests

Note: you need `pytest` installed

```bash
cd my-path/rom-tools-and-workflows
mkdir my_tests && cd my_tests
pytest ../
```


## Building the documentation

```
pdoc <path-to-romtools> --math --docformat google
```

this opens a browser with the module documentation.
More info about `pdoc` can be found [here](https://pdoc.dev/docs/pdoc.html)