# Distributed SVD examples

These examples demonstrate two ways to construct a row-distributed POD basis
using only NumPy and MPI communication:

- `distributed_svd.py` calls `DistributedSvd` directly and uses its local left
  singular vectors as a POD basis.
- `vector_space_from_pod.py` passes `DistributedSvd` to `VectorSpaceFromPOD`
  and truncates the resulting basis.

In both examples, MPI ranks own different numbers of rows. The global snapshot
matrix and global POD basis are never gathered.

## Requirements

```bash
source .venv/bin/activate
python -m pip install -e '.[WithMPI]'
```

## `DistributedSvd` example

From the repository root, run:

```bash
mpiexec -n 4 python examples/distributed_svd/distributed_svd.py
```

## `VectorSpaceFromPOD` example

From the repository root, run:

```bash
mpiexec -n 4 python examples/distributed_svd/vector_space_from_pod.py
```
