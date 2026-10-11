Distributed SVD
===============

``DistributedSvd`` computes a thin singular value decomposition of a
two-dimensional matrix distributed by rows across MPI ranks. It uses NumPy for
local QR and SVD operations and ``mpi4py`` for communication. The global input
matrix is never gathered on any rank.

API
---

.. autoclass:: romtools.linalg.DistributedSvd
   :members:
   :special-members: __call__
   :no-index:

Result distribution
-------------------

For a global matrix :math:`A \in \mathbb{C}^{m \times n}`, let
:math:`k = \min(m,n)`. Each rank supplies its local rows ``A_local`` and calls
the operation collectively. The result has the following distribution:

* ``U_local`` has shape ``(A_local.shape[0], k)`` and the same row distribution
  as the input.
* ``s`` has shape ``(k,)`` and is replicated on all ranks.
* ``Vh`` has shape ``(k, n)`` and is replicated on all ranks.

Uneven partitions and ranks with zero local rows are supported. Real and
complex numeric dtypes, wide matrices, tall matrices, rank-deficient matrices,
and globally empty dimensions are supported. Input arrays and the communicator
are not modified.

Supported NumPy options
-----------------------

The callable has the same argument names and defaults as ``numpy.linalg.svd``.
The following behavior is intentional:

* ``full_matrices=False, compute_uv=True, hermitian=False`` returns
  ``(U_local, s, Vh)``.
* ``compute_uv=False, hermitian=False`` returns only the replicated singular
  values. As in NumPy, ``full_matrices`` does not affect this result.
* ``full_matrices=True`` with ``compute_uv=True`` raises ``NotImplementedError``
  because constructing the global full left null space is not supported.
* ``hermitian=True`` raises ``NotImplementedError``.
* Inputs must be two-dimensional, and every rank must use the same column
  count, dtype, and options. Invalid distributed metadata raises ``ValueError``
  collectively.

Algorithm
---------

The implementation uses an adaptive Tall-Skinny QR (TSQR) factorization. Every
rank first computes ``A_local = Q_local @ R_local``. The communicator size then
selects one of two exact reduction strategies:

* Below ``tree_threshold``, rank zero gathers the local ``R_local`` factors and
  factors their vertical stack. This has low communication overhead for small
  MPI jobs.
* At or above ``tree_threshold``, neighboring ranks combine their ``R`` factors
  through a binary tree. The critical reduction path has
  :math:`O(\log P)` stages instead of placing all reduction work on rank zero.

After the final reduced factor is formed, rank zero computes its SVD. When
singular vectors are requested, the binary-tree path traverses the tree in
reverse and propagates the left transformations back to the leaf ranks. Each
rank then multiplies its first-level ``Q_local`` by its final transformation.

Both paths are mathematically equivalent TSQR factorizations and avoid forming
:math:`A^H A`, which would square the condition number.

Why the tree scales better: a high-level explanation
---------------------------------------------------

Both approaches start with each process producing a compact representation of
its local data. The difference is how those representations are combined.

With the rank-zero reduction, every process sends its representation to one
process (rank zero), which combines them all. As the number of processes grows,
rank zero has more data to hold and more combining work to do, becoming a
bottleneck.

The tree reduction combines representations in pairs, then combines those
results in pairs, and continues until one remains. Several pairs can work at
the same time. For example, eight processes combine their results in three
rounds. Each merge handles only two compact representations, avoiding one
large stack of all processes' representations on rank zero.

This shares the combining work and reduces the memory pressure on rank zero,
making the tree better suited to larger parallel jobs. Rank zero still performs
the final SVD, but only on the final compact representation. The original
matrix stays distributed in both approaches.

The benefit is scalability, rather than a more accurate approximation: both
paths compute the same mathematical decomposition, subject to floating-point
differences. The tree is not always faster, because its extra communication
rounds add overhead. Small jobs can therefore benefit from the simpler
rank-zero reduction, which is why the implementation switches between the two
using a configurable threshold.

Configuring the reduction threshold
-----------------------------------

The default is available as ``romtools.linalg.DEFAULT_TSQR_TREE_THRESHOLD`` and
is currently 8 MPI ranks. It can be overridden for a particular callable:

.. code-block:: python

   from romtools.linalg import DistributedSvd

   # Use the tree for jobs with 16 or more MPI ranks.
   distributed_svd = DistributedSvd(comm, tree_threshold=16)

The selection is:

.. code-block:: python

   if comm.Get_size() >= tree_threshold:
       # Binary-tree TSQR reduction
   else:
       # Rank-zero gather TSQR reduction

Set ``tree_threshold=1`` to force the tree for testing or benchmarking. Set it
above the MPI job size to force the rank-zero path. The best crossover depends
on the matrix column count, MPI implementation, network, and node layout, so
production clusters should benchmark both paths.

MPI example
-----------

The following program can be run with ``mpiexec -n 4 python distributed_pod.py``:

.. code-block:: python

   import numpy as np
   from mpi4py import MPI

   from romtools.linalg import DistributedSvd
   from romtools.vector_space import VectorSpaceFromPOD

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()

   # Every rank owns a potentially different number of matrix rows.
   local_rows = rank + 2
   rng = np.random.default_rng(1000 + rank)
   A_local = rng.normal(size=(local_rows, 6))

   distributed_svd = DistributedSvd(comm)
   U_local, s, Vh = distributed_svd(
      A_local,
      full_matrices=False,
      compute_uv=True,
      hermitian=False,
   )

   # VectorSpaceFromPOD accepts the same communicator-bound callable. With one
   # variable, the tensor's spatial axis is distributed across ranks.
   local_snapshots = A_local.reshape(1, local_rows, 6)
   pod_space = VectorSpaceFromPOD(
      local_snapshots,
      svdFnc=distributed_svd,
   )
   local_basis = pod_space.get_basis()
