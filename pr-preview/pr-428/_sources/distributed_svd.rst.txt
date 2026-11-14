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

The implementation uses a two-level Tall-Skinny QR (TSQR) factorization:

1. Each rank computes ``A_local = Q_local @ R_local``.
2. Rank zero gathers only the local ``R_local`` factors and performs a second
   reduced QR factorization of their vertical stack.
3. Rank zero computes the SVD of the final reduced factor.
4. The singular values and right singular vectors are broadcast, while the
   reduced left transformations are scattered and multiplied by each
   ``Q_local``.

This avoids forming :math:`A^H A`, which would square the condition number.

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
