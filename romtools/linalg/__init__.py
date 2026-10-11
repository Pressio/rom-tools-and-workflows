'''This module defines the linear algebra functions used throughout the romtools library.
These functions can be run in serial or parallel.'''

from romtools.linalg.linalg import DEFAULT_TSQR_TREE_THRESHOLD, DistributedSvd

__all__ = ["DEFAULT_TSQR_TREE_THRESHOLD", "DistributedSvd"]
