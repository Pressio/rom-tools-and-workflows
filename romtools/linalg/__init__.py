'''This module defines the linear algebra functions used throughout the romtools library.
These functions can be run in serial or parallel.'''

from romtools.linalg.linalg import DistributedSvd

__all__ = ["DistributedSvd"]
