'''
The hyper-reduction library comprises a set of routines for hyper-reduction. Presently, we support algorithms for

- The discrete empirical interpolation method (DEIM)

There are a number of approaches to sample mesh degrees of freedom and computed weight matrices for hyper-reduction approaches
The hyper_reduction module provides these functionalities. Note that certain aspects of hyper-reduction are inherently problem-dependent.
Implementations of these aspects, including the generation of residual snapshots and the construction of a sample mesh, are left to the user. 
'''
from romtools.hyper_reduction.ecsw import *
