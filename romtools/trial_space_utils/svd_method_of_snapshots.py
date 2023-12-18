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

import numpy as np


## Helper functions will be moved to python mpi library at some point

def A_transpose_dot_bImpl(A, b, comm):
    '''
    @private
    Compute A^T A when A's columns are distributed
    '''
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    if num_processes == 1:
        return np.dot(A.transpose(), b)

    tmp = np.dot(A.transpose(), b)

    data = comm.gather(tmp.flatten(), root=0)

    ATb_glob = np.zeros(np.size(tmp))
    if mpi_rank == 0:
        for j in range(0, num_processes):
            ATb_glob[:] += data[j]
        for j in range(1, num_processes):
            comm.Send(ATb_glob, dest=j)

    else:
        comm.Recv(ATb_glob, source=0)
    return np.reshape(ATb_glob, np.shape(tmp))

def svd_method_of_snapshots_impl(snapshots, comm):
    '''@private'''
    #
    # outputs:
    # modes, Phi: numpy array where each column is a POD mode
    # energy, sigma: energy associated with each mode (singular values)

    STS = A_transpose_dot_bImpl(snapshots, snapshots, comm)
    Lam, E = np.linalg.eig(STS)
    sigma = np.sqrt(Lam)
    U = np.zeros(np.shape(snapshots))
    U[:] = np.dot(snapshots, np.dot(E, np.diag(1./sigma)))
    # sort by singular values
    ordering = np.argsort(sigma)[::-1]
    return U[:, ordering], sigma[ordering]

def global_abs_sum_impl(r, comm):
    '''@private'''
    mpi_rank = comm.Get_rank()
    num_processes = comm.Get_size()

    if num_processes == 1:
        return np.sum(r)

    data = comm.gather(np.sum(np.abs(r)), root=0)
    rn_glob = np.zeros(1)
    if mpi_rank == 0:
        for j in range(0, num_processes):
            rn_glob[:] += data[j]
        for j in range(1, num_processes):
            comm.Send(rn_glob, dest=j)
    else:
        comm.Recv(rn_glob, source=0)
    return rn_glob[0]


class SvdMethodOfSnapshots:
    '''
    #Parallel implementation of the method of snapshots to mimic the SVD for basis construction
    Sample usage:

                  mySvd = SvdMethodOfSnapshots(comm)
                  U,s,_ = mySvd(snapshots)

    where snapshots is the local portion of a distributed memory array.

    The standard reduced-basis problem requires solving the optimization problem
    $$ \\boldsymbol \\Phi = \\underset{ \\boldsymbol \\Phi_{\\*} \\in \\mathbb{R}^{N \\times K} | \\boldsymbol
    \\Phi_{\\*}^T \\boldsymbol \\Phi_{\\*} = \\mathbf{I}}{ \\mathrm{arg \\; min} } \\| \\Phi_{\\*} \\Phi_{\\*}^T
    \\mathbf{S} - \\mathbf{S} \\|_2,$$
    where $\\mathbf{S} \\in \\mathbb{R}^{N \\times N_s}$, with $N_s$ being the number of snapshots.
    The standard way to solve this is with the thin SVD. An alternative approach is to use the method of
    snapshts/kernel trick, see, e.g., https://web.stanford.edu/group/frg/course_work/CME345/CA-CME345-Ch4.pdf.
    Here, we instead solve the eigenvalue probelm
    $$ \\mathbf{S}^T \\mathbf{S} \\boldsymbol \\psi_i = \\lambda_i \\boldsymbol \\psi_i$$
    for $i = 1,\\ldots,N_s$. It can be shown that the left singular vectors from the SVD of $\\mathbf{S}$ are
    related to the eigen-vectors of the above by
    $$ \\mathbf{u}_i = \\frac{1}{\\sqrt{\\lambda_i}} \\mathbf{S} \\boldsymbol \\psi_i.$$

    An advantage of the method of snapshots is that it can be easily parallelized and is efficient if we don't
    have many snapshots. We compute $\\mathbf{S}^T \\mathbf{S}$ in parallel, and then solve the (typically small)
    eigenvalue problem in serial.
    '''

    def __init__(self, comm):
        self._comm = comm

    def __call__(self, snapshots: np.ndarray, full_matrices=False, compute_uv=False, hermitian=False):
        U, s = svd_method_of_snapshots_impl(snapshots, self._comm)
        return U, s, 'not_computed_in_method_of_snapshots'


class SvdMethodOfSnapshotsForQr:
    '''
    Same as SvdMethodOfSnapshots, but call only returns two arguments to be
    compatible with QR routine.
    '''
    def __init__(self, comm):
        self._comm = comm

    def __call__(self, snapshots: np.ndarray, full_matrices=False, compute_uv=False, hermitian=False):
        U, _ = svd_method_of_snapshots_impl(snapshots, self._comm)
        return U, 'not_computed_in_method_of_snapshots'
