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
from typing import Any, Tuple
import numpy as np
import romtools.linalg.linalg as la


class SvdMethodOfSnapshots:
    '''
    Parallel implementation of the method of snapshots for basis construction.

    Sample usage:

    .. code-block:: python

       my_svd = SvdMethodOfSnapshots(comm)
       U, s, _ = my_svd(snapshots)

    where snapshots is the local portion of a distributed memory array.

    The standard reduced-basis problem requires solving the optimization problem

    .. math::

       \\boldsymbol \\Phi = \\underset{ \\boldsymbol \\Phi_{\\ast} \\in \\mathbb{R}^{N \\times K} | \\boldsymbol
       \\Phi_{\\ast}^T \\boldsymbol \\Phi_{\\ast} = \\mathbf{I}}{ \\mathrm{arg \\; min} }
       \\| \\Phi_{\\ast} \\Phi_{\\ast}^T \\mathbf{S} - \\mathbf{S} \\|_2,

    where :math:`\\mathbf{S} \\in \\mathbb{R}^{N \\times N_s}`, with :math:`N_s` being the number of snapshots.
    The standard way to solve this is with the thin SVD. An alternative approach is to use the method of
    snapshots/kernel trick, see, e.g., https://web.stanford.edu/group/frg/course_work/CME345/CA-CME345-Ch4.pdf.
    Here, we instead solve the eigenvalue problem

    .. math::

       \\mathbf{S}^T \\mathbf{S} \\boldsymbol \\psi_i = \\lambda_i \\boldsymbol \\psi_i

    for :math:`i = 1,\\ldots,N_s`. It can be shown that the left singular vectors from the SVD of :math:`\\mathbf{S}` are
    related to the eigen-vectors of the above by

    .. math::

       \\mathbf{u}_i = \\frac{1}{\\sqrt{\\lambda_i}} \\mathbf{S} \\boldsymbol \\psi_i.

    An advantage of the method of snapshots is that it can be easily parallelized and is efficient if we don't
    have many snapshots. We compute :math:`\\mathbf{S}^T \\mathbf{S}` in parallel, and then solve the (typically small)
    eigenvalue problem in serial.
    '''

    def __init__(self, comm) -> None:
        self._comm = comm

    def __call__(self, snapshots: np.ndarray,
                 full_matrices: bool = False,
                 compute_uv: bool = False,
                 hermitian: bool = False) -> Tuple[np.ndarray, np.ndarray, Any]:
        U, s = la.thin_svd(snapshots, self._comm, method='method_of_snapshots')
        return U, s, 'not_computed_in_method_of_snapshots'


class SvdMethodOfSnapshotsForQr:
    '''
    Similar to SvdMethodOfSnapshots, but call only returns two arguments to be
    compatible with QR routine.
    '''
    def __init__(self, comm) -> None:
        self._comm = comm

    def __call__(self, snapshots: np.ndarray, 
                 mode: str = 'reduced') -> Tuple[np.ndarray, Any]:

        if mode != 'reduced':
            raise ValueError('mode must be = reduced')

        U, _ = la.thin_svd(snapshots, self._comm, method='method_of_snapshots')
        return U, 'not_computed_in_method_of_snapshots'
