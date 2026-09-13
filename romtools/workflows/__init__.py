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

'''
The workflows module contains all of our "outerloop" workflows used for ROM construction and ROM exploitation.
We currently support:
- Greedy sampling for construction of reduced-basis ROMs
- Random sampling for model exploitation
- Coupling classes to Dakota for
  - Random sampling
'''
from importlib import import_module as _import_module

from romtools.workflows.dakota import *
from romtools.workflows.greedy import *
from romtools.workflows.sampling import *
from romtools.workflows.inverse import *
from romtools.workflows.formatting import *
from romtools.workflows.uq import *
from romtools.workflows._work_dir_compat import patch_work_dir_argument as _patch_work_dir

for _module_name, _functions, _old_name in (
    ("sampling.sampling", ("run_sampling",), "absolute_sampling_directory"),
    ("sampling_with_holdout.sampling_with_holdout", ("run_sampling_with_holdout",), "absolute_work_directory"),
    ("greedy.run_greedy", ("run_greedy",), "absolute_greedy_work_directory"),
    ("uq.monte_carlo", ("run_monte_carlo", "run_multifidelity_monte_carlo"), "absolute_uq_directory"),
    ("inverse.ego_drivers", ("run_ego",), "absolute_ego_directory"),
    ("inverse.eki_drivers", ("run_eki",), "absolute_eki_directory"),
    ("inverse.mf_eki_drivers", ("run_mf_eki", "mf_eki_with_auto_rom"), "absolute_eki_directory"),
    ("inverse.vi_drivers", ("run_vi",), "absolute_vi_directory"),
    ("inverse.mf_vi_drivers", ("run_mf_vi", "mf_vi_with_auto_rom"), "absolute_vi_directory"),
):
    _module = _import_module(f"romtools.workflows.{_module_name}")
    for _function in _functions:
        globals()[_function] = _patch_work_dir(_module, _function, _old_name)
