from importlib import import_module as _import_module

from romtools.workflows.inverse.eki_drivers import *
from romtools.workflows.inverse.mf_eki_drivers import *

# Expose submodules as attributes so `import romtools.workflows.inverse.mf_eki_drivers as m`
# binds to the module instead of the function imported above.
#run_eki = _import_module("romtools.workflows.inverse.eki_drivers")
#run_mf_eki = _import_module("romtools.workflows.inverse.mf_eki_drivers")
