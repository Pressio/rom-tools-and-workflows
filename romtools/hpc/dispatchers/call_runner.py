"""
The wire contract behind dispatcher.call(): resolving a "module:qualname"
target, packing arguments and results into a JSON + .npz pair, and building the
standalone runner script that executes the target under a remote interpreter.

The runner has no access to romtools, so it is assembled from this module's own
source to keep both ends of the transfer on the same format.
"""

import os
import sys
import inspect
import importlib
import contextlib
from functools import lru_cache

import numpy as np

CALL_INPUT_JSON = "input.json"
CALL_INPUT_NPZ = "input_arrays.npz"
CALL_OUTPUT_JSON = "output.json"
CALL_OUTPUT_NPZ = "output_arrays.npz"
CALL_RUNNER = "runner.py"


def pack(obj, arrays):
    """Replace ndarrays with references into `arrays` so `obj` is JSON-safe."""
    if isinstance(obj, np.ndarray):
        name = f"arr_{len(arrays)}"
        arrays[name] = obj
        return {"__ndarray__": name}

    if isinstance(obj, tuple):
        return {"__tuple__": [pack(x, arrays) for x in obj]}

    if isinstance(obj, list):
        return [pack(x, arrays) for x in obj]

    if isinstance(obj, dict):
        return {k: pack(v, arrays) for k, v in obj.items()}

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def unpack(obj, arrays):
    """Inverse of pack(): restore ndarrays and tuples from their references."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return arrays[obj["__ndarray__"]]
        if "__tuple__" in obj:
            return tuple(unpack(x, arrays) for x in obj["__tuple__"])
        return {k: unpack(v, arrays) for k, v in obj.items()}

    if isinstance(obj, list):
        return [unpack(x, arrays) for x in obj]

    return obj


def resolve_target(target):
    """Resolve a "module:qualname" string to the object it names."""
    module_name, _, qualname = target.partition(":")
    if not module_name or not qualname:
        raise ValueError(
            f"Invalid call target {target!r}; expected 'module:qualname' "
            "(for example 'my_model:evaluate')."
        )

    try:
        obj = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(
            f"Could not import module {module_name!r} for call target {target!r}: {e}"
        ) from e

    for attr in qualname.split("."):
        obj = getattr(obj, attr)

    return obj


@contextlib.contextmanager
def working_directory(path: str):
    """Run in `path`, with `path` importable ahead of the rest of sys.path."""
    original_directory = os.getcwd()
    os.chdir(path)
    inserted = os.getcwd()
    sys.path.insert(0, inserted)
    try:
        yield
    finally:
        os.chdir(original_directory)
        with contextlib.suppress(ValueError):
            sys.path.remove(inserted)


_RUNNER_PREAMBLE = '''\
import os
import sys
import json
import importlib
import traceback

import numpy as np

# The runner is executed by path, which would otherwise leave its own staging
# directory (not the run directory) at the front of sys.path.
sys.path.insert(0, os.getcwd())
'''

_RUNNER_MAIN = '''\
def main():
    target, input_json, input_npz, output_json, output_npz = sys.argv[1:6]

    with open(input_json, "r") as f:
        payload = json.load(f)

    with np.load(input_npz, allow_pickle=False) as arrays:
        args = unpack(payload["args"], arrays)
        kwargs = unpack(payload["kwargs"], arrays)

    result = resolve_target(target)(*args, **kwargs)

    output_arrays = {}
    packed_result = pack(result, output_arrays)

    with open(output_json, "w") as f:
        json.dump({"result": packed_result}, f)

    np.savez(output_npz, **output_arrays)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


@lru_cache(maxsize=1)
def build_call_runner() -> str:
    """
    Build the standalone script that runs a call target on the remote host.

    The wire-format helpers are injected from this module's own source so the
    two ends of the transfer cannot drift apart.
    """
    injected = (pack, unpack, resolve_target)
    return "\n".join(
        [_RUNNER_PREAMBLE]
        + [inspect.getsource(fn) for fn in injected]
        + [_RUNNER_MAIN]
    )
