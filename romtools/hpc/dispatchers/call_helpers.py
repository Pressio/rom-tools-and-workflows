import numpy as np

_CALL_RUNNER = r"""
import sys
import json
import traceback
import importlib
import numpy as np

def unpack(obj, arrays):
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return arrays[obj["__ndarray__"]]
        if "__tuple__" in obj:
            return tuple(unpack(x, arrays) for x in obj["__tuple__"])
        return {k: unpack(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unpack(x, arrays) for x in obj]
    return obj

def pack(obj, arrays):
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

try:
    target = sys.argv[1]
    input_json = sys.argv[2]
    input_npz = sys.argv[3]
    output_json = sys.argv[4]
    output_npz = sys.argv[5]

    with open(input_json, "r") as f:
        payload = json.load(f)

    with np.load(input_npz, allow_pickle=False) as arrays:
        args = unpack(payload["args"], arrays)
        kwargs = unpack(payload["kwargs"], arrays)

    module_name, qualname = target.split(":", 1)
    obj = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)

    result = obj(*args, **kwargs)

    output_arrays = {}
    packed_result = pack(result, output_arrays)

    with open(output_json, "w") as f:
        json.dump({"result": packed_result}, f)

    np.savez(output_npz, **output_arrays)

except Exception:
    traceback.print_exc()
    sys.exit(1)
"""

def _pack(obj, arrays):
    if isinstance(obj, np.ndarray):
        name = f"arr_{len(arrays)}"
        arrays[name] = obj
        return {"__ndarray__": name}

    if isinstance(obj, tuple):
        return {"__tuple__": [_pack(x, arrays) for x in obj]}

    if isinstance(obj, list):
        return [_pack(x, arrays) for x in obj]

    if isinstance(obj, dict):
        return {k: _pack(v, arrays) for k, v in obj.items()}

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def _unpack(obj, arrays):
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return arrays[obj["__ndarray__"]]
        if "__tuple__" in obj:
            return tuple(_unpack(x, arrays) for x in obj["__tuple__"])
        return {k: _unpack(v, arrays) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_unpack(x, arrays) for x in obj]

    return obj
