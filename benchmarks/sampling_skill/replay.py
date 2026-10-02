"""Run a submitted workflow while observing its actual sampling API arguments."""
import inspect
import importlib
import json
from pathlib import Path
import runpy
import sys

import romtools.workflows as workflows
sampling = importlib.import_module("romtools.workflows.sampling")
implementation = importlib.import_module("romtools.workflows.sampling.sampling")

original = implementation.run_sampling
calls = []


def observed(*args, **kwargs):
    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    calls.append({k: values[k] for k in ("number_of_samples", "evaluation_concurrency", "random_seed", "dry_run")})
    calls[-1]["names"] = list(values["parameter_space"].get_names())
    calls[-1]["dispatcher"] = type(values["dispatcher"]).__name__ if values["dispatcher"] is not None else "LocalDispatcher"
    return original(*args, **kwargs)


if __name__ == "__main__":
    workflows.run_sampling = sampling.run_sampling = implementation.run_sampling = observed
    sys.path.insert(0, str(Path.cwd()))
    try:
        runpy.run_path("workflow.py", run_name="__main__")
    finally:
        Path(sys.argv[1]).write_text(json.dumps(calls))
