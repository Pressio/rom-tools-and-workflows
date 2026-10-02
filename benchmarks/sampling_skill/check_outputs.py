"""Independent artifact checks. Exit zero means all objective checks passed."""
import json
from pathlib import Path
import sys
import numpy as np


def check(root):
    root = Path(root)
    checks = {}
    try:
        samples = np.loadtxt(root / "sampling_output/sample_parameters.txt")
        checks["sample_shape"] = samples.shape == (8, 2)
        checks["finite_and_bounds"] = bool(np.isfinite(samples).all() and ((samples >= 0) & (samples <= 1)).all())
        checks["nonconstant_parameters"] = bool(checks["sample_shape"] and (np.ptp(samples, axis=0) > 0).all())
        dirs = sorted((root / "sampling_output").glob("run_*"))
        checks["eight_runs"] = len(dirs) == 8
        correct = []
        for i in range(8):
            d = root / "sampling_output" / f"run_{i}"
            params = json.loads((d / "parameters.json").read_text())
            value = json.loads((d / "result.json").read_text())["value"]
            correct.append((d / "passed.txt").exists() and
                           np.allclose([params["alpha"], params["beta"]], samples[i]) and
                           np.isclose(value, samples[i, 0] + 2 * samples[i, 1]))
        checks["numerical_outputs"] = bool(all(correct))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        checks["readable_outputs"] = False
        checks["error"] = str(error)
    checks["audit_present"] = (root / "sampling_output/AUDIT.md").is_file()
    return {"checks": checks, "passed": all(v is True for k, v in checks.items() if k != "error")}


if __name__ == "__main__":
    result = check(sys.argv[1])
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
