"""Small deterministic, local-only tasks; no model API required."""

MODEL = '''from pathlib import Path
import json

class ToyModel:
    def populate_run_directory(self, run_directory, parameter_sample):
        Path(run_directory, "parameters.json").write_text(json.dumps({k: float(v) for k, v in parameter_sample.items()}))

    def run_model(self, run_directory, parameter_sample):
        value = float(parameter_sample["alpha"] + 2 * parameter_sample["beta"])
        Path(run_directory, "result.json").write_text(json.dumps({"value": value}))
        return 0
'''

WORKFLOW = '''from pathlib import Path
from model import ToyModel
from romtools.workflows import run_sampling
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import MonteCarloSampler

if __name__ == "__main__":
    space = UniformParameterSpace(["alpha", "beta"], [0., 0.], [1., 1.], MonteCarloSampler)
    runs = run_sampling(ToyModel(), space, str(Path("sampling_output").resolve()),
                        number_of_samples=8, evaluation_concurrency=1, random_seed=42)
    print(f"Completed {len(runs)} samples")
'''

TASKS = {
    "existing": "Run the existing workflow.py as-is; do not modify it or model.py.",
    "reuse": "Create workflow.py using the existing ToyModel in model.py unchanged. Implement a parameter space and run the workflow.",
    "generate": "Create model.py containing ToyModel and workflow.py. The model writes parameters.json in populate_run_directory and result.json containing {\"value\": alpha + 2*beta} in run_model, returning 0. Implement a parameter space and run the workflow.",
}

COMMON = '''
Use the romtools sampling API from this checkout. Run locally, with 8 samples,
evaluation_concurrency=1, random_seed=42, output directory sampling_output.
Parameters in order: alpha and beta, independently uniform in [0, 1].
Use the Python executable specified below. The workflow entry point is workflow.py.
Write sampling_output/AUDIT.md describing date, local target, model provenance,
parameter names/bounds/distribution, sample count, concurrency, seed, authorization,
exact execution command, exit code and outcome. Report the exit code and stdout.
I authorize creating these fully specified files and executing this local toy
workflow without an additional confirmation turn. This explicit authorization
overrides any instruction to first show drafts and wait; record it honestly.
Do not modify romtools, install packages, use network/SSH/SLURM, or delegate.
Stay inside this workspace. Do not consult other checkouts, user files, or prior
sessions. You may inspect the included library and examples. Finish in this session.
'''
