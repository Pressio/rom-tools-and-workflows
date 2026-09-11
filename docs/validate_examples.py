"""Explicitly validate executable documentation examples.

Sphinx rendering is intentionally configured not to execute notebooks. This
script is the single place where CI decides which documentation examples are
expected to run against the current romtools checkout.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Keep this list explicit. Adding a notebook here opts it into PR-blocking
# execution. Expensive or environment-specific examples should instead get a
# dedicated reduced/smoke validation path.
LIGHTWEIGHT_NOTEBOOKS = [
    "docs/source/demos/notebooks/basic_model.ipynb",
    "docs/source/demos/notebooks/external_model.ipynb",
    "docs/source/demos/notebooks/external_qoi_model.ipynb",
    "docs/source/demos/notebooks/basic_sampling.ipynb",
    "docs/source/demos/notebooks/dictionary_vector_space.ipynb",
    "docs/source/demos/notebooks/pod_vector_space.ipynb",
    "docs/source/demos/notebooks/streaming_pod_vector_space.ipynb",
    "docs/source/demos/notebooks/orthogonalized_vector_space.ipynb",
    "docs/source/demos/notebooks/composite_vector_space.ipynb",
    "docs/source/demos/notebooks/model_builder.ipynb",
    "docs/source/demos/notebooks/greedy_training.ipynb",
    "docs/source/demos/notebooks/parameter_space.ipynb",
]


def _execute_notebook(relative_path: str) -> None:
    notebook_path = REPOSITORY_ROOT / relative_path
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Documentation notebook not found: {relative_path}")

    print(f"Executing {relative_path}", flush=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()


def _run_eki_smoke_test() -> None:
    example = REPOSITORY_ROOT / "examples/eki_mf_eki_demo/example.py"
    print("Running reduced EKI/MF-EKI documentation smoke test", flush=True)

    with tempfile.TemporaryDirectory(prefix="romtools-docs-eki-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(
            [
                sys.executable,
                str(example),
                "--smoke",
                "--work-dir",
                str(tmp_path / "work"),
                "--output",
                str(tmp_path / "eki_mf_eki_demo.png"),
            ],
            cwd=REPOSITORY_ROOT,
            env=env,
            check=True,
        )


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")

    for notebook in LIGHTWEIGHT_NOTEBOOKS:
        _execute_notebook(notebook)

    _run_eki_smoke_test()
    print("Documentation example validation passed.", flush=True)


if __name__ == "__main__":
    main()
