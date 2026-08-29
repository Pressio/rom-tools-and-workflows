from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.mpi_skip
def test_cdr_uq_example_runs(tmp_path):
    repository_root = Path(__file__).resolve().parents[4]
    example = repository_root / "examples" / "uq_cdr_demo" / "example.py"
    subprocess.run(
        [
            sys.executable,
            str(example),
            "--output-directory",
            str(tmp_path),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with np.load(tmp_path / "monte_carlo" / "uq_stats.npz") as mc_stats:
        assert mc_stats["number_of_samples"][0] == 12
    with np.load(tmp_path / "multifidelity" / "uq_stats.npz") as mfmc_stats:
        assert mfmc_stats["pilot_sample_count"][0] == 4
        assert mfmc_stats["high_fidelity_equivalent_cost"][0] <= 12.0
