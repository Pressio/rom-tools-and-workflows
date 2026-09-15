"""Run the analytic-sine VI/MF-VI documentation example."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.vi_mf_vi_demo.diagonal_example import main as run_diagonal_example


def main(
    smoke: bool = False,
    work_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    root = Path(work_dir).resolve() if work_dir else Path(__file__).parent / "work"
    output = Path(output_dir).resolve() if output_dir else Path(__file__).parent
    run_diagonal_example(
        smoke=smoke,
        work_dir=str(root / "diagonal"),
        output_dir=str(output),
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--work-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        smoke=arguments.smoke,
        work_dir=arguments.work_dir,
        output_dir=arguments.output_dir,
    )
