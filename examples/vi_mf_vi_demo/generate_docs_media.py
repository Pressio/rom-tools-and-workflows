"""Generate the VI/MF-VI analytic-sine figures used by the documentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.vi_mf_vi_demo.example import main


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(REPOSITORY_ROOT / "docs" / "source" / "demos" / "notebooks"),
    )
    parser.add_argument(
        "--work-dir",
        default=str(REPOSITORY_ROOT / ".docs_vi_mf_vi_work"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        smoke=False,
        work_dir=arguments.work_dir,
        output_dir=arguments.output_dir,
    )
