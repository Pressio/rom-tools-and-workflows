import argparse
import os
import sys

import numpy as np

from romtools.hyper_reduction import QDEIM
from romtools.vector_space.utils.truncater import BasisSizeTruncater


EXAMPLE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.abspath(os.path.join(EXAMPLE_DIR, "..", "models"))
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from h2_air_flame import H2AirFlame  # noqa: E402


def _rhs_snapshots(model, states, parameters):
    snapshots = [model.rhs(state, *parameters).reshape(-1) for state in states]
    return np.column_stack(snapshots)


def main(smoke=False):
    if smoke:
        model = H2AirFlame(
            nx=10,
            ny=7,
            dt=1.0e-4,
            t_end=5.0e-4,
            snapshot_stride=1,
        )
        basis_dimension = 4
    else:
        model = H2AirFlame(
            nx=18,
            ny=10,
            dt=1.0e-4,
            t_end=2.0e-3,
            snapshot_stride=1,
        )
        basis_dimension = 10

    training_parameters = (2.0, 8.0, 40.0, 7.0)
    test_parameters = (2.2, 8.4, 36.0, 8.0)

    training_states, _ = model.solve(*training_parameters)
    rhs_training_snapshots = _rhs_snapshots(
        model, training_states, training_parameters
    )

    # QDEIM shares the DEIM reconstruction API; only the default point
    # selection changes to a pivoted-QR selection of interpolation points.
    qdeim = QDEIM.from_snapshots(
        rhs_training_snapshots,
        truncater=BasisSizeTruncater(basis_dimension),
    )

    test_states, _ = model.solve(*test_parameters)
    rhs_test_snapshots = _rhs_snapshots(model, test_states, test_parameters)

    relative_errors = []
    for rhs in rhs_test_snapshots.T:
        sampled_rhs = rhs[qdeim.sample_indices]
        reconstructed_rhs = qdeim.reconstruct(sampled_rhs)
        denominator = max(np.linalg.norm(rhs), np.finfo(float).eps)
        relative_errors.append(
            np.linalg.norm(reconstructed_rhs - rhs) / denominator
        )

    print(f"Full RHS dimension: {rhs_test_snapshots.shape[0]}")
    print(f"QDEIM sample points: {qdeim.sample_indices.size}")
    print(f"Mean relative RHS reconstruction error: {np.mean(relative_errors):.3e}")
    print(f"Max relative RHS reconstruction error: {np.max(relative_errors):.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct the H2-air flame RHS with QDEIM."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced configuration for CI validation.",
    )
    args = parser.parse_args()
    main(smoke=args.smoke)
