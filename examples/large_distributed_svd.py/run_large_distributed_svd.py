"""Run romtools' distributed SVD on a large row-distributed random matrix.

Every MPI rank creates only its own block of matrix rows. The global matrix is
never assembled. Run this program through ``srun``, ``mpiexec``, or ``mpirun``.
"""

import argparse
import socket
import sys
import time

import numpy as np
from mpi4py import MPI

from romtools.linalg.linalg import _distributed_svd


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--global-rows",
        type=int,
        required=True,
        help="total number of matrix rows across all MPI ranks",
    )
    parser.add_argument(
        "--columns",
        type=int,
        required=True,
        help="number of matrix columns",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="matrix data type (default: float64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="base random seed (default: 2026)",
    )
    parser.add_argument(
        "--values-only",
        action="store_true",
        help="compute only singular values, not U and Vh",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="also check distributed U and Vh orthogonality",
    )
    return parser.parse_args()


def _local_row_count(global_rows, rank, size):
    quotient, remainder = divmod(global_rows, size)
    return quotient + int(rank < remainder)


def _host_summary(comm):
    hosts = comm.gather(socket.gethostname(), root=0)
    if comm.Get_rank() != 0:
        return None
    summary = {}
    for host in hosts:
        summary[host] = summary.get(host, 0) + 1
    return summary


def _print_problem_summary(arguments, comm, local_rows, allocation_seconds):
    rank = comm.Get_rank()
    size = comm.Get_size()
    dtype = np.dtype(arguments.dtype)
    local_row_counts = comm.gather(local_rows, root=0)
    host_summary = _host_summary(comm)
    maximum_allocation_seconds = comm.reduce(
        allocation_seconds, op=MPI.MAX, root=0
    )

    if rank == 0:
        global_input_gib = (
            arguments.global_rows * arguments.columns * dtype.itemsize
            / 1024**3
        )
        largest_local_input_gib = (
            max(local_row_counts) * arguments.columns * dtype.itemsize
            / 1024**3
        )
        print("romtools large distributed SVD")
        print(
            f"MPI ranks: {size} on {len(host_summary)} host(s): "
            f"{host_summary}"
        )
        print(
            f"global matrix: {arguments.global_rows} x {arguments.columns} "
            f"{arguments.dtype} ({global_input_gib:.3f} GiB input)"
        )
        print(
            f"local rows: {min(local_row_counts)}..{max(local_row_counts)}; "
            f"largest local input: {largest_local_input_gib:.3f} GiB"
        )
        print(
            "mode: "
            + ("singular values only" if arguments.values_only else "U, s, Vh")
        )
        print(
            "matrix allocation time (slowest rank): "
            f"{maximum_allocation_seconds:.3f} s"
        )
        sys.stdout.flush()


def _verify_factors(local_u, singular_values, vh, dtype, comm):
    """Check orthogonality without gathering the distributed U matrix."""
    local_gram = local_u.T @ local_u
    global_gram = np.empty_like(local_gram)
    comm.Allreduce(local_gram, global_gram, op=MPI.SUM)
    identity = np.eye(singular_values.size, dtype=dtype)
    left_error = np.linalg.norm(global_gram - identity)
    right_error = np.linalg.norm(vh @ vh.T - identity)
    tolerance = 1.0e-3 if dtype == np.dtype("float32") else 1.0e-8
    passed = (
        np.isfinite(left_error)
        and np.isfinite(right_error)
        and left_error <= tolerance
        and right_error <= tolerance
    )
    return passed, left_error, right_error, tolerance


def _run(arguments, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    dtype = np.dtype(arguments.dtype)
    local_rows = _local_row_count(arguments.global_rows, rank, size)

    allocation_error = None
    allocation_start = time.perf_counter()
    try:
        random_generator = np.random.default_rng(arguments.seed + rank)
        local_matrix = random_generator.standard_normal(
            (local_rows, arguments.columns), dtype=dtype
        )
    except (MemoryError, ValueError) as exception:
        local_matrix = None
        allocation_error = f"rank {rank}: {exception}"
    allocation_seconds = time.perf_counter() - allocation_start

    allocation_errors = comm.allgather(allocation_error)
    allocation_errors = [error for error in allocation_errors if error]
    if allocation_errors:
        if rank == 0:
            print("Matrix allocation failed:", file=sys.stderr)
            for error in allocation_errors:
                print(f"  {error}", file=sys.stderr)
        return 2

    _print_problem_summary(arguments, comm, local_rows, allocation_seconds)

    comm.Barrier()
    start = time.perf_counter()
    try:
        result = _distributed_svd(
            local_matrix,
            comm=comm,
            full_matrices=False,
            compute_uv=not arguments.values_only,
            hermitian=False,
        )
        svd_error = None
    except Exception as exception:  # Report collective numerical/MPI failures.
        result = None
        svd_error = f"rank {rank}: {type(exception).__name__}: {exception}"
    elapsed_seconds = time.perf_counter() - start

    svd_errors = comm.allgather(svd_error)
    svd_errors = [error for error in svd_errors if error]
    if svd_errors:
        if rank == 0:
            print("Distributed SVD failed:", file=sys.stderr)
            for error in svd_errors:
                print(f"  {error}", file=sys.stderr)
        return 1

    maximum_seconds = comm.reduce(elapsed_seconds, op=MPI.MAX, root=0)
    average_seconds = comm.reduce(elapsed_seconds, op=MPI.SUM, root=0)

    if arguments.values_only:
        singular_values = result
        local_shape_is_valid = singular_values.shape == (arguments.columns,)
        verification = None
    else:
        local_u, singular_values, vh = result
        local_shape_is_valid = (
            local_u.shape == (local_rows, arguments.columns)
            and singular_values.shape == (arguments.columns,)
            and vh.shape == (arguments.columns, arguments.columns)
        )
        verification = None
        if arguments.verify:
            verification = _verify_factors(
                local_u, singular_values, vh, dtype, comm
            )

    local_basic_check = (
        local_shape_is_valid
        and np.all(np.isfinite(singular_values))
        and np.all(singular_values >= 0.0)
        and np.all(np.diff(singular_values) <= 0.0)
    )
    passed = comm.allreduce(local_basic_check, op=MPI.LAND)
    if verification is not None:
        passed = passed and comm.allreduce(verification[0], op=MPI.LAND)

    if rank == 0:
        average_seconds /= size
        print(f"SVD time (slowest rank): {maximum_seconds:.3f} s")
        print(f"SVD time (rank average): {average_seconds:.3f} s")
        print(
            "first singular values: "
            + np.array2string(singular_values[: min(10, singular_values.size)])
        )
        if verification is not None:
            _, left_error, right_error, tolerance = verification
            print(f"U orthogonality error: {left_error:.3e}")
            print(f"Vh orthogonality error: {right_error:.3e}")
            print(f"verification tolerance: {tolerance:.3e}")
        print("RESULT: PASS" if passed else "RESULT: FAIL")

    return 0 if passed else 1


def main():
    arguments = _parse_arguments()
    comm = MPI.COMM_WORLD

    error = None
    if arguments.global_rows < 1:
        error = "--global-rows must be positive"
    elif arguments.columns < 1:
        error = "--columns must be positive"
    elif arguments.global_rows < arguments.columns:
        error = "this runner requires --global-rows >= --columns"

    if error is not None:
        if comm.Get_rank() == 0:
            print(f"error: {error}", file=sys.stderr)
        return 2
    if arguments.values_only and arguments.verify:
        if comm.Get_rank() == 0:
            print("error: --verify cannot be used with --values-only", file=sys.stderr)
        return 2

    return _run(arguments, comm)


if __name__ == "__main__":
    sys.exit(main())
