import numpy as np

try:
    import mpi4py
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")

from romtools.linalg.linalg import _snapshot_loader, _streaming_pod_mpi

def test_streaming_pod_mpi_rank3_5x8_k1_p1():
    np.random.seed(327)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "tests/romtools/linalg/snapshots/rank3_5x8"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    N = 5
    M = 8
    k = 1
    p = 1

    # execution
    U, S, Vt = _streaming_pod_mpi(
        snapshot_loader=loader,
        N=N, M=M,
        k=k, p=p,
        comm=comm
    )

    if rank == 0:
        assert U.shape == (N, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, M)

        # POD singular values
        gold_S = np.array([23.46])
        print("k1_p1" + str(S))

def test_streaming_pod_mpi_rank3_5x8_k1_p2():
    np.random.seed(327)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "tests/romtools/linalg/snapshots/rank3_5x8"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    N = 5
    M = 8
    k = 1
    p = 2

    # execution
    U, S, Vt = _streaming_pod_mpi(
        snapshot_loader=loader,
        N=N, M=M,
        k=k, p=p,
        comm=comm
    )

    if rank == 0:
        assert U.shape == (N, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, M)

        # POD singular values
        print("k1_p2" + str(S))

def test_streaming_pod_mpi_rank3_5x8_k3_p1():
    np.random.seed(327)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = "tests/romtools/linalg/snapshots/rank3_5x8"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    N = 5
    M = 8
    k = 3
    p = 1

    # execution
    U, S, Vt = _streaming_pod_mpi(
        snapshot_loader=loader,
        N=N, M=M,
        k=k, p=p,
        comm=comm
    )

    if rank == 0:
        assert U.shape == (N, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, M)

        # POD singular values
        gold_S = np.array([23.46, 7.69, 1.54])
        print("k3_p1" + str(S))
        print(gold_S)
        assert np.allclose(S, gold_S, atol=1e-2)

if __name__ == "__main__":
    test_streaming_pod_mpi_rank3_5x8_k1_p1()
    test_streaming_pod_mpi_rank3_5x8_k1_p2()
    test_streaming_pod_mpi_rank3_5x8_k3_p1()
