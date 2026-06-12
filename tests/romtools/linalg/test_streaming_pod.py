from pathlib import Path

import numpy as np

from romtools.linalg.linalg import _streaming_pod

########################
###  Set up problem  ###
########################

def load_snapshot(dataset_dir: str, i: int):
    path = Path(dataset_dir)
    return np.loadtxt(path / f"snapshot_{i}.txt")

def snapshot_loader(dataset_dir: str, start: int, end: int):
    '''
    Parameters:
        dataset_dir
        start (int)
        end (int)
    Return:
        Xb
    '''
    snapshots = []
    for i in range(start, end):
        Xbi = load_snapshot(dataset_dir, i)
        snapshots.append(Xbi)
    Xb = np.column_stack(snapshots)
    return Xb

########################
###   Define Tests   ###
########################

def test_python_snapshot_loader_rank1_2x2_size1():
    Xb = snapshot_loader("tests/romtools/linalg/snapshots/rank1_2x2", 0, 1)
    gold = np.array([
        [2.],
        [1.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank2_2x2_size1():
    Xb = snapshot_loader("tests/romtools/linalg/snapshots/rank2_2x2", 0, 1)
    gold = np.array([
        [3.],
        [0.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank2_2x4_size1():
    Xb = snapshot_loader("tests/romtools/linalg/snapshots/rank2_2x4", 0, 1)
    gold = np.array([
        [1.],
        [5.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank2_2x4_size2():
    Xb = snapshot_loader("tests/romtools/linalg/snapshots/rank2_2x4", 0, 2)
    gold = np.array([
        [1., 2.],
        [5., 6.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank2_2x4_size3():
    Xb = snapshot_loader("tests/romtools/linalg/snapshots/rank2_2x4", 1, 4)
    gold = np.array([
        [2., 3., 4.],
        [6., 7., 8.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_streaming_pod_serial_rank1_2x2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank1_2x2"
    loader = lambda s, e: snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=1,
        N=2, M=2,
        k=1, p=1,
    )

    # POD singular values
    gold_S = 5.0
    assert np.isclose(S[0], gold_S, atol=1e-8)

    # POD modes
    # TODO: gold_U = np.array([2., 1.])

def test_python_streaming_pod_serial_rank2_2x2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank2_2x2"
    loader = lambda s, e: snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=1,
        N=2, M=2,
        k=1, p=1,
    )

    # POD modes
    gold_Uk = np.array([
        [1.],
        [0.]
    ])
    corr = abs(U[:, 0] @ gold_Uk[:, 0])
    assert np.isclose(corr, 1.0, atol=1e-2)

    # POD singular values
    gold_S = 3.
    assert np.isclose(S[0], gold_S, atol=0.1)

    # right singular vectors
    gold_Vk = np.array([[1., 0.]])
    corr_v = abs(Vt[0, :] @ gold_Vk[0, :])
    assert np.isclose(corr_v, 1.0, atol=1e-2)

def test_python_streaming_pod_serial_rank2_2x4_block_size_1():
    # setup
    path = "tests/romtools/linalg/snapshots/rank2_2x4"
    loader = lambda s, e: snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=1,
        N=2, M=4,
        k=1, p=1,
    )

    # POD modes
    gold_Uk = np.array([
        [-0.274],
        [0.962]
    ])
    # TODO, maybe?

    # POD singular values
    gold_S = 14.269
    assert np.isclose(S[0], gold_S, atol=0.1)

    # right singular vectors
    # TODO: gold_Vk = np.array([-0.302, -0.372, -0.443, -0.514])

def test_python_streaming_pod_serial_rank2_2x4_block_size_2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank2_2x4"
    loader = lambda s, e: snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=2,
        N=2, M=4,
        k=1, p=1,
    )

    # POD singular values
    gold_S = 14.269
    assert np.isclose(S[0], gold_S, atol=0.1)

def test_python_streaming_pod_serial_rank2_2x4_block_size_3():
    # setup
    path = "tests/romtools/linalg/snapshots/rank2_2x4"
    loader = lambda s, e: snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        N=2, M=4,
        k=1, p=1,
    )

    # POD singular values
    gold_S = 14.269
    assert np.isclose(S[0], gold_S, atol=0.1)

if __name__ == "__main__":
    # snapshot_loader
    test_python_snapshot_loader_rank1_2x2_size1()

    test_python_snapshot_loader_rank2_2x2_size1()

    test_python_snapshot_loader_rank2_2x4_size1()
    test_python_snapshot_loader_rank2_2x4_size2()
    test_python_snapshot_loader_rank2_2x4_size3()

    # streaming_pod
    test_python_streaming_pod_serial_rank1_2x2()

    test_python_streaming_pod_serial_rank2_2x2()

    test_python_streaming_pod_serial_rank2_2x4_block_size_1()
    test_python_streaming_pod_serial_rank2_2x4_block_size_2()
    test_python_streaming_pod_serial_rank2_2x4_block_size_3()
