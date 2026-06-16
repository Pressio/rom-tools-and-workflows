import numpy as np

from romtools.linalg.linalg import _snapshot_loader, _streaming_pod

def test_python_streaming_pod_serial_rank1_2x2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank1_2x2"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=1,
        N=2, M=2,
        k=1, p=1,
    )

    # POD modes
    gold_U = np.array([[0.89442719], [0.44721360]])
    corr = abs(U[:, 0] @ gold_U[:, 0])
    assert np.isclose(corr, 1.0, atol=1e-4)

    # POD singular values
    gold_S = 5.0
    assert np.isclose(S[0], gold_S, atol=1e-8)

    # right singular vectors
    gold_Vk = np.array([[0.44721360, 0.89442719]])
    corr_v = abs(Vt[0, :] @ gold_Vk[0, :])
    assert np.isclose(corr_v, 1.0, atol=1e-4)

def test_python_streaming_pod_serial_rank2_2x2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank2_2x2"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=1,
        N=2, M=2,
        k=1, p=1,
    )

    # POD modes
    gold_Uk = np.array([[1.], [0.]])
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
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=1,
        N=2, M=4,
        k=2, p=0,
    )

    # POD modes
    gold_U = np.array([
        [0.3762, -0.9265],
        [0.9265, 0.3762]
    ])
    corr = abs(U[:, 0] @ gold_U[:, 0])
    assert np.isclose(corr, 1.0, atol=1e-4)

    # POD singular values
    gold_S = np.array([14.2267, 1.2691])
    assert np.allclose(S, gold_S, atol=0.1)

    # right singular vectors
    gold_Vk = np.array([
        [0.3513, 0.4436, 0.5358, 0.6281],
        [0.7589, 0.3212, -0.1165, -0.5542]
    ])
    corr_v = abs(Vt[0, :] @ gold_Vk[0, :])
    assert np.isclose(corr_v, 1.0, atol=1e-3)

def test_python_streaming_pod_serial_rank2_2x4_block_size_2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank2_2x4"
    loader = lambda s, e: _snapshot_loader(path, s, e)

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
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        N=2, M=4,
        k=1, p=1, # TODO, create test with k=2, p=0
    )

    # POD singular values
    gold_S = 14.269
    assert np.isclose(S[0], gold_S, atol=0.1)

def test_python_streaming_pod_serial_rank3_5x8_block_size_2_k3_p1():
    # setup
    path = "tests/romtools/linalg/snapshots/rank3_5x8"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=2,
        N=5, M=8,
        k=3, p=1,
    )

    # POD modes
    print(U)
    # TODO

    # POD singular values
    gold_S = np.array([23.46, 7.69, 1.54])
    assert np.allclose(S[:3], gold_S, atol=1e-2)

    # right singular vectors
    print(Vt)
    # TODO

def test_python_streaming_pod_serial_rank3_5x8_block_size_3_k3_p1():
    # setup
    path = "tests/romtools/linalg/snapshots/rank3_5x8"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        N=5, M=8,
        k=3, p=1,
    )

    # POD modes
    print(U)
    # TODO

    # POD singular values
    gold_S = np.array([23.46, 7.69, 1.54])
    assert np.allclose(S[:3], gold_S, atol=1e-2)

    # right singular vectors
    print(Vt)
    # TODO

def test_python_streaming_pod_serial_rank3_5x8_block_size_3_k2_p2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank3_5x8"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        N=5, M=8,
        k=2, p=2,
    )

    # POD modes
    print(U)
    # TODO

    # POD singular values
    gold_S = np.array([23.46, 7.69])
    assert np.allclose(S, gold_S, atol=1e-2)

    # right singular vectors
    print(Vt)
    # TODO

if __name__ == "__main__":
    test_python_streaming_pod_serial_rank1_2x2()

    test_python_streaming_pod_serial_rank2_2x2()

    test_python_streaming_pod_serial_rank2_2x4_block_size_1()
    test_python_streaming_pod_serial_rank2_2x4_block_size_2()
    test_python_streaming_pod_serial_rank2_2x4_block_size_3()

    test_python_streaming_pod_serial_rank3_5x8_block_size_2_k3_p1()
    test_python_streaming_pod_serial_rank3_5x8_block_size_3_k3_p1()
    test_python_streaming_pod_serial_rank3_5x8_block_size_3_k2_p2()
