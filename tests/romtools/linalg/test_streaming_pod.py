import numpy as np
import pytest

from romtools.linalg.linalg import _snapshot_loader, _streaming_pod

def test_python_streaming_pod_serial_rank1_2x2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank1_2x2"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=1,
        n_snapshots=2,
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
        n_snapshots=2,
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
        n_snapshots=4,
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
        n_snapshots=4,
        k=1, p=1,
    )

    # POD singular values
    gold_S = 14.269
    assert np.isclose(S[0], gold_S, atol=0.1)

def test_python_streaming_pod_serial_rank2_2x4_block_size_3_k1_p1():
    # setup
    path = "tests/romtools/linalg/snapshots/rank2_2x4"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        n_snapshots=4,
        k=1, p=1,
    )

    # POD singular values
    gold_S = 14.269
    assert np.isclose(S[0], gold_S, atol=0.1)

def test_python_streaming_pod_serial_rank2_2x4_block_size_3_k2_p0():
    # setup
    path = "tests/romtools/linalg/snapshots/rank2_2x4"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        n_snapshots=4,
        k=2, p=0,
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
        n_snapshots=8,
        k=3, p=1,
    )

    # POD modes
    gold_U = np.array([
        [0.60745381, -0.12139554, -0.06134325],
        [0.71823071, 0.02364053, -0.0664763],
        [0.32900733, -0.00665702, 0.41838163],
        [0.05720764, 0.99161197, 0.03054496],
        [0.06021605, 0.03695457, -0.90323957],
    ])
    corr = abs(U[:, 0] @ gold_U[:, 0])
    assert np.isclose(corr, 1.0, atol=1e-4)

    # POD singular values
    gold_S = np.array([23.46, 7.69, 1.54])
    assert np.allclose(S[:3], gold_S, atol=1e-2)

def test_python_streaming_pod_serial_rank3_5x8_block_size_3_k3_p1():
    # setup
    path = "tests/romtools/linalg/snapshots/rank3_5x8"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        n_snapshots=8,
        k=3, p=1,
    )

    # POD modes
    gold_U = np.array([
        [0.60745381, -0.12139554, -0.06134325],
        [0.71823071, 0.02364053, -0.0664763],
        [0.32900733, -0.00665702, 0.41838163],
        [0.05720764, 0.99161197, 0.03054496],
        [0.06021605, 0.03695457, -0.90323957],
    ])
    corr = abs(U[:, 0] @ gold_U[:, 0])
    assert np.isclose(corr, 1.0, atol=1e-4)

    # POD singular values
    gold_S = np.array([23.46, 7.69, 1.54])
    assert np.allclose(S[:3], gold_S, atol=1e-2)

def test_python_streaming_pod_serial_rank3_5x8_block_size_3_k2_p2():
    # setup
    path = "tests/romtools/linalg/snapshots/rank3_5x8"
    loader = lambda s, e: _snapshot_loader(path, s, e)

    # execution
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=3,
        n_snapshots=8,
        k=2, p=2,
    )

    # POD modes
    gold_U = np.array([
        [0.60745381, -0.12139554, -0.06134325],
        [0.71823071, 0.02364053, -0.0664763],
        [0.32900733, -0.00665702, 0.41838163],
        [0.05720764, 0.99161197, 0.03054496],
        [0.06021605, 0.03695457, -0.90323957],
    ])
    corr = abs(U[:, 0] @ gold_U[:, 0])
    assert np.isclose(corr, 1.0, atol=1e-4)

    # POD singular values
    gold_S = np.array([23.46, 7.69])
    assert np.allclose(S, gold_S, atol=1e-2)


def test_python_streaming_pod_tensor_loader_and_ranges():
    snapshots = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
    requested_ranges = []

    class Loader:
        def __call__(self, start, end):
            requested_ranges.append((start, end))
            return snapshots[..., start:end]

    loader = Loader()

    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=2,
        n_snapshots=5,
        k=2,
        p=1,
    )

    assert U.shape == (3, 4, 2)
    assert S.shape == (2,)
    assert Vt.shape == (2, 5)
    assert requested_ranges == [(0, 2), (2, 4), (4, 5)] * 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0, "n_snapshots": 4, "k": 1, "p": 0},
        {"block_size": 2, "n_snapshots": 0, "k": 1, "p": 0},
        {"block_size": 2, "n_snapshots": 4, "k": 0, "p": 0},
        {"block_size": 2, "n_snapshots": 4, "k": 1, "p": -1},
    ],
)
def test_python_streaming_pod_rejects_invalid_parameters(kwargs):
    snapshots = np.ones((2, 4))
    loader = lambda start, end: snapshots[:, start:end]
    with pytest.raises(ValueError):
        _streaming_pod(snapshot_loader=loader, **kwargs)


def test_python_streaming_pod_rejects_incorrect_block_size():
    loader = lambda start, end: np.ones((3, end - start + 1))
    with pytest.raises(ValueError, match="incorrect number"):
        _streaming_pod(loader, block_size=2, n_snapshots=4, k=1, p=0)


def test_python_streaming_pod_rejects_inconsistent_state_shape():
    call_count = 0

    def loader(start, end):
        nonlocal call_count
        call_count += 1
        n_rows = 3 if call_count == 1 else 4
        return np.ones((n_rows, end - start))

    with pytest.raises(ValueError, match="inconsistent state dimensions"):
        _streaming_pod(loader, block_size=2, n_snapshots=4, k=1, p=0)


def test_python_streaming_pod_uses_custom_range_svd_once():
    snapshots = np.random.default_rng(17).normal(size=(6, 5))
    calls = []

    def custom_svd(matrix, **kwargs):
        calls.append((matrix.shape, kwargs))
        return np.linalg.svd(matrix, **kwargs)

    loader = lambda start, end: snapshots[:, start:end]
    U, S, Vt = _streaming_pod(
        snapshot_loader=loader,
        block_size=2,
        n_snapshots=5,
        k=2,
        p=3,
        svdFnc=custom_svd,
    )

    exact_U, exact_S, _ = np.linalg.svd(snapshots, full_matrices=False)
    assert len(calls) == 1
    assert calls[0][0] == (6, 5)
    assert U.shape == (6, 2)
    assert Vt.shape == (2, 5)
    assert np.allclose(S, exact_S[:2])
    assert np.allclose(np.abs(U.T @ exact_U[:, :2]), np.eye(2))


def test_python_streaming_pod_single_rank_comm_uses_serial_path():
    class SingleRankComm:
        def Get_size(self):
            return 1

    snapshots = np.random.default_rng(23).normal(size=(5, 4))
    loader = lambda start, end: snapshots[:, start:end]

    np.random.seed(11)
    expected = _streaming_pod(loader, 2, 4, 2, 2)
    np.random.seed(11)
    actual = _streaming_pod(loader, 2, 4, 2, 2, comm=SingleRankComm())

    for actual_value, expected_value in zip(actual, expected):
        assert np.allclose(actual_value, expected_value)

if __name__ == "__main__":
    test_python_streaming_pod_serial_rank1_2x2()

    test_python_streaming_pod_serial_rank2_2x2()

    test_python_streaming_pod_serial_rank2_2x4_block_size_1()
    test_python_streaming_pod_serial_rank2_2x4_block_size_2()
    test_python_streaming_pod_serial_rank2_2x4_block_size_3_k1_p1()
    test_python_streaming_pod_serial_rank2_2x4_block_size_3_k2_p0()

    test_python_streaming_pod_serial_rank3_5x8_block_size_2_k3_p1()
    test_python_streaming_pod_serial_rank3_5x8_block_size_3_k3_p1()
    test_python_streaming_pod_serial_rank3_5x8_block_size_3_k2_p2()
