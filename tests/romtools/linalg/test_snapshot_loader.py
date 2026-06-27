import numpy as np

from romtools.linalg.linalg import _snapshot_loader

def test_python_snapshot_loader_rank1_2x2_size1():
    Xb = _snapshot_loader("tests/romtools/linalg/snapshots/rank1_2x2", 0, 1)
    gold = np.array([
        [2.],
        [1.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank2_2x2_size1():
    Xb = _snapshot_loader("tests/romtools/linalg/snapshots/rank2_2x2", 0, 1)
    gold = np.array([
        [3.],
        [0.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank2_2x4_size1():
    Xb = _snapshot_loader("tests/romtools/linalg/snapshots/rank2_2x4", 0, 1)
    gold = np.array([
        [1.],
        [5.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank2_2x4_size2():
    Xb = _snapshot_loader("tests/romtools/linalg/snapshots/rank2_2x4", 0, 2)
    gold = np.array([
        [1., 2.],
        [5., 6.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank2_2x4_size3():
    Xb = _snapshot_loader("tests/romtools/linalg/snapshots/rank2_2x4", 1, 4)
    gold = np.array([
        [2., 3., 4.],
        [6., 7., 8.]
    ])
    np.testing.assert_array_equal(Xb, gold)

def test_python_snapshot_loader_rank3_5x8_size1():
    Xb = _snapshot_loader("tests/romtools/linalg/snapshots/rank3_5x8", 3, 4)
    gold = np.array([
        [4.],
        [5.],
        [2.],
        [2.],
        [1.]
    ])
    np.testing.assert_array_equal(Xb, gold)

if __name__ == "__main__":
    test_python_snapshot_loader_rank1_2x2_size1()

    test_python_snapshot_loader_rank2_2x2_size1()

    test_python_snapshot_loader_rank2_2x4_size1()
    test_python_snapshot_loader_rank2_2x4_size2()
    test_python_snapshot_loader_rank2_2x4_size3()

    test_python_snapshot_loader_rank3_5x8_size1()
