import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "solution.py"


def load():
    assert PATH.is_file(), "solution.py was not created"
    spec = importlib.util.spec_from_file_location("benchmark_solution", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_uniform


def test_deterministic_and_bounded():
    f = load()
    bounds = [(1., 2.), (-3., 5.)]
    a = f(bounds, 50, 17)
    assert isinstance(a, np.ndarray)
    assert a.shape == (50, 2)
    np.testing.assert_array_equal(a, f(bounds, 50, 17))
    assert np.all(a[:, 0] >= 1) and np.all(a[:, 0] <= 2)
    assert np.all(a[:, 1] >= -3) and np.all(a[:, 1] <= 5)
    np.testing.assert_array_equal(a, np.random.default_rng(17).uniform([1, -3], [2, 5], (50, 2)))


@pytest.mark.parametrize("bounds,n", [([(2, 1)], 3), ([(0, 0)], 3), ([(0, 1)], 0), ([(0, 1)], 1.5), ([], 3)])
def test_invalid(bounds, n):
    with pytest.raises(ValueError):
        load()(bounds, n, 1)
