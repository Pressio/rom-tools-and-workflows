import pytest
import numpy as np
import romtools as rt
import pressiolinalg.linalg as pla

def test_pressiolinalg_serial():

    test_arr = np.array([1,2,3,4,5,6])

    np_max = np.max(test_arr)
    pla_max = pla.max(test_arr)
    assert(np_max == pla_max)

    np_min = np.min(test_arr)
    pla_min = pla.min(test_arr)
    assert(np_min == pla_min)

if __name__ == "__main__":
    test_pressiolinalg_serial()