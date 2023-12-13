import romtools as rt
import typing
import copy
import numpy as np
import pytest
import matplotlib.pyplot as plt

def exact_solution(x,t, alpha):
    """
    Returns the exact solution of the 1D heat equation with
    heat source term sin(np.pi*x) and initial condition sin(2*np.pi*x)

    Parameters
    ----------
    x : array of floats, grid points coordinates
    t : float, time

    Returns
    -------
    f : array of floats, exact solution
    """
    f = (np.exp(-4*np.pi**2*alpha*t) * np.sin(2*np.pi*x)
      + 2.0*(1-np.exp(-np.pi**2*alpha*t)) * np.sin(np.pi*x)
      / (np.pi**2*alpha))

    return f

import romtools as rt
class HeatSnapshots(rt.AbstractSnapshotData):
    def __init__(self, snapshots: list):
        self.snapshots = snapshots

    def getMeshGids(self):
      return np.arange(21,dtype='int')

    def getSnapshotTensor(self):
        ## put into a numpy array of shape N_vars x N_x x N_t
        return np.array(self.snapshots).transpose()[None]


@pytest.mark.mpi_skip
def test_demo1():
    numPoints, numTimes = 21, 11
    x     = np.linspace(0., 1., numPoints)
    times = np.linspace(0., 5., numTimes)
    alpha = 0.1
    data = [exact_solution(x, t, alpha) for t in times]
    snapshots = HeatSnapshots(data)
    assert snapshots.getSnapshotTensor().shape[0] == 1
    assert snapshots.getSnapshotTensor().shape[1] == 21
    assert snapshots.getSnapshotTensor().shape[2] == 11

if __name__=="__main__":
    test_demo1()
