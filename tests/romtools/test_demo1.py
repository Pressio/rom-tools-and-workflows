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
        rt.AbstractSnapshotData.__init__(self, var_names=['T'])
        self.snapshots = snapshots

    def getMeshGids(self):
        # this method is a noop for now but needs to be defined
        # since it is an abstract method in the base class
        pass

    def getSnapshotsAsListOfArrays(self):
        return self.snapshots

@pytest.mark.mpi_skip
def test_demo1():
    numPoints, numTimes = 21, 11
    x     = np.linspace(0., 1., numPoints)
    times = np.linspace(0., 5., numTimes)
    alpha = 0.1
    data = [exact_solution(x, t, alpha) for t in times]
    snapshots = HeatSnapshots(data)
    assert snapshots.getVariableNames() == ['T']
    assert snapshots.getNumVars() == 1
    assert len(snapshots.getSnapshotsAsListOfArrays()) == 11

if __name__=="__main__":
    test_demo1()
