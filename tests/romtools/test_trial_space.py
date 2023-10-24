import romtools as rt
import romtools.trial_space_utils as utils
import copy
import numpy as np
import pytest

class PythonSnapshotData(rt.AbstractSnapshotData):

    def __init__(self, snapshots):
        if isinstance(snapshots, list):
            self.snapshots = snapshots
        else:
            self.snapshots = [snapshots[:,i] for i in range(0,snapshots.shape[1])]

    def getSnapshotsAsListOfArrays(self):
        return self.snapshots

    def getMeshGids(self):
        return np.arange(0,5)

    def getVariableNames(self):
        return ['u','v','w']

    def getNumVars(self) -> int:
        return 3

@pytest.mark.mpi_skip
def test_list_snapshots_to_array():
    snapshots = [np.random.normal(size=(15,5)), np.random.normal(size=(15,2))]
    snapshot_data = PythonSnapshotData(snapshots)
    matrix = snapshot_data.getSnapshotsAsArray()
    assert matrix.shape[0] == 15
    assert matrix.shape[1] == 7

@pytest.mark.mpi_skip
def test_dictionary_trial_space():
    snapshots = np.random.normal(size=(15,5))
    snapshot_data = PythonSnapshotData(snapshots)
    # default test
    my_shifter = utils.NoOpShifter()
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.DictionaryTrialSpace(snapshot_data,my_shifter,my_splitter,my_orthogonalizer)
    assert(np.allclose(myTrialSpace.getBasis(),snapshots))
    assert(np.allclose(myTrialSpace.getShiftVector(),0))
    assert(np.allclose(myTrialSpace.getDimension(),5))

    # test with a shift
    my_shifter = utils.AverageShifter()
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.DictionaryTrialSpace(snapshot_data,my_shifter,my_splitter,my_orthogonalizer)
    assert(np.allclose(myTrialSpace.getBasis(),snapshots - np.mean(snapshots,axis=1)[:,None]))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),5))

    # test with a shift and splitting
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3,'F')
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.DictionaryTrialSpace(snapshot_data,my_shifter,my_splitter,my_orthogonalizer)
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),10))

    # test with a shift, splitting, and orthogonalization
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3,'F')
    my_orthogonalizer = utils.EuclideanL2Orthogonalizer()
    myTrialSpace = rt.DictionaryTrialSpace(snapshot_data,my_shifter,my_splitter,my_orthogonalizer)
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),10))
    basis = myTrialSpace.getBasis()
    assert(np.allclose( basis.transpose() @ basis, np.eye(10)))

@pytest.mark.mpi_skip
def test_trial_space_from_pod():
    snapshots = np.random.normal(size=(15,5))
    snapshot_data = PythonSnapshotData(snapshots)
    my_truncater = utils.NoOpTruncater()
    my_shifter = utils.NoOpShifter()
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromPOD(snapshot_data,my_truncater,my_shifter,my_splitter,my_orthogonalizer)
    ## truth trial space
    u,s,v = np.linalg.svd(snapshots,full_matrices=False)
    assert(np.allclose(u,myTrialSpace.getBasis()))
    assert(np.allclose(5,myTrialSpace.getDimension()))
    assert(np.allclose(0,myTrialSpace.getShiftVector()))

    # test with a shift
    my_shifter = utils.AverageShifter()
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromPOD(snapshot_data,my_truncater,my_shifter,my_splitter,my_orthogonalizer)
    u,s,v = np.linalg.svd(snapshots - np.mean(snapshots,axis=1)[:,None],full_matrices=False)
    assert(np.allclose(myTrialSpace.getBasis(),u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),5))

    # test with a shift and splitting
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3,'F')
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromPOD(snapshot_data,my_truncater,my_shifter,my_splitter,my_orthogonalizer)
    u,s,v = np.linalg.svd(my_splitter(snapshots - np.mean(snapshots,axis=1)[:,None]),full_matrices=False)
    assert(np.allclose(myTrialSpace.getBasis(),u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),10))


    # test with a shift, splitting, and orthogonalization
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3,'F')
    weighting = np.abs(np.random.normal(size=15))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    myTrialSpace = rt.TrialSpaceFromPOD(snapshot_data,my_truncater,my_shifter,my_splitter,my_orthogonalizer)
    u,s,v = np.linalg.svd(my_splitter(snapshots - np.mean(snapshots,axis=1)[:,None]),full_matrices=False)
    u = my_orthogonalizer(u)
    assert(np.allclose(myTrialSpace.getBasis(),u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),10))

@pytest.mark.mpi_skip
def test_trial_space_from_scaled_pod():
    n_var = 3
    snapshots = np.random.normal(size=(15,5))
    snapshot_data = PythonSnapshotData(copy.deepcopy(snapshots))
    my_truncater = utils.NoOpTruncater()
    my_shifter = utils.NoOpShifter()
    my_scaler = utils.VariableScaler('max_abs','F',n_var)
    my_splitter = utils.NoOpSplitter()
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromScaledPOD(snapshot_data,my_truncater,my_shifter,my_scaler,my_splitter,my_orthogonalizer)
    u,s,v = np.linalg.svd(my_scaler.preScaling(snapshots),full_matrices=False)
    u = my_scaler.postScaling(u)
    assert np.allclose(u,myTrialSpace.getBasis()) , print(u,myTrialSpace.getBasis())
    assert(np.allclose(5,myTrialSpace.getDimension()))
    assert(np.allclose(0,myTrialSpace.getShiftVector()))

    # test with a shift
    snapshot_data = PythonSnapshotData(copy.deepcopy(snapshots))
    my_shifter = utils.AverageShifter()
    my_splitter = utils.NoOpSplitter()
    my_scaler = utils.VariableScaler('max_abs','F',n_var)
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromScaledPOD(snapshot_data,my_truncater,my_shifter,my_scaler,my_splitter,my_orthogonalizer)
    shifted_snapshots,shift_vector = my_shifter(snapshots)
    my_scaler = utils.VariableScaler('max_abs','F',n_var)
    scaled_shifted_snapshots = my_scaler.preScaling(shifted_snapshots)
    u,s,v = np.linalg.svd(scaled_shifted_snapshots,full_matrices=False)
    u = my_scaler.postScaling(u)
    assert(np.allclose(myTrialSpace.getBasis(),u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),5))

    # test with a shift and splitting
    snapshot_data = PythonSnapshotData(copy.deepcopy(snapshots))
    my_scaler = utils.VariableScaler('max_abs','F',n_var)
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3,'F')
    my_orthogonalizer = utils.NoOpOrthogonalizer()
    myTrialSpace = rt.TrialSpaceFromScaledPOD(snapshot_data,my_truncater,my_shifter,my_scaler,my_splitter,my_orthogonalizer)
    shifted_snapshots,shift_vector = my_shifter(snapshots)
    my_scaler = utils.VariableScaler('max_abs','F',n_var)
    scaled_shifted_snapshots = my_scaler.preScaling(shifted_snapshots)
    scaled_shifted_and_split_snapshots = my_splitter(scaled_shifted_snapshots)
    u,s,v = np.linalg.svd(scaled_shifted_and_split_snapshots,full_matrices=False)
    u = my_scaler.postScaling(u)
    assert(np.allclose(myTrialSpace.getBasis(),u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),10))


    # test with a shift, splitting, and orthogonalization
    snapshot_data = PythonSnapshotData(copy.deepcopy(snapshots))
    my_scaler = utils.VariableScaler('max_abs','F',n_var)
    my_shifter = utils.AverageShifter()
    my_splitter = utils.BlockSplitter([[0],[1,2]],3,'F')
    weighting = np.abs(np.random.normal(size=15))
    my_orthogonalizer = utils.EuclideanVectorWeightedL2Orthogonalizer(weighting)
    myTrialSpace = rt.TrialSpaceFromScaledPOD(snapshot_data,my_truncater,my_shifter,my_scaler,my_splitter,my_orthogonalizer)
    shifted_snapshots,shift_vector = my_shifter(snapshots)
    my_scaler = utils.VariableScaler('max_abs','F',n_var)
    scaled_shifted_snapshots = my_scaler.preScaling(shifted_snapshots)
    scaled_shifted_and_split_snapshots = my_splitter(scaled_shifted_snapshots)
    u,s,v = np.linalg.svd(scaled_shifted_and_split_snapshots,full_matrices=False)
    u = my_scaler.postScaling(u)
    u = my_orthogonalizer(u)
    assert(np.allclose(myTrialSpace.getBasis(),u))
    assert(np.allclose(myTrialSpace.getShiftVector(),np.mean(snapshots,axis=1)))
    assert(np.allclose(myTrialSpace.getDimension(),10))


if __name__=="__main__":
    test_dictionary_trial_space()
    test_trial_space_from_pod()
    test_trial_space_from_scaled_pod()
