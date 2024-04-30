from pathlib import Path
import pytest
import numpy as np

from romtools.vector_space import VectorSpace, VectorSpaceFromPOD
from romtools.vector_space.utils import outputter


@pytest.fixture(scope='module', name='vector_space')
def _fixture_vector_space() -> VectorSpaceFromPOD:
    return VectorSpaceFromPOD(snapshots=np.random.rand(10, 3, 1))


def test_npz_output(tmp_path: Path, vector_space: VectorSpace) -> None:
    outputter.npz_output(output_filename=f'{tmp_path}/test.npz',
                         vector_space=vector_space, compress=True)


def test_hdf5_output(tmp_path: Path, vector_space: VectorSpace) -> None:
    if not outputter.hdf5_available:
        pytest.skip("h5py not installed")
    else:
        outputter.hdf5_output(output_filename=f'{tmp_path}/test.hdf5',
                              vector_space=vector_space)
