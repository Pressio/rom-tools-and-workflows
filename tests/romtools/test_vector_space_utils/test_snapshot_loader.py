import romtools.vector_space.utils as utils
from romtools.vector_space.utils.snapshot_loader import SnapshotLoader


def test_snapshot_loader_is_exported_from_vector_space_utils():
    assert utils.SnapshotLoader is SnapshotLoader
