import pytest
import numpy as np
from romtools.vector_space.utils.truncater import NoOpTruncater, BasisSizeTruncater, EnergyBasedTruncater
from romtools.linalg.parallel_utils import generate_random_local_and_global_arrays_impl
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    print("module 'mpi4py' is not installed")


@pytest.mark.mpi_skip
def test_noop_truncater():
    truncater = NoOpTruncater()
    my_basis = np.random.normal(size=(10, 2))
    singular_vectors = np.ones(2)
    my_truncated_basis = truncater.truncate(my_basis, singular_vectors)
    assert np.allclose(my_truncated_basis, my_basis)


@pytest.mark.mpi(min_size=3)
def test_noop_truncater_mpi():
    comm = MPI.COMM_WORLD
    truncater = NoOpTruncater()
    basis_shape = (10, 2)
    local_basis, global_basis = generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
    singular_vectors = np.ones(2)
    local_truncated_basis = truncater.truncate(local_basis, singular_vectors)
    assert np.allclose(local_truncated_basis, local_basis)


@pytest.mark.mpi_skip
def test_basis_size_truncater():
    reduced_size = 4
    truncater = BasisSizeTruncater(reduced_size)
    my_basis = np.random.normal(size=(10, 8))
    u,singular_values,v = np.linalg.svd(my_basis)
    my_truncated_basis = truncater.truncate(my_basis, singular_values)
    energy = truncater.get_energy()
    energy_gold = np.cumsum(singular_values**2)/np.sum(singular_values**2 + 1.e-30)
    assert np.allclose(energy_gold[reduced_size-1],energy)
    assert np.allclose(my_truncated_basis, my_basis[:, 0:reduced_size])
    assert my_truncated_basis.shape[1] == 4


@pytest.mark.mpi_skip
def test_basis_size_truncater_uses_supplied_total_energy():
    truncater = BasisSizeTruncater(1)
    basis = np.eye(2)
    truncater.truncate(basis, np.array([3.0, 2.0]), total_energy=14.0)

    assert np.isclose(truncater.get_energy(), 9.0 / 14.0)


@pytest.mark.mpi_skip
def test_basis_size_truncater_rejects_dimension_above_candidate_basis():
    truncater = BasisSizeTruncater(3)

    with pytest.raises(ValueError, match="greater than size of basis array"):
        truncater.truncate(np.eye(2), np.ones(2), total_energy=2.0)


@pytest.mark.mpi(min_size=3)
def test_basis_size_truncater_mpi():
    comm = MPI.COMM_WORLD
    reduced_size = 4
    truncater = BasisSizeTruncater(reduced_size)
    basis_shape = (10, 8)
    local_basis, global_basis = generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
    singular_values = np.ones(2)
    truncated_local_basis = truncater.truncate(local_basis, singular_values)
    assert np.allclose(truncated_local_basis, local_basis[:, 0:reduced_size])
    assert truncated_local_basis.shape[1] == 4


@pytest.mark.mpi_skip
def test_energy_truncater():
    np.random.seed(1)
    energy_threshold = 0.65
    singular_values = np.random.normal(size=10)**2
    np.sort(singular_values)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    K = 1
    for i in range(0, energy.size):
        if energy[i] < energy_threshold:
            K += 1

    truncater = EnergyBasedTruncater(energy_threshold)
    my_basis = np.random.normal(size=(10, 8))
    my_truncated_basis = truncater.truncate(my_basis, singular_values)

    energy = truncater.get_energy()
    energy_gold = np.cumsum(singular_values**2)/np.sum(singular_values**2 + 1.e-30)
    assert(np.allclose(energy,energy_gold[K-1]))
    assert np.allclose(my_truncated_basis, my_basis[:, 0:K])
    assert my_truncated_basis.shape[1] == K


@pytest.mark.mpi_skip
def test_energy_truncater_rejects_incomplete_energy_target():
    truncater = EnergyBasedTruncater(0.8)

    with pytest.raises(ValueError, match="increase max_basis_dimension"):
        truncater.truncate(
            np.ones((3, 1)), np.array([3.0]), total_energy=14.0
        )


@pytest.mark.mpi(min_size=3)
def test_energy_truncater_mpi():
    comm = MPI.COMM_WORLD
    np.random.seed(1)
    energy_threshold = 0.65
    singular_values = np.random.normal(size=10)**2
    np.sort(singular_values)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    K = 1
    for i in range(0, energy.size):
        if energy[i] < energy_threshold:
            K += 1

    truncater = EnergyBasedTruncater(energy_threshold)
    basis_shape = (10, 8)
    local_basis, global_basis = generate_random_local_and_global_arrays_impl(basis_shape, comm=comm)
    local_truncated_basis = truncater.truncate(local_basis, singular_values)
    assert np.allclose(local_truncated_basis, local_basis[:, 0:K])
    assert local_truncated_basis.shape[1] == K


if __name__ == "__main__":
    test_noop_truncater()
    test_noop_truncater_mpi()
    test_basis_size_truncater()
    test_basis_size_truncater_mpi()
    test_energy_truncater()
    test_energy_truncater_mpi()
