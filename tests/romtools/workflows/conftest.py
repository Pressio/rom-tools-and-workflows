"""Fixtures for exercising model execution through workflow entry points."""

import pytest


class DeferredModel:
    blocking = False

    def __init__(self, model):
        self.model = model
        self.pending = set()
        self.completed = set()
        self.peak_pending = 0

    def populate_run_directory(self, directory, parameters):
        self.completed.discard(directory)
        return self.model.populate_run_directory(directory, parameters)

    def run_model(self, directory, parameters):
        self.pending.add(directory)
        self.peak_pending = max(self.peak_pending, len(self.pending))
        return DeferredJob(self, directory, parameters)

    def compute_qoi(self, directory, parameters):
        assert directory in self.completed
        return self.model.compute_qoi(directory, parameters)


class DeferredJob:
    def __init__(self, model, directory, parameters):
        self.model = model
        self.directory = directory
        self.parameters = parameters

    def done(self):
        return True

    def result(self):
        status = self.model.model.run_model(self.directory, self.parameters)
        self.model.pending.remove(self.directory)
        self.model.completed.add(self.directory)
        return 0 if status is None else status


@pytest.fixture
def deferred_model():
    return DeferredModel
