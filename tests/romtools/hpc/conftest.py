import sys

import pytest

from romtools.hpc.configuration import Configuration


@pytest.fixture
def make_config():
    def _make(**overrides):
        config = Configuration.defaults()
        config.remote = "test-host"
        config.user = "test-user"
        config.timeout = 0
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    return _make


@pytest.fixture(autouse=True)
def _isolate_argv(monkeypatch):
    """
    Configuration() reads real sys.argv unless given an explicit list, so
    pytest's own command line would otherwise leak into these tests.
    """
    monkeypatch.setattr(sys, "argv", ["prog"])
