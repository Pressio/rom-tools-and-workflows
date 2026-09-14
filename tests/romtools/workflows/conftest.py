import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_argv(monkeypatch):
    """
    A dispatcher reads the real sys.argv, so pytest's own command line would
    otherwise leak into these tests through "-c".
    """
    monkeypatch.setattr(sys, "argv", ["prog"])
