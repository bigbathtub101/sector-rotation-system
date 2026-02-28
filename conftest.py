"""
conftest.py — pytest fixtures for the Sector Rotation System test suite.
"""

import os
import shutil
import tempfile

import pytest

# Import helpers from the test module
from tests.test_all_modules import TestCounter, make_temp_db, seed_db


@pytest.fixture
def tc():
    """Return a fresh TestCounter for each test."""
    return TestCounter()


@pytest.fixture
def db_path():
    """Create a seeded temp SQLite DB and yield its path; clean up after the test."""
    path = make_temp_db()
    seed_db(path, n_days=60)
    yield path
    tmpdir = os.path.dirname(path)
    shutil.rmtree(tmpdir, ignore_errors=True)
