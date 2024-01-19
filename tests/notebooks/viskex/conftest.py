# Copyright (C) 2024 by the viskex authors
#
# This file is part of image cache testing for viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for notebooks tests."""

import pytest

import image_cache_tester.pytest_hooks_notebooks

pytest_addoption = image_cache_tester.pytest_hooks_notebooks.addoption
pytest_collect_file = image_cache_tester.pytest_hooks_notebooks.collect_file
pytest_sessionstart = image_cache_tester.pytest_hooks_notebooks.sessionstart


def pytest_runtest_setup(item: image_cache_tester.pytest_hooks_notebooks.IPyNbFile) -> None:
    """Check backend availability."""
    # Get notebook name
    notebook_name = item.parent.name
    # Check backend availability depending on the item name
    if notebook_name.endswith("dolfinx.ipynb"):
        pytest.importorskip("dolfinx")
    elif notebook_name.endswith("firedrake.ipynb"):
        pytest.importorskip("firedrake")
    else:
        raise ValueError("Invalid notebook name " + notebook_name)
