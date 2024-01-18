# Copyright (C) 2024 by the viskex authors
#
# This file is part of image cache testing for viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for notebooks tests."""

import nbvalx.pytest_hooks_notebooks
import pytest

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file


def pytest_sessionstart(session: pytest.Session) -> None:
    """Automatically add **/.image_cache as data to be linked in the work directory."""
    # Add mesh files as data to be linked
    link_data_in_work_dir = session.config.option.link_data_in_work_dir
    if len(link_data_in_work_dir) == 0:
        link_data_in_work_dir.append("**/.image_cache")
    else:
        assert len(link_data_in_work_dir) == 1
        assert link_data_in_work_dir[0] == "**/.image_cache"
    # Start session as in notebooks hooks
    nbvalx.pytest_hooks_notebooks.sessionstart(session)
