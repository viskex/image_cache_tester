# Copyright (C) 2024 by the viskex authors
#
# This file is part of image cache testing for viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for unit tests."""

import os

import pytest


@pytest.fixture
def image_cache() -> str:
    """Return the image_cache subdirectory."""
    return os.path.join(os.path.dirname(__file__), ".image_cache")
