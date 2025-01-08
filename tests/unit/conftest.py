# Copyright (C) 2024-2025 by the viskex authors
#
# This file is part of image cache testing for viskex.
#
# SPDX-License-Identifier: MIT
"""pytest configuration file for unit tests."""

import os

import pytest
import screeninfo


@pytest.fixture
def image_cache() -> str:
    """Return the image_cache subdirectory."""
    return os.path.join(os.path.dirname(__file__), ".image_cache")


@pytest.fixture
def check_monitor_resolution() -> bool:
    """Check that the monitor resolution is the same as the one which was used to generate pyvista images."""
    monitors = screeninfo.get_monitors()
    if len(monitors) == 0:  # pragma: no cover
        raise RuntimeError("No monitors found")
    elif len(monitors) > 1:  # pragma: no cover
        raise RuntimeError("Too many monitors found")

    monitor = monitors[0]
    if monitor.width != 1024 or monitor.height != 768:  # pragma: no cover
        raise RuntimeError(f"Wrong monitor resolution: expected 1024x768, got {monitor.width}x{monitor.height}")

    # All checks were successful
    return True
