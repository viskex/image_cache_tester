# Copyright (C) 2024 by the viskex authors
#
# This file is part of image cache testing for viskex.
#
# SPDX-License-Identifier: MIT
"""Utility functions to be used in pytest configuration file for notebooks tests."""

import fnmatch
import os

import _pytest.pathlib
import nbformat
import nbvalx.pytest_hooks_notebooks
import pytest

collect_file = nbvalx.pytest_hooks_notebooks.collect_file
IPyNbFile = nbvalx.pytest_hooks_notebooks.IPyNbFile


def addoption(parser: pytest.Parser, pluginmanager: pytest.PytestPluginManager) -> None:
    """Add options to control verification of images from cache."""
    # Add options as in nbvalx
    nbvalx.pytest_hooks_notebooks.addoption(parser, pluginmanager)
    # Add options to control image verification
    parser.addoption("--verify-images", action="store_true", help="Verify images against image cache")
    parser.addoption("--refresh-image-cache", action="store_true", help="Refresh images in cache")


def sessionstart(session: pytest.Session) -> None:
    """Add image verification at the end of cells with plot."""
    # Do consistency checks between image verification and nbvalx options
    refresh_image_cache = session.config.option.refresh_image_cache
    if refresh_image_cache:
        session.config.option.verify_images = True
    verify_images = session.config.option.verify_images
    np = session.config.option.np
    # Add image cache to data to be linked if image verification options are requested
    if verify_images:
        link_data_in_work_dir = session.config.option.link_data_in_work_dir
        if "**/.image_cache" not in link_data_in_work_dir:
            link_data_in_work_dir.append("**/.image_cache")
    # Start session as in nbvalx
    nbvalx.pytest_hooks_notebooks.sessionstart(session)
    # Proceed with the rest only if image verification was requested
    if not verify_images:  # pragma: no cover
        return
    # Get all notebooks in the work directory
    calling_dirs = session.config.args
    assert len(calling_dirs) == 1
    calling_dir = calling_dirs[0]
    work_dir = os.path.join(calling_dir, session.config.option.work_dir)
    notebooks = dict()
    for dir_entry in _pytest.pathlib.visit(work_dir, lambda _: True):
        if dir_entry.is_file():
            filepath = str(dir_entry.path)
            if fnmatch.fnmatch(filepath, "**/*.ipynb"):
                assert not fnmatch.fnmatch(filepath, "**/*.log.ipynb")
                with open(filepath) as f:
                    notebooks[filepath] = nbformat.read(f, as_version=4)  # type: ignore[no-untyped-call]
    # Update notebook with image verification
    for (ipynb_path, nb) in notebooks.items():
        # Add a cell on top for computation of expected and actual image paths
        image_paths_code = f'''import os
import shutil

import IPython.display
import mpi4py.MPI
import numpy as np
import pyvista
import screeninfo

import viskex
import viskex.utils

import image_cache_tester.compare_images  # isort: skip

# Check that the monitor resolution is the same as the one which was used to generate pyvista images.
monitors = screeninfo.get_monitors()
if len(monitors) == 0:  # pragma: no cover
    raise RuntimeError("No monitors found")
elif len(monitors) > 1:  # pragma: no cover
    raise RuntimeError("Too many monitors found")
monitor = monitors[0]
if monitor.width != 1024 or monitor.height != 768:  # pragma: no cover
    raise RuntimeError(
        f"Wrong monitor resolution: expected 1024x768, got " + str(monitor.width) + "x" + str(monitor.height))

# Check that the pyvista jupyter backend is compatible with cache generation. Note that this
# cannot be done in the sessionstart code because that would force an import of viskex
# in the pytest hooks themselevs, rather than in the notebook.
pyvista_jupyter_backend = pyvista.global_theme.jupyter_backend
if pyvista_jupyter_backend not in ("html", "static"):
    raise RuntimeError(
        "Invalid pyvista jupyter backend: got " + pyvista_jupyter_backend + ", "
        "expected either html or static. "
        "Please set the environment variable VISKEX_PYVISTA_BACKEND.")


def _image_path_generator(directory: str, cell_id: str, comm_size: int, comm_rank: int) -> str:
    """Return the image name associated to a cell id."""
    ipynb_name = "{os.path.basename(ipynb_path)}"
    assert ipynb_name.endswith(".ipynb")
    ipynb_name = ipynb_name[:-6]  # drop extension
    ipynb_dir = "{os.path.dirname(ipynb_path)}"
    if np.issubdtype(viskex.utils.ScalarType, np.complexfloating):
        output_scalar_type = "complex"
    else:
        output_scalar_type = "real"
    output_dir = os.path.join(
        ipynb_dir, directory, ipynb_name, output_scalar_type, "comm_size=" + str(comm_size),
        "comm_rank=" + str(comm_rank), pyvista_jupyter_backend)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, cell_id + ".png")


def expected_image_path_generator(cell_id: str, comm_size: int, comm_rank: int) -> str:
    """Return the expected image path associated to a cell id."""
    return _image_path_generator(".image_cache", cell_id, comm_size, comm_rank)


def screenshot_image_path_generator(cell_id: str, comm_size: int, comm_rank: int) -> str:
    """Return where to save the screenshot associated to a cell id."""
    return _image_path_generator(".image_from_pytest", cell_id, comm_size, comm_rank)

def verify_plotter_image(plotter: pyvista.Plotter, cell_id: str) -> None:
    """Compare plotter image to cache, and raise an error if comparison fails."""
    screenshot_image_path = screenshot_image_path_generator(cell_id, {np}, mpi4py.MPI.COMM_WORLD.rank)
    expected_image_path = expected_image_path_generator(cell_id, {np}, mpi4py.MPI.COMM_WORLD.rank)
    screenshot_image, expected_image, difference_image = image_cache_tester.compare_images.compare_images(
        plotter, screenshot_image_path, expected_image_path, True)
    if difference_image.getbbox():
        IPython.display.display("Actual screenshot")
        IPython.display.display(screenshot_image)
        IPython.display.display("Expected screenshot")
        IPython.display.display(expected_image)
        IPython.display.display("Difference between screenshots")
        IPython.display.display(difference_image)
    if {refresh_image_cache}:
        shutil.copy(screenshot_image_path, expected_image_path)
    if difference_image.getbbox():
        raise RuntimeError("Image cache verification failed for cell " + cell_id)'''
        if np > 1:
            # Add the cell after the cluster start one, so that %%px is available
            image_paths_position = 1
            image_paths_code = "%%px --no-stream\n" + image_paths_code
        else:
            image_paths_position = 0
        image_paths_cell = nbformat.v4.new_code_cell(image_paths_code)  # type: ignore[no-untyped-call]
        image_paths_cell.id = "image_paths"
        nb.cells.insert(image_paths_position, image_paths_cell)
        # Process the rest of the cells
        for cell in nb.cells:
            if (
                cell.cell_type == "code"
                    and
                any(f"viskex.{plotter}.plot" in cell.source for plotter in [
                    "DolfinxPlotter", "FiredrakePlotter", "PyvistaPlotter",
                    "dolfinx", "firedrake"
                ])
            ):
                lines = cell.source.splitlines()
                cell_id = cell.id.replace("-", "_")
                # Need to change the code to ensure that the plotter is fetched as a return variable
                plotter_variable = ""
                for (line_index, line) in enumerate(lines):
                    if "viskex." in line and ".plot" in line:
                        assert plotter_variable == "", "Expecting one plot per cell"
                        if line.lstrip().startswith("viskex."):
                            # The plotter is returned but not stored in a local variable, so we need to add it
                            plotter_variable = f"plotter_{cell_id}"
                            line = line.replace("viskex.", f"{plotter_variable} = viskex.", 1)
                        else:
                            # The plotter is already stored in a local variable, so we need to get its name
                            first_viskex_occurrence = line.find("viskex.")
                            assert "=" in  line[:first_viskex_occurrence]
                            plotter_variable, _ = line.split("=", 1)
                        # Do not automatically show plotter
                        if "viskex.dolfinx" in line:
                            line = line.replace("viskex.dolfinx", "viskex.DolfinxPlotter")
                        elif "viskex.firedrake" in line:
                            line = line.replace("viskex.firedrake", "viskex.FiredrakePlotter")
                        # Replace in cell code
                        lines[line_index] = line
                assert plotter_variable != ""
                # Add call to verify_image
                verify_image_code = f"""verify_plotter_image({plotter_variable}, "{cell_id}")"""
                lines.append(verify_image_code)
                cell.source = "\n".join(lines)
    # Write modified notebooks to the work directory
    for (ipynb_path, nb) in notebooks.items():
        with open(ipynb_path, "w") as f:
            nbformat.write(nb, f)  # type: ignore[no-untyped-call]
