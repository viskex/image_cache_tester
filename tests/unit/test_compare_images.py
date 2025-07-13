# Copyright (C) 2024-2025 by the viskex authors
#
# This file is part of image cache testing for viskex.
#
# SPDX-License-Identifier: MIT
"""Tests for image_cache_tester.compare_images module."""

import contextlib
import io
import os
import tempfile

import numpy as np
import PIL
import pytest
import pyvista

import image_cache_tester.compare_images


def test_compare_images_single_pixel_success(image_cache: str) -> None:
    """Test that two images with a single non-zero pixel in the same position are the same."""
    expected_image_path = os.path.join(image_cache, "test_compare_images_single_pixel_success.png")
    assert os.path.exists(expected_image_path)
    with tempfile.NamedTemporaryFile(suffix=".png") as actual_image_file:
        actual_image_path = actual_image_file.name
        actual_image = PIL.Image.new("RGB", (50, 50))
        actual_image.putpixel((0, 0), (255, 0, 0))
        actual_image.save(actual_image_path)
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            actual_image_copy, expected_image, difference_image = image_cache_tester.compare_images._compare_images(
                actual_image_path, expected_image_path, True)
        assert np.array_equal(np.asarray(actual_image_copy), np.asarray(actual_image))
        assert difference_image.getbbox() is None
        assert stdout_buffer.getvalue() == ""
        stdout_buffer.close()


def test_compare_images_single_pixel_failure(image_cache: str) -> None:
    """Test that two images with a single non-zero pixel in different positions are different."""
    expected_image_path = os.path.join(image_cache, "test_compare_images_single_pixel_failure.png")
    assert os.path.exists(expected_image_path)
    with tempfile.NamedTemporaryFile(suffix=".png") as actual_image_file:
        actual_image_path = actual_image_file.name
        actual_image = PIL.Image.new("RGB", (50, 50))
        actual_image.putpixel((1, 1), (255, 0, 0))
        actual_image.save(actual_image_path)
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            actual_image_copy, expected_image, difference_image = image_cache_tester.compare_images._compare_images(
                actual_image_path, expected_image_path, True)
        assert np.array_equal(np.asarray(actual_image_copy), np.asarray(actual_image))
        assert difference_image.getbbox() == (0, 0, 2, 2)
        assert stdout_buffer.getvalue().strip("\n") == (
            f"Bounding box for difference between {actual_image_path} and {expected_image_path} "
            f"is {difference_image.getbbox()}")
        stdout_buffer.close()


def test_compare_images_expected_not_existing(image_cache: str) -> None:
    """Test that image comparison fails when the expected image does not exist."""
    expected_image_path = os.path.join(image_cache, "test_compare_images_expected_not_existing.png")
    assert not os.path.exists(expected_image_path)
    with tempfile.NamedTemporaryFile(suffix=".png") as actual_image_file:
        actual_image_path = actual_image_file.name
        actual_image = PIL.Image.new("RGB", (50, 50), (255, 0, 0))
        actual_image.save(actual_image_path)
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            actual_image_copy, expected_image, difference_image = image_cache_tester.compare_images._compare_images(
                actual_image_path, expected_image_path, True)
        assert np.array_equal(np.asarray(actual_image_copy), np.asarray(actual_image))
        assert np.array_equal(np.asarray(expected_image), np.asarray(PIL.Image.new("RGB", (50, 50))))
        assert difference_image.getbbox() == (0, 0, 50, 50)
        assert (  # stdout will also contain a message with the bounding box for the difference
            f"Expected image {expected_image_path} does not exist: creating an empty one" in stdout_buffer.getvalue())
        stdout_buffer.close()


def test_compare_images_actual_not_existing(image_cache: str) -> None:
    """Test that image comparison raises a runtime error when the actual image does not exist."""
    expected_image_path = os.path.join(image_cache, "test_compare_images_actual_not_existing.png")
    assert os.path.exists(expected_image_path)
    actual_image_path = os.path.join("/a/non/exisisting/path.png")
    assert not os.path.exists(actual_image_path)
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), pytest.raises(RuntimeError) as excinfo:
        image_cache_tester.compare_images._compare_images(actual_image_path, expected_image_path, True)
    assert stdout_buffer.getvalue() == ""
    stdout_buffer.close()
    runtime_error_text = str(excinfo.value)
    assert runtime_error_text == f"{actual_image_path} does not exist"


def test_compare_images_different_size(image_cache: str) -> None:
    """Test that image comparison fails when images have different size."""
    expected_image_path = os.path.join(image_cache, "test_compare_images_different_size.png")
    assert os.path.exists(expected_image_path)
    with tempfile.NamedTemporaryFile(suffix=".png") as actual_image_file:
        actual_image_path = actual_image_file.name
        actual_image = PIL.Image.new("RGB", (51, 51))
        actual_image.save(actual_image_path)
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            actual_image_copy, expected_image, difference_image = image_cache_tester.compare_images._compare_images(
                actual_image_path, expected_image_path, True)
        assert np.array_equal(np.asarray(actual_image_copy), np.asarray(actual_image))
        assert np.array_equal(np.asarray(expected_image), np.asarray(difference_image))
        assert stdout_buffer.getvalue().strip("\n") == (
            f"Size of {actual_image_path} is (51, 51), while size of {expected_image_path} is (50, 50)")
        stdout_buffer.close()


def test_compare_images_pyvista_success(image_cache: str, check_monitor_resolution: bool) -> None:
    """Test that two images representing half a sphere are the same."""
    expected_image_path = os.path.join(image_cache, "test_compare_images_pyvista_success.png")
    assert os.path.exists(expected_image_path)
    with tempfile.NamedTemporaryFile(suffix=".png") as plotter_screenshot_file:
        plotter_screenshot_path = plotter_screenshot_file.name
        plotter = pyvista.Plotter(off_screen=True)  # type: ignore[no-untyped-call]
        plotter.add_mesh(pyvista.Sphere(start_phi=0, end_phi=90))
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            plotter_screenshot, expected_image, difference_image = image_cache_tester.compare_images.compare_images(
                plotter, plotter_screenshot_path, expected_image_path, True)
        assert np.array_equal(np.asarray(plotter_screenshot), np.asarray(expected_image))
        assert difference_image.getbbox() is None
        assert stdout_buffer.getvalue() == ""
        stdout_buffer.close()


def test_compare_images_pyvista_failure(image_cache: str, check_monitor_resolution: bool) -> None:
    """Test that two images representing different halves of a sphere are different."""
    expected_image_path = os.path.join(image_cache, "test_compare_images_pyvista_failure.png")
    assert os.path.exists(expected_image_path)
    expected_difference_image_path = os.path.join(
        image_cache, "test_compare_images_pyvista_failure_difference_image.png")
    assert os.path.exists(expected_difference_image_path)
    with tempfile.NamedTemporaryFile(suffix=".png") as plotter_screenshot_file:
        plotter_screenshot_path = plotter_screenshot_file.name
        plotter = pyvista.Plotter(off_screen=True)  # type: ignore[no-untyped-call]
        plotter.add_mesh(pyvista.Sphere(start_phi=90, end_phi=180))
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            plotter_screenshot, expected_image, difference_image = image_cache_tester.compare_images.compare_images(
                plotter, plotter_screenshot_path, expected_image_path, True)
        assert np.array_equal(
            np.asarray(difference_image), np.asarray(PIL.Image.open(expected_difference_image_path).convert("RGB")))
        assert difference_image.getbbox() == (248, 160, 775, 653)
        assert stdout_buffer.getvalue().strip("\n") == (
            f"Bounding box for difference between {plotter_screenshot_path} and {expected_image_path} "
            f"is {difference_image.getbbox()}")
        stdout_buffer.close()
