# Copyright (C) 2024 by the viskex authors
#
# This file is part of image cache testing for viskex.
#
# SPDX-License-Identifier: MIT
"""Compare two images using PIL."""

import os

import PIL.Image
import PIL.ImageChops
import pyvista


def compare_images(  # type: ignore[no-any-unimported]
    plotter: pyvista.Plotter, plotter_screenshot: str, expected_screenshot: str, verbose: bool
) -> tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    """
    Compare the image contained in a pyvista plotter to a cached one.

    Parameters
    ----------
    plotter
        The pyvista plotter to be used. The plotter must not have been shown yet.
    plotter_screenshot
        The location where to save a screenshot of the plotter.
    expected_screenshot
        Path to a cached image representing the expected content of the plotter.
    verbose
        Print additional messages on failed comparison.

    Returns
    -------
    :
        A tuple containing three images in pillow format: plotter screenshot, expected screenshot and their difference.
    """
    plotter.show(auto_close=False)
    plotter.screenshot(plotter_screenshot)
    return _compare_images(plotter_screenshot, expected_screenshot, verbose)


def _compare_images(
    actual_image_path: str, expected_image_path: str, verbose: bool
) -> tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    """
    Compare two images. RGBA images are silently converted to RGB ignoring alpha channels.

    Parameters
    ----------
    actual_image_path
        Path to the image content from the current evaluation of the code.
    expected_image_path
        Path to the reference image content.
    verbose
        Print additional messages on failed comparison.

    Returns
    -------
    :
        A tuple containing three images in pillow format: actual image, expected image and their difference.
    """
    if not os.path.exists(actual_image_path):
        raise RuntimeError(f"{actual_image_path} does not exist")
    else:
        actual_image = PIL.Image.open(actual_image_path).convert("RGB")

    if not os.path.exists(expected_image_path):
        if verbose:
            print(f"Expected image {expected_image_path} does not exist: creating an empty one")
        expected_image = PIL.Image.new("RGB", actual_image.size)
    else:
        expected_image = PIL.Image.open(expected_image_path).convert("RGB")

    if actual_image.size != expected_image.size:
        if verbose:
            print(
                f"Size of {actual_image_path} is {actual_image.size}, while size of {expected_image_path} "
                f"is {expected_image.size}")
        # Since we cannot compare images with different sizes, return a difference image equal to the entire
        # expected image, as if the actual image was empty.
        return actual_image, expected_image, expected_image

    difference_image = PIL.ImageChops.difference(actual_image, expected_image)
    if difference_image.getbbox() and verbose:
        print(
            f"Bounding box for difference between {actual_image_path} and {expected_image_path} "
            f"is {difference_image.getbbox()}")
    return actual_image, expected_image, difference_image
