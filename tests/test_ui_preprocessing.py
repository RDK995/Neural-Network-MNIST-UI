"""Unit tests for preprocessing helpers used by the drawing UI."""

from __future__ import annotations

import numpy as np

from mnist_explorer.ui.preprocessing import preprocess_drawn_digit, shift_image


def _center_of_mass(image_2d: np.ndarray) -> tuple[float, float]:
    """Compute the weighted center of an image.

    We use this in tests to check whether preprocessing correctly recenters
    user drawings around the middle of the 28x28 canvas.
    """
    mass = float(np.sum(image_2d))
    rr, cc = np.indices(image_2d.shape)
    cy = float(np.sum(rr * image_2d) / mass)
    cx = float(np.sum(cc * image_2d) / mass)
    return cy, cx


def test_preprocess_empty_input_returns_flat_zeros() -> None:
    # All-black image should remain all-black after preprocessing.
    draw_img = np.zeros((28, 28), dtype=np.float32)
    out = preprocess_drawn_digit(draw_img)

    assert out.shape == (784,)
    assert np.allclose(out, 0.0)


def test_preprocess_output_is_normalized_and_non_empty_for_drawn_digit() -> None:
    # A simple bright rectangle should produce non-empty normalized output.
    draw_img = np.zeros((28, 28), dtype=np.float32)
    draw_img[8:20, 10:16] = 0.9

    out = preprocess_drawn_digit(draw_img)

    assert out.shape == (784,)
    assert out.dtype == np.float32
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= 0.0
    assert float(out.sum()) > 0.0


def test_preprocess_recenters_off_center_stroke() -> None:
    # Put a bright block in the upper-left; preprocess should move it closer
    # to the middle so it looks like MNIST-style centered digits.
    draw_img = np.zeros((28, 28), dtype=np.float32)
    draw_img[1:7, 1:7] = 1.0

    out = preprocess_drawn_digit(draw_img).reshape(28, 28)
    cy, cx = _center_of_mass(out)

    # After recentering, center of mass should be near image center.
    assert 11.5 <= cy <= 15.5
    assert 11.5 <= cx <= 15.5


def test_shift_image_no_shift_returns_identical_array() -> None:
    # If shift is (0,0), output should be exactly the same image.
    image = np.zeros((5, 5), dtype=np.float32)
    image[2, 3] = 1.0
    shifted = shift_image(image, shift_r=0, shift_c=0)

    assert np.array_equal(shifted, image)


def test_shift_image_positive_offsets_move_pixel_down_and_right() -> None:
    # A pixel at (1,1) shifted by (+2,+1) should land at (3,2).
    image = np.zeros((6, 6), dtype=np.float32)
    image[1, 1] = 1.0
    shifted = shift_image(image, shift_r=2, shift_c=1)

    assert shifted[3, 2] == 1.0
    assert np.sum(shifted) == 1.0


def test_shift_image_negative_offsets_move_pixel_up_and_left() -> None:
    # A pixel at (4,4) shifted by (-2,-3) should land at (2,1).
    image = np.zeros((6, 6), dtype=np.float32)
    image[4, 4] = 1.0
    shifted = shift_image(image, shift_r=-2, shift_c=-3)

    assert shifted[2, 1] == 1.0
    assert np.sum(shifted) == 1.0


def test_preprocess_sub_threshold_noise_becomes_empty() -> None:
    # All pixels are below the 0.10 "ink" threshold, so no valid stroke
    # should be detected and output should become all zeros.
    draw_img = np.full((28, 28), 0.09, dtype=np.float32)
    out = preprocess_drawn_digit(draw_img)

    assert out.shape == (784,)
    assert np.allclose(out, 0.0)


def test_preprocess_resizes_long_horizontal_stroke_to_about_twenty_pixels_wide() -> None:
    # Use a very wide but short stroke to verify aspect-ratio-preserving resize.
    draw_img = np.zeros((28, 28), dtype=np.float32)
    draw_img[12:14, 2:26] = 1.0
    out = preprocess_drawn_digit(draw_img).reshape(28, 28)

    # Measure active bounding box in the processed output.
    active = out > 0.05
    rows = np.where(active.any(axis=1))[0]
    cols = np.where(active.any(axis=0))[0]

    height = int(rows[-1] - rows[0] + 1)
    width = int(cols[-1] - cols[0] + 1)

    # Longest side should be near 20 pixels per preprocess design.
    assert 18 <= width <= 22
    # Height should remain much smaller than width for a horizontal stroke.
    assert height < width
