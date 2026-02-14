"""Unit tests for preprocessing helpers used by the drawing UI."""

from __future__ import annotations

import numpy as np

from ui_preprocessing import preprocess_drawn_digit


def _center_of_mass(image_2d: np.ndarray) -> tuple[float, float]:
    mass = float(np.sum(image_2d))
    rr, cc = np.indices(image_2d.shape)
    cy = float(np.sum(rr * image_2d) / mass)
    cx = float(np.sum(cc * image_2d) / mass)
    return cy, cx


def test_preprocess_empty_input_returns_flat_zeros() -> None:
    draw_img = np.zeros((28, 28), dtype=np.float32)
    out = preprocess_drawn_digit(draw_img)

    assert out.shape == (784,)
    assert np.allclose(out, 0.0)


def test_preprocess_output_is_normalized_and_non_empty_for_drawn_digit() -> None:
    draw_img = np.zeros((28, 28), dtype=np.float32)
    draw_img[8:20, 10:16] = 0.9

    out = preprocess_drawn_digit(draw_img)

    assert out.shape == (784,)
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= 0.0
    assert float(out.sum()) > 0.0


def test_preprocess_recenters_off_center_stroke() -> None:
    draw_img = np.zeros((28, 28), dtype=np.float32)
    # Put a bright block in the upper-left corner.
    draw_img[1:7, 1:7] = 1.0

    out = preprocess_drawn_digit(draw_img).reshape(28, 28)
    cy, cx = _center_of_mass(out)

    # After recentering, center of mass should be near image center.
    assert 11.5 <= cy <= 15.5
    assert 11.5 <= cx <= 15.5
