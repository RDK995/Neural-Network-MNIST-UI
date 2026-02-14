"""Unit tests for visualization utility helpers."""

from __future__ import annotations

import numpy as np

from mnist_explorer.ui.render import format_top_contribs, layer_y_positions, mix_with_white, pick_indices


def test_pick_indices_returns_full_range_when_k_exceeds_n() -> None:
    # Asking for more samples than available should return every index.
    idx = pick_indices(5, 10)
    assert np.array_equal(idx, np.array([0, 1, 2, 3, 4], dtype=np.int32))


def test_pick_indices_returns_middle_index_when_k_is_one() -> None:
    # With one sample, helper chooses the middle item.
    idx = pick_indices(9, 1)
    assert np.array_equal(idx, np.array([4], dtype=np.int32))


def test_pick_indices_returns_sorted_unique_values_for_general_case() -> None:
    # For normal sampling, output should be strictly increasing and unique.
    idx = pick_indices(100, 12)
    assert len(idx) <= 12
    assert np.all(idx[:-1] < idx[1:])
    assert len(np.unique(idx)) == len(idx)
    assert int(idx[0]) >= 0
    assert int(idx[-1]) <= 99


def test_mix_with_white_respects_bounds() -> None:
    # Intensity 0 means full white, intensity 1 means original color.
    assert mix_with_white("#123456", 0.0) == "#ffffff"
    assert mix_with_white("#123456", 1.0) == "#123456"
    # Values above 1.0 are clamped.
    assert mix_with_white("#123456", 2.0) == "#123456"


def test_mix_with_white_half_intensity_matches_expected_blend() -> None:
    # Red at 50% intensity blended with white should become a light red.
    # Formula in helper is channel-wise linear interpolation.
    assert mix_with_white("#ff0000", 0.5) == "#ff7f7f"


def test_layer_y_positions_handles_single_and_multi_counts() -> None:
    # Single node should be vertically centered.
    single = layer_y_positions(10.0, 30.0, 1)
    assert np.allclose(single, np.array([20.0], dtype=np.float32))

    # Three nodes should be evenly spaced from top to bottom.
    multi = layer_y_positions(10.0, 30.0, 3)
    assert np.allclose(multi, np.array([10.0, 20.0, 30.0], dtype=np.float32))
    assert multi.dtype == np.float32


def test_format_top_contribs_includes_positive_and_negative_sections() -> None:
    # Output text should include both sections for explainability panel.
    values = np.array([0.5, -0.2, 0.1, -0.7, 0.9], dtype=np.float32)
    text = format_top_contribs(values, prefix="h1_", top_n=2)

    assert "Top positive contributors:" in text
    assert "Top negative contributors:" in text
    assert "+ h1_" in text
    assert "- h1_" in text


def test_format_top_contribs_reports_expected_top_and_bottom_indices() -> None:
    # Build a vector with clearly known strongest/weakest entries.
    values = np.array([0.01, 0.7, -0.2, 0.4, -0.9, 0.65], dtype=np.float32)
    text = format_top_contribs(values, prefix="n_", top_n=2)

    # Highest positives should include indices 1 and 5.
    assert "n_  1" in text
    assert "n_  5" in text
    # Most negative entries should include indices 4 and 2.
    assert "n_  4" in text
    assert "n_  2" in text
