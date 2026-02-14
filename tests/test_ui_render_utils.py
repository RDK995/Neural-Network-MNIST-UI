"""Unit tests for visualization utility helpers."""

from __future__ import annotations

import numpy as np

from ui_render_utils import format_top_contribs, layer_y_positions, mix_with_white, pick_indices


def test_pick_indices_returns_full_range_when_k_exceeds_n() -> None:
    idx = pick_indices(5, 10)
    assert np.array_equal(idx, np.array([0, 1, 2, 3, 4], dtype=np.int32))


def test_pick_indices_returns_middle_index_when_k_is_one() -> None:
    idx = pick_indices(9, 1)
    assert np.array_equal(idx, np.array([4], dtype=np.int32))


def test_mix_with_white_respects_bounds() -> None:
    assert mix_with_white("#123456", 0.0) == "#ffffff"
    assert mix_with_white("#123456", 1.0) == "#123456"
    # Values above 1.0 are clamped.
    assert mix_with_white("#123456", 2.0) == "#123456"


def test_layer_y_positions_handles_single_and_multi_counts() -> None:
    single = layer_y_positions(10.0, 30.0, 1)
    assert np.allclose(single, np.array([20.0], dtype=np.float32))

    multi = layer_y_positions(10.0, 30.0, 3)
    assert np.allclose(multi, np.array([10.0, 20.0, 30.0], dtype=np.float32))


def test_format_top_contribs_includes_positive_and_negative_sections() -> None:
    values = np.array([0.5, -0.2, 0.1, -0.7, 0.9], dtype=np.float32)
    text = format_top_contribs(values, prefix="h1_", top_n=2)

    assert "Top positive contributors:" in text
    assert "Top negative contributors:" in text
    assert "+ h1_" in text
    assert "- h1_" in text
