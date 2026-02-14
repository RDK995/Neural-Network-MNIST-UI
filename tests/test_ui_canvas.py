"""Unit tests for canvas drawing helpers."""

from __future__ import annotations

import numpy as np

from mnist_explorer.ui.canvas import draw_pixel_grid


class _FakeCanvas:
    def __init__(self) -> None:
        self.created = 0
        self.configured = 0
        self.deleted = 0
        self._next_id = 1

    def delete(self, _what: str) -> None:
        self.deleted += 1

    def create_rectangle(self, *_args, **_kwargs) -> int:
        item = self._next_id
        self._next_id += 1
        self.created += 1
        return item

    def itemconfigure(self, _item: int, **_kwargs) -> None:
        self.configured += 1


def test_draw_pixel_grid_reuses_canvas_items_between_calls() -> None:
    canvas = _FakeCanvas()
    image = np.zeros((2, 2), dtype=np.float32)

    draw_pixel_grid(canvas=canvas, image_2d=image, margin=0, size=20)
    first_created = canvas.created
    first_configured = canvas.configured

    draw_pixel_grid(canvas=canvas, image_2d=image, margin=0, size=20)
    assert canvas.created == first_created
    assert canvas.configured == first_configured

    image[0, 0] = 1.0
    draw_pixel_grid(canvas=canvas, image_2d=image, margin=0, size=20)
    assert canvas.created == first_created
    assert canvas.configured == first_configured + 1


def test_draw_pixel_grid_rebuilds_cache_when_geometry_changes() -> None:
    canvas = _FakeCanvas()
    image = np.zeros((2, 2), dtype=np.float32)

    draw_pixel_grid(canvas=canvas, image_2d=image, margin=0, size=20)
    first_created = canvas.created

    draw_pixel_grid(canvas=canvas, image_2d=image, margin=1, size=20)

    assert canvas.deleted == 2
    assert canvas.created > first_created
