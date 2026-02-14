"""Canvas and drawing-buffer helpers for the MNIST UI."""

from __future__ import annotations

import tkinter as tk

import numpy as np


def paint_brush_stamp(
    draw_buffer: np.ndarray,
    row: int,
    col: int,
    brush: list[tuple[int, int, float]],
) -> None:
    """Paint a soft brush stamp into draw_buffer centered at (row, col)."""
    rows, cols = draw_buffer.shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return

    for dr, dc, strength in brush:
        rr = row + dr
        cc = col + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            draw_buffer[rr, cc] = min(1.0, draw_buffer[rr, cc] + strength)


def draw_pixel_grid(
    canvas: tk.Canvas,
    image_2d: np.ndarray,
    margin: int,
    size: int,
) -> None:
    """Draw a grayscale image as a scaled pixel grid on a canvas.

    This function caches rectangle items and only updates pixels whose
    intensity changed since the last draw. That avoids expensive full redraws
    during high-frequency mouse motion events.
    """
    image_h, image_w = image_2d.shape
    key = (image_h, image_w, margin, size)
    cache = getattr(canvas, "_pixel_grid_cache", None)

    if cache is None or cache.get("key") != key:
        canvas.delete("all")
        cell_h = size / image_h
        cell_w = size / image_w
        canvas.create_rectangle(
            margin - 1,
            margin - 1,
            margin + size + 1,
            margin + size + 1,
            outline="#334155",
            width=2,
        )

        ids: list[list[int]] = []
        for r in range(image_h):
            row_ids: list[int] = []
            for c in range(image_w):
                x0 = margin + c * cell_w
                y0 = margin + r * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                item_id = canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill="#000000",
                    outline="#000000",
                )
                row_ids.append(item_id)
            ids.append(row_ids)

        cache = {
            "key": key,
            "ids": ids,
            "last_gray": np.full((image_h, image_w), -1, dtype=np.int16),
        }
        setattr(canvas, "_pixel_grid_cache", cache)

    gray_values = np.rint(np.clip(image_2d, 0.0, 1.0) * 255.0).astype(np.int16)
    changed = np.where(gray_values != cache["last_gray"])
    ids = cache["ids"]

    for r, c in zip(changed[0], changed[1]):
        g = int(gray_values[r, c])
        color = f"#{g:02x}{g:02x}{g:02x}"
        canvas.itemconfigure(ids[r][c], fill=color, outline=color)

    cache["last_gray"][changed] = gray_values[changed]
