"""Reusable math/color helpers for visualization rendering.

These helpers are pure functions (no UI state), which makes them easy
to test and easy to reuse in other visualization modules.
"""

from __future__ import annotations

import numpy as np


def pick_indices(n: int, k: int) -> np.ndarray:
    """Pick k representative indices from range [0, n-1].

    Why this matters:
    Some layers have hundreds of neurons, but drawing all of them
    would clutter the canvas. This returns an evenly spread subset.
    """
    if k >= n:
        return np.arange(n, dtype=np.int32)
    if k <= 1:
        return np.array([n // 2], dtype=np.int32)
    return np.unique(np.linspace(0, n - 1, num=k, dtype=np.int32))


def mix_with_white(hex_color: str, intensity: float) -> str:
    """Blend a color with white based on intensity in [0,1].

    intensity=0 -> white
    intensity=1 -> original color
    """
    # Clamp value so invalid inputs still produce valid color output.
    intensity = float(np.clip(intensity, 0.0, 1.0))
    # Remove leading '#' if provided and parse RGB bytes.
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Linear interpolation channel-by-channel toward white (255,255,255).
    rr = int((255 * (1.0 - intensity)) + (r * intensity))
    gg = int((255 * (1.0 - intensity)) + (g * intensity))
    bb = int((255 * (1.0 - intensity)) + (b * intensity))
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def layer_y_positions(top: float, bottom: float, count: int) -> np.ndarray:
    """Return evenly spaced y positions between top and bottom."""
    if count <= 1:
        # If only one node, center it vertically.
        return np.array([(top + bottom) / 2.0], dtype=np.float32)
    return np.linspace(top, bottom, num=count, dtype=np.float32)


def format_top_contribs(values: np.ndarray, prefix: str, top_n: int = 5) -> str:
    """Create a human-readable positive/negative contribution summary.

    values are contribution scores (for example activation * weight).
    """
    # Largest values contribute most positively to the decision.
    pos_idx = np.argsort(values)[-top_n:][::-1]
    # Smallest values contribute most negatively to the decision.
    neg_idx = np.argsort(values)[:top_n]
    pos_lines = [f"  + {prefix}{int(i):>3}: {float(values[i]): .5f}" for i in pos_idx]
    neg_lines = [f"  - {prefix}{int(i):>3}: {float(values[i]): .5f}" for i in neg_idx]
    return "\n".join(["Top positive contributors:", *pos_lines, "Top negative contributors:", *neg_lines])
