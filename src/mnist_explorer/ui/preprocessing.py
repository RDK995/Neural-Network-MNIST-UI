"""Preprocessing for converting a user drawing into MNIST-style input.

Goal:
- Take a hand-drawn 28x28 grayscale image from the UI.
- Make it look more like MNIST training examples.
- Return a 784-length vector ready for model inference.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def shift_image(image: np.ndarray, shift_r: int, shift_c: int) -> np.ndarray:
    """Shift an image up/down/left/right without changing output size.

    Any pixels shifted out of bounds are dropped.
    Newly uncovered areas are filled with zeros (black).
    """
    shifted = np.zeros_like(image)
    h, w = image.shape

    # Source coordinates define what we copy from the original image.
    # Destination coordinates define where copied pixels land in result.
    src_r0 = max(0, -shift_r)
    src_r1 = min(h, h - shift_r) if shift_r >= 0 else h
    dst_r0 = max(0, shift_r)
    dst_r1 = dst_r0 + (src_r1 - src_r0)

    src_c0 = max(0, -shift_c)
    src_c1 = min(w, w - shift_c) if shift_c >= 0 else w
    dst_c0 = max(0, shift_c)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    if src_r1 > src_r0 and src_c1 > src_c0:
        shifted[dst_r0:dst_r1, dst_c0:dst_c1] = image[src_r0:src_r1, src_c0:src_c1]
    return shifted


def preprocess_drawn_digit(draw_img: np.ndarray) -> np.ndarray:
    """Normalize a drawn digit into a model-ready 784-length vector.

    Pipeline summary:
    1) Find the drawn content (ignore near-empty background).
    2) Crop tightly around the digit.
    3) Scale longest side to ~20 pixels (MNIST-like size).
    4) Center on 28x28 canvas.
    5) Re-center by center-of-mass to reduce user placement errors.
    6) Clip to [0,1] and flatten to length 784.
    """
    # Work on float copy so we can safely modify values.
    img = draw_img.astype(np.float32).copy()
    # If canvas is empty, keep behavior predictable and return empty vector.
    if float(np.max(img)) <= 0.0:
        return img.reshape(-1)

    # Create mask of "ink" pixels; threshold avoids tiny accidental noise.
    mask = img > 0.10
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((28 * 28,), dtype=np.float32)

    # Crop to the tightest bounding box around drawn content.
    crop = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    # Normalize crop so brightest point is exactly 1.0.
    crop /= max(float(np.max(crop)), 1e-8)

    # Resize while preserving aspect ratio.
    # 20 pixels is a common target used for MNIST-style normalization.
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = tf.image.resize(
        crop[..., np.newaxis],
        size=(new_h, new_w),
        method="bilinear",
    ).numpy()[..., 0]

    # Paste resized digit into center of a 28x28 black canvas.
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    # Center-of-mass correction:
    # move the "mass center" of strokes near canvas center.
    # This helps when users draw off-center.
    mass = float(np.sum(canvas))
    if mass > 1e-8:
        rr, cc = np.indices(canvas.shape)
        cy = float(np.sum(rr * canvas) / mass)
        cx = float(np.sum(cc * canvas) / mass)
        shift_r = int(round(13.5 - cy))
        shift_c = int(round(13.5 - cx))
        canvas = shift_image(canvas, shift_r, shift_c)

    # Final safety normalization + flatten for dense network input shape (784,).
    return np.clip(canvas, 0.0, 1.0).astype(np.float32).reshape(-1)
