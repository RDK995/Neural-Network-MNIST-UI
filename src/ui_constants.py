"""Centralized UI constants for the MNIST visual trainer.

This file only stores values (numbers, colors, labels).
Keeping these in one place makes the UI easier to tune later because
you do not need to search through the full application for each value.
"""

# The model expects 28x28 pixel images (MNIST format).
DRAW_GRID_SIZE = 28
# The visible drawing canvas is larger so users can draw comfortably.
# 280 means each logical MNIST pixel is displayed as a 10x10 square.
DRAW_CANVAS_SIZE = 280

# Main window sizing defaults.
WINDOW_SIZE = "1500x980"
WINDOW_MIN_SIZE = (1300, 860)

# Core color palette used by the app.
COLOR_BG = "#f3f7fb"
COLOR_CARD = "#ffffff"
COLOR_INK = "#0f172a"
COLOR_SUB = "#475569"
COLOR_EDGE = "#cbd5e1"

# Colors used by the status banner at the bottom.
COLOR_STATUS_INFO_BG = "#e0f2fe"
COLOR_STATUS_INFO_FG = "#0c4a6e"
COLOR_STATUS_WARN_BG = "#fff7ed"
COLOR_STATUS_WARN_FG = "#9a3412"

# Neutral button colors (normal, hover, pressed).
COLOR_NEUTRAL_BTN = "#e5e7eb"
COLOR_NEUTRAL_BTN_HOVER = "#d1d5db"
COLOR_NEUTRAL_BTN_PRESS = "#c7ccd4"

# Small paint brush kernel used when drawing with the mouse.
# Each tuple is: (row_offset, col_offset, paint_strength).
# This creates smoother strokes than editing a single pixel at a time.
DRAW_BRUSH = [
    (-1, -1, 0.30),
    (-1, 0, 0.55),
    (-1, 1, 0.30),
    (0, -1, 0.55),
    (0, 0, 1.00),
    (0, 1, 0.55),
    (1, -1, 0.30),
    (1, 0, 0.55),
    (1, 1, 0.30),
]

# Time between "thinking" animation stages in milliseconds.
THINK_STEP_MS = 320
# Human-readable labels shown while animation advances layer-by-layer.
THINK_STAGE_LABELS = [
    "Reading the input pixels",
    "Activating hidden layer 1",
    "Applying dropout path (inference mode)",
    "Activating hidden layer 2",
    "Computing output probabilities",
]
