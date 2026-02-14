"""Export a small MNIST training reference CSV from the original dataset.

This script intentionally exports only a small subset so the repository stays
lightweight while still providing a concrete reference sample.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from tensorflow import keras


OUT_PATH = Path("data/mnist_train_reference_100.csv")
ROW_COUNT = 100
CACHED_DATASET_PATH = Path.home() / ".keras" / "datasets" / "mnist.npz"


def main() -> None:
    # Prefer local cache file if present, so this works without internet.
    if CACHED_DATASET_PATH.exists():
        with np.load(CACHED_DATASET_PATH, allow_pickle=False) as data:
            x_train = data["x_train"]
            y_train = data["y_train"]
    else:
        # Fallback to Keras loader (may download if cache missing).
        (x_train, y_train), _ = keras.datasets.mnist.load_data()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    header = ["label"] + [f"pixel_{i}" for i in range(28 * 28)]

    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for i in range(ROW_COUNT):
            flattened = x_train[i].reshape(-1).tolist()
            writer.writerow([int(y_train[i]), *flattened])

    print(f"Wrote {ROW_COUNT} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
