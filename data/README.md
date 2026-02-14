# MNIST Reference Data

This folder contains a small reference extract from the **original MNIST training set** used by this project.

- Source loader used in code: `keras.datasets.mnist.load_data()`
- Original training split size: `60,000` images
- Original test split size: `10,000` images
- Image shape: `28 x 28` grayscale (pixel range `0-255`)

Included file:
- `mnist_train_reference_100.csv`
  - First 100 training samples from the original training split
  - Columns:
    - `label` (0-9)
    - `pixel_0` ... `pixel_783` (flattened 28x28 image)

This sample is provided for quick human reference and inspection without adding the full dataset to git.
