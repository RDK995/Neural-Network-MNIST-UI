# MNIST Original Training Dataset Reference

This project uses the **original MNIST dataset** loaded via:

- `keras.datasets.mnist.load_data()`

The returned original split is:
- Training: `60,000` images and labels
- Test: `10,000` images and labels
- Image size: `28 x 28` grayscale pixels (`0-255`)

## Why this file exists
The full raw dataset is not committed to git in this repository. Instead, this file records the exact source and shape used by training code in `src/mnist_explorer/model/basic_nn.py`.

## Generate a local CSV reference extract
Run:

```bash
python src/mnist_explorer/tools/export_mnist_reference.py
```

This creates:
- `data/mnist_train_reference_100.csv`

The CSV includes the first 100 rows from the original training split with columns:
- `label`
- `pixel_0` through `pixel_783`

If your environment has no network access, run the command on a machine with internet and copy the generated CSV into this repo.
