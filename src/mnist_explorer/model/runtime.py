"""Reusable model lifecycle and inference helpers for the UI.

The UI should focus on "display logic", not TensorFlow setup details.
This module hides model-loading/training/probing so the UI class stays
shorter and easier to read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

from mnist_explorer.model.basic_nn import build_model, load_data


def load_or_train_model(model_path: Path) -> keras.Model:
    """Load a saved model from disk, or train a quick fallback model.

    Why this exists:
    - New users may run the UI before training anything.
    - Instead of failing, we train a small baseline model automatically.
    """
    # Fast path: if a trained model already exists, use it.
    if model_path.exists():
        return keras.models.load_model(model_path)

    # Slow path: bootstrap a model from scratch.
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    # Keep epochs low so first-time startup is still reasonable.
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=3,
        batch_size=128,
        verbose=2,
    )

    # Print a quick quality check so users know what they are working with.
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(
        "UI bootstrap training complete - "
        f"test loss: {loss:.4f}, test accuracy: {accuracy:.4f}"
    )

    # Save the trained model so future launches can skip bootstrap training.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    return model


def ensure_model_callable(model: keras.Model) -> None:
    """Force model graph/tensors to exist before building probe views.

    Keras Sequential models can exist in an "unbuilt" state until first call.
    The UI needs fully materialized tensors for intermediate-layer probing.
    """
    # Provide expected input shape if model has not been built yet.
    if not model.built:
        model.build((None, 784))
    # Run a dummy inference pass to finalize graph/tensor wiring.
    _ = model(np.zeros((1, 784), dtype=np.float32), training=False)


def build_probe_model(model: keras.Model) -> keras.Model:
    """Create a probe model that returns internal activations and final output.

    Output order:
    1) Dense layer 1 activations
    2) Dropout output (in inference mode dropout is pass-through)
    3) Dense layer 2 activations
    4) Final softmax probabilities
    """
    return keras.Model(
        inputs=model.inputs,
        outputs=[
            model.layers[0].output,
            model.layers[1].output,
            model.layers[2].output,
            model.layers[3].output,
        ],
    )


def extract_dense_weights(model: keras.Model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract dense-layer weights once and reuse them.

    Contributor analysis repeatedly computes terms like:
    activation * weight
    Caching these arrays avoids repeatedly asking Keras for the same weights.
    """
    w_dense1, b_dense1 = model.layers[0].get_weights()
    w_dense2, b_dense2 = model.layers[2].get_weights()
    w_out, b_out = model.layers[3].get_weights()
    return w_dense1, b_dense1, w_dense2, b_dense2, w_out, b_out


def run_probe_prediction(
    probe_model: keras.Model,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one forward pass and return flattened stage outputs.

    The probe model returns batches (shape starts with batch dimension).
    UI code works with single examples, so we return index 0 from each output.
    """
    # Calling the model directly avoids some overhead from Model.predict
    # during frequent live-draw updates.
    dense1, dropout_out, dense2, probs = probe_model(
        np.expand_dims(x, axis=0),
        training=False,
    )
    return dense1.numpy()[0], dropout_out.numpy()[0], dense2.numpy()[0], probs.numpy()[0]
