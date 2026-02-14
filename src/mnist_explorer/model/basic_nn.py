import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


MODEL_PATH = Path("models/basic_nn_mnist.keras")


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keras gives us MNIST already split into:
    # - training data (used to learn)
    # - test data (used to measure final performance)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Each image pixel is originally an integer from 0 to 255.
    # We convert to float and divide by 255 so values are between 0 and 1.
    # This usually helps neural networks train faster and more stably.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # MNIST images are 28x28 (2D), but this model expects a 1D vector.
    # So each image becomes a single list of 784 numbers (28 * 28).
    # Example: (60000, 28, 28) -> (60000, 784)
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))

    # Return all four arrays so the rest of the program can use them.
    return x_train, y_train, x_test, y_test


def build_model(input_dim: int = 784, num_classes: int = 10) -> keras.Model:
    # We use a "Sequential" model, which means data flows layer-by-layer
    # from top to bottom in the order listed here.
    model = keras.Sequential(
        [
            # Input layer: tells Keras each example has 784 features.
            layers.Input(shape=(input_dim,)),

            # First hidden layer:
            # - 128 neurons
            # - ReLU activation introduces non-linearity so the model can
            #   learn complex patterns.
            layers.Dense(128, activation="relu"),

            # Dropout randomly turns off 20% of neurons during training only.
            # This is a regularization technique to reduce overfitting.
            layers.Dropout(0.2),

            # Second hidden layer with 64 neurons and ReLU.
            layers.Dense(64, activation="relu"),

            # Output layer:
            # - 10 neurons because MNIST has 10 digit classes (0-9)
            # - softmax converts outputs into class probabilities
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # "compile" picks how the model learns.
    # - optimizer="adam": a popular optimizer that adapts learning rates
    # - loss="sparse_categorical_crossentropy":
    #   good for multi-class classification when labels are integers
    # - metrics=["accuracy"]: report accuracy during training/evaluation
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Return a ready-to-train model.
    return model


def main() -> None:
    # Fix random seeds so results are more reproducible between runs.
    # (There can still be small differences depending on hardware/backend.)
    tf.random.set_seed(42)
    np.random.seed(42)

    # 1) Load and prepare data.
    x_train, y_train, x_test, y_test = load_data()

    # 2) Build the neural network architecture.
    model = build_model()

    # Print a summary so we can see layers and parameter counts.
    model.summary()

    # 3) Train the model.
    # - validation_split=0.1 means 10% of training data is held out
    #   for validation (not used for weight updates).
    # - epochs=5 means we pass through the training data 5 times.
    # - batch_size=128 means model updates happen every 128 samples.
    # - verbose=2 prints one concise line per epoch.
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        verbose=2,
    )

    # 4) Evaluate on completely separate test data.
    # This gives a better estimate of real-world performance.
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # 5) Make a few sample predictions so we can inspect model output.
    sample_preds = model.predict(x_test[:5], verbose=0)

    # The model returns probabilities for each class.
    # np.argmax picks the class with the highest probability.
    pred_labels = np.argmax(sample_preds, axis=1)
    print("Sample predictions:", pred_labels)
    print("Ground truth:", y_test[:5])

    # 6) Save the trained model so other tools (like the visual UI)
    # can load it without retraining every time.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")


# Standard Python entry point:
# this makes sure main() runs only when we execute this file directly.
if __name__ == "__main__":
    main()
