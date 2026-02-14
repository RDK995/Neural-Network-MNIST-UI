from __future__ import annotations

"""Interactive MNIST explanation UI with real-time and on-demand inference.

The UI keeps the same core features:
- dataset sample inference
- drawn digit inference with preprocessing
- neuron-style stage visualization
- contributor text panel

The visualizer renders final decision states directly to match the real-time
prediction workflow.
"""

import random
import sys
import tkinter as tk
import time
from pathlib import Path

import numpy as np

# Allow running this file directly (e.g. `python src/mnist_explorer/app.py`)
# by ensuring `src/` is on sys.path for absolute package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnist_explorer.model.basic_nn import MODEL_PATH, load_data
from mnist_explorer.model.runtime import (
    build_probe_model,
    ensure_model_callable,
    extract_dense_weights,
    load_or_train_model,
    run_probe_prediction,
)
from mnist_explorer.ui.constants import (
    COLOR_BG,
    COLOR_STATUS_INFO_BG,
    COLOR_STATUS_INFO_FG,
    COLOR_STATUS_WARN_BG,
    COLOR_STATUS_WARN_FG,
    DRAW_BRUSH,
    DRAW_CANVAS_SIZE,
    DRAW_GRID_SIZE,
    LIVE_DRAW_HEAVY_REFRESH_INTERVAL_MS,
    LIVE_DRAW_PREDICT_INTERVAL_MS,
    WINDOW_MIN_SIZE,
    WINDOW_SIZE,
)
from mnist_explorer.ui.canvas import draw_pixel_grid, paint_brush_stamp
from mnist_explorer.ui.decision_panel import build_contributor_text, render_decision_stages
from mnist_explorer.ui.layout import bind_shortcuts, build_layout, configure_styles
from mnist_explorer.ui.preprocessing import preprocess_drawn_digit
from mnist_explorer.services.live_inference import LiveInferenceResult, LiveInferenceWorker


class NNTrainingToolUI:
    """Main application class for model visualization and interaction.

    If you are new to Python/Tkinter, think of this class as:
    - the "controller" (responds to user actions),
    - the "view" builder (creates and updates widgets),
    - and the "orchestrator" that calls model utilities.
    """

    def __init__(self, root: tk.Tk) -> None:
        # Tk root is the main desktop window.
        self.root = root
        self.root.title("MNIST Neural Network Decision Explorer")
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*WINDOW_MIN_SIZE)
        self.root.configure(bg=COLOR_BG)

        # Tkinter StringVar objects let labels auto-update when values change.
        self.status_var = tk.StringVar(value="Loading data and model...")
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.truth_var = tk.StringVar(value="Ground Truth: -")

        # In-memory 28x28 drawing buffer (float values from 0.0 to 1.0).
        self.draw_buffer = np.zeros((DRAW_GRID_SIZE, DRAW_GRID_SIZE), dtype=np.float32)

        # Scheduled callback id for real-time prediction while drawing.
        self._live_predict_job: str | None = None
        # Timestamp of the last live prediction pass for throttling.
        self._last_live_predict_ts = 0.0
        # Timestamp + class cache for throttling expensive visual refreshes.
        self._last_live_heavy_render_ts = 0.0
        self._last_live_heavy_pred: int | None = None
        # Track latest painted cell to avoid redundant redraw work during drag.
        self._last_draw_cell: tuple[int, int] | None = None
        # Background worker (single responsibility: non-blocking live inference).
        self._live_worker: LiveInferenceWorker | None = None

        # Build visual style and layout before loading data/model so the user
        # sees a responsive window early.
        configure_styles(self.root)
        build_layout(self)
        bind_shortcuts(self)

        # Load MNIST data + model. This is the core inference backend.
        self.x_train, self.y_train, self.x_test, self.y_test = load_data()
        self.model = load_or_train_model(MODEL_PATH)
        ensure_model_callable(self.model)

        # Probe model returns intermediate activations used for visualization.
        self.probe_model = build_probe_model(self.model)

        self.max_index = len(self.x_test) - 1
        self.index_var.set(0)
        self.index_spin.configure(to=self.max_index)

        # Cache dense weights once for contributor analysis later.
        self._cache_layer_weights()

        self._set_status(
            "Ready. Use test samples or draw your own digit and run decision flow.",
            level="info",
        )
        self._start_live_inference_worker()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.update_view_from_dataset()
        self._draw_digit_canvas()

    def _cache_layer_weights(self) -> None:
        """Pull dense weights out of the model and store for reuse."""
        (
            self.w_dense1,
            self.b_dense1,
            self.w_dense2,
            self.b_dense2,
            self.w_out,
            self.b_out,
        ) = extract_dense_weights(self.model)

    def _set_status(self, message: str, level: str = "info") -> None:
        """Update bottom status banner text + color based on severity."""
        self.status_var.set(message)
        if level == "warn":
            self.status_label.configure(bg=COLOR_STATUS_WARN_BG, fg=COLOR_STATUS_WARN_FG)
        else:
            self.status_label.configure(bg=COLOR_STATUS_INFO_BG, fg=COLOR_STATUS_INFO_FG)

    # ------------------------------
    # Input interactions
    # ------------------------------
    def _on_draw(self, event: tk.Event) -> None:
        """Handle mouse draw events and paint into the 28x28 buffer."""
        # Convert from screen pixels to 28x28 logical coordinates.
        cell_size = DRAW_CANVAS_SIZE / DRAW_GRID_SIZE
        col = int(event.x / cell_size)
        row = int(event.y / cell_size)
        if self._last_draw_cell == (row, col):
            return
        self._last_draw_cell = (row, col)
        self._paint_to_draw_buffer(row, col)
        self._draw_digit_canvas()
        self._schedule_live_draw_prediction()

    def _schedule_live_draw_prediction(self) -> None:
        """Throttle live inference so updates happen regularly while drawing.

        Unlike pure debounce, this gives users steady refreshes even when they
        continuously draw (mouse events keep firing).
        """
        now = time.monotonic()
        interval_s = LIVE_DRAW_PREDICT_INTERVAL_MS / 1000.0
        elapsed = now - self._last_live_predict_ts

        # If enough time passed, run immediately for a snappy feel.
        if elapsed >= interval_s:
            if self._live_predict_job is not None:
                self.root.after_cancel(self._live_predict_job)
                self._live_predict_job = None
            self._run_live_draw_prediction()
            return

        # Otherwise schedule one trailing update if not already queued.
        if self._live_predict_job is None:
            delay_ms = max(1, int((interval_s - elapsed) * 1000))
            self._live_predict_job = self.root.after(delay_ms, self._run_live_draw_prediction)

    def _run_live_draw_prediction(self) -> None:
        """Run inference against current drawing and update UI immediately."""
        self._live_predict_job = None
        self._last_live_predict_ts = time.monotonic()

        x = preprocess_drawn_digit(self.draw_buffer)
        if float(np.max(x)) <= 0.0:
            self.truth_var.set("Ground Truth: N/A (drawn input)")
            self.prediction_var.set("Prediction: -")
            self._draw_input_image(x.reshape(28, 28))
            return

        self.truth_var.set("Ground Truth: N/A (drawn input)")
        self._draw_input_image(x.reshape(28, 28))

        # In production we run inference off the Tk thread.
        # Tests that construct UI shells via __new__ keep working via fallback.
        if self._submit_live_inference(x):
            return

        dense1, dropout_out, dense2, probs = run_probe_prediction(self.probe_model, x)
        self._apply_live_inference_result(x, dense1, dropout_out, dense2, probs)

    def _apply_live_inference_result(
        self,
        x: np.ndarray,
        dense1: np.ndarray,
        dropout_out: np.ndarray,
        dense2: np.ndarray,
        probs: np.ndarray,
    ) -> None:
        """Apply one live inference result on the Tk thread."""
        y_pred = int(np.argmax(probs))
        confidence = float(probs[y_pred])
        self.prediction_var.set(f"Live prediction: {y_pred}  (confidence {confidence * 100:.2f}%)")

        now = time.monotonic()
        heavy_interval_s = LIVE_DRAW_HEAVY_REFRESH_INTERVAL_MS / 1000.0
        class_changed = self._last_live_heavy_pred != y_pred
        due_for_heavy = (now - self._last_live_heavy_render_ts) >= heavy_interval_s
        if class_changed or due_for_heavy:
            self._draw_decision_stages(
                x_input=x,
                dense1=dense1,
                dropout_out=dropout_out,
                dense2=dense2,
                probs=probs,
                y_true=None,
                y_pred=y_pred,
            )
            self._render_contributor_text(x, dense1, dropout_out, dense2, probs, y_pred)
            self._last_live_heavy_render_ts = now
            self._last_live_heavy_pred = y_pred

    def _submit_live_inference(self, x: np.ndarray) -> bool:
        """Queue latest drawn input for background inference.

        We keep this wrapper method on the UI class so unit tests can still
        construct lightweight shells without fully initializing the worker.
        """
        worker = getattr(self, "_live_worker", None)
        if worker is None:
            return False
        return worker.submit(x)

    def _start_live_inference_worker(self) -> None:
        """Start asynchronous live inference support."""
        self._live_worker = LiveInferenceWorker(
            predict_fn=lambda x: run_probe_prediction(self.probe_model, x)
        )
        self._live_worker.start()
        self.root.after(16, self._poll_live_inference_results)

    def _poll_live_inference_results(self) -> None:
        """Drain queued live results and apply only the newest one.

        Polling is scheduled via Tk's timer so all UI mutations stay on the
        main thread (Tk is not thread-safe).
        """
        worker = getattr(self, "_live_worker", None)
        if worker is None:
            return
        latest: LiveInferenceResult | None = worker.poll_latest()
        if latest is not None:
            self._apply_live_inference_result(
                latest.x,
                latest.dense1,
                latest.dropout_out,
                latest.dense2,
                latest.probs,
            )
        if not worker.is_stopped():
            self.root.after(16, self._poll_live_inference_results)

    def _paint_to_draw_buffer(self, row: int, col: int) -> None:
        """Paint a soft brush stamp centered at (row, col)."""
        paint_brush_stamp(self.draw_buffer, row=row, col=col, brush=DRAW_BRUSH)

    def _draw_digit_canvas(self) -> None:
        """Render current draw_buffer to the visible draw canvas."""
        self._draw_pixel_grid(self.draw_canvas, self.draw_buffer, margin=0, size=DRAW_CANVAS_SIZE)

    def clear_drawing(self) -> None:
        """Clear drawing buffer and reset prediction state."""
        if self._live_predict_job is not None:
            self.root.after_cancel(self._live_predict_job)
            self._live_predict_job = None
        self._last_live_heavy_pred = None
        self._last_live_heavy_render_ts = 0.0
        self._last_draw_cell = None
        self.draw_buffer.fill(0.0)
        self._draw_digit_canvas()
        self.truth_var.set("Ground Truth: -")
        self.prediction_var.set("Prediction: -")
        self._set_status("Drawing cleared.", level="info")

    def _on_close(self) -> None:
        """Shutdown background worker and close the Tk window cleanly."""
        if self._live_worker is not None:
            self._live_worker.stop()
        self.root.destroy()

    def pick_random_sample(self) -> None:
        """Pick a random test index and run full visualization update."""
        self.index_var.set(random.randint(0, self.max_index))
        self.update_view_from_dataset()

    # ------------------------------
    # Entry points
    # ------------------------------
    def update_view_from_dataset(self) -> None:
        """Run inference/explanation flow using selected MNIST test sample."""
        try:
            idx = int(self.index_var.get())
        except (ValueError, tk.TclError):
            self._set_status("Invalid index. Please enter a number.", level="warn")
            return

        if idx < 0 or idx > self.max_index:
            self._set_status(f"Index out of range. Use 0 to {self.max_index}.", level="warn")
            return

        x = self.x_test[idx]
        y_true = int(self.y_test[idx])
        self._run_inference_and_render(x=x, y_true=y_true, input_label=f"Dataset sample {idx}")

    def update_view_from_drawn(self) -> None:
        """Preprocess user drawing and run inference/explanation flow."""
        if self._live_predict_job is not None:
            self.root.after_cancel(self._live_predict_job)
            self._live_predict_job = None
        x = preprocess_drawn_digit(self.draw_buffer)
        if float(np.max(x)) <= 0.0:
            self._set_status("Draw a digit first, then run inference.", level="warn")
            return

        self._run_inference_and_render(
            x=x,
            y_true=None,
            input_label="Drawn input (preprocessed to MNIST style)",
        )

    # ------------------------------
    # Inference + render
    # ------------------------------
    def _run_inference_and_render(self, x: np.ndarray, y_true: int | None, input_label: str) -> None:
        """Run model forward pass and render final decision outputs immediately."""
        dense1, dropout_out, dense2, probs = run_probe_prediction(self.probe_model, x)

        # Prediction is the class with highest probability.
        y_pred = int(np.argmax(probs))
        confidence = float(probs[y_pred])

        self.truth_var.set(
            "Ground Truth: N/A (drawn input)" if y_true is None else f"Ground Truth: {y_true}"
        )
        self.prediction_var.set(f"Prediction: {y_pred}  (confidence {confidence * 100:.2f}%)")

        # Show the actual 28x28 input the model received.
        image_2d = x.reshape(28, 28)
        self._draw_input_image(image_2d)
        self._draw_decision_stages(
            x_input=x,
            dense1=dense1,
            dropout_out=dropout_out,
            dense2=dense2,
            probs=probs,
            y_true=y_true,
            y_pred=y_pred,
        )
        self._render_contributor_text(x, dense1, dropout_out, dense2, probs, y_pred)
        self._set_status(
            f"{input_label} processed. Final decision: {y_pred} "
            f"(confidence {confidence * 100:.2f}%).",
            level="info",
        )

    def _set_contributor_text(self, text: str) -> None:
        """Replace contents of the read-only contributor text box."""
        self.contrib_text.configure(state="normal")
        self.contrib_text.delete("1.0", "end")
        self.contrib_text.insert("1.0", text)
        self.contrib_text.configure(state="disabled")

    # ------------------------------
    # Small rendering helpers
    # ------------------------------
    def _draw_pixel_grid(
        self,
        canvas: tk.Canvas,
        image_2d: np.ndarray,
        margin: int,
        size: int,
    ) -> None:
        """Draw a grayscale image as a scaled pixel grid on a canvas."""
        draw_pixel_grid(canvas=canvas, image_2d=image_2d, margin=margin, size=size)

    def _draw_input_image(self, image_2d: np.ndarray) -> None:
        """Render main input preview (larger panel)."""
        self._draw_pixel_grid(self.input_canvas, image_2d, margin=12, size=336)

    # ------------------------------
    # Decision stage visualizer
    # ------------------------------
    def _draw_decision_stages(
        self,
        x_input: np.ndarray,
        dense1: np.ndarray,
        dropout_out: np.ndarray,
        dense2: np.ndarray,
        probs: np.ndarray,
        y_true: int | None,
        y_pred: int,
    ) -> None:
        """Draw decision stages for the current inference state."""
        render_decision_stages(
            canvas=self.stages_canvas,
            x_input=x_input,
            dense1=dense1,
            dropout_out=dropout_out,
            dense2=dense2,
            probs=probs,
            y_true=y_true,
            y_pred=y_pred,
        )

    # ------------------------------
    # Contributions
    # ------------------------------
    def _render_contributor_text(
        self,
        x: np.ndarray,
        dense1: np.ndarray,
        dropout_out: np.ndarray,
        dense2: np.ndarray,
        probs: np.ndarray,
        y_pred: int,
    ) -> None:
        """Build human-readable contributor report for the prediction."""
        self._set_contributor_text(
            build_contributor_text(
                x=x,
                dense1=dense1,
                dropout_out=dropout_out,
                dense2=dense2,
                probs=probs,
                y_pred=y_pred,
                w_dense1=self.w_dense1,
                w_dense2=self.w_dense2,
                w_out=self.w_out,
            )
        )


def main() -> None:
    root = tk.Tk()
    NNTrainingToolUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
