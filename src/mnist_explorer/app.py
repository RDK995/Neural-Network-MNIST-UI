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
from tkinter import ttk

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
    COLOR_CARD,
    COLOR_EDGE,
    COLOR_INK,
    COLOR_NEUTRAL_BTN,
    COLOR_NEUTRAL_BTN_HOVER,
    COLOR_NEUTRAL_BTN_PRESS,
    COLOR_STATUS_INFO_BG,
    COLOR_STATUS_INFO_FG,
    COLOR_STATUS_WARN_BG,
    COLOR_STATUS_WARN_FG,
    COLOR_SUB,
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
from mnist_explorer.ui.preprocessing import preprocess_drawn_digit


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

        # Build visual style and layout before loading data/model so the user
        # sees a responsive window early.
        self._configure_styles()
        self._build_layout()
        self._bind_shortcuts()

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
        self.update_view_from_dataset()
        self._draw_digit_canvas()

    # ------------------------------
    # Style / setup
    # ------------------------------
    def _configure_styles(self) -> None:
        """Define ttk style rules so widgets share one visual language."""
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background=COLOR_BG)

        style.configure(
            "Title.TLabel",
            background=COLOR_BG,
            foreground=COLOR_INK,
            font=("Avenir Next", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=COLOR_BG,
            foreground=COLOR_SUB,
            font=("Avenir Next", 11),
        )
        style.configure(
            "Section.TLabel",
            background=COLOR_CARD,
            foreground=COLOR_INK,
            font=("Avenir Next", 11, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=COLOR_CARD,
            foreground=COLOR_SUB,
            font=("Avenir Next", 10),
        )

        style.configure(
            "Card.TLabelframe",
            background=COLOR_CARD,
            foreground=COLOR_INK,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=COLOR_CARD,
            foreground=COLOR_INK,
            font=("Avenir Next", 10, "bold"),
        )

        style.configure(
            "Neutral.TButton",
            background=COLOR_NEUTRAL_BTN,
            foreground="#1f2937",
            borderwidth=1,
            font=("Avenir Next", 10, "bold"),
            padding=(12, 8),
        )
        style.map(
            "Neutral.TButton",
            background=[("active", COLOR_NEUTRAL_BTN_HOVER), ("pressed", COLOR_NEUTRAL_BTN_PRESS)],
            foreground=[("disabled", "#9ca3af"), ("!disabled", "#111827")],
        )

        style.configure(
            "App.TSpinbox",
            fieldbackground="#f8fafc",
            foreground=COLOR_INK,
            padding=4,
        )

    def _bind_shortcuts(self) -> None:
        """Register keyboard shortcuts for fast interaction."""
        self.root.bind("<Return>", lambda _e: self.update_view_from_dataset())
        self.root.bind("<KP_Enter>", lambda _e: self.update_view_from_dataset())
        self.root.bind("r", lambda _e: self.pick_random_sample())
        self.root.bind("R", lambda _e: self.pick_random_sample())
        self.root.bind("d", lambda _e: self.update_view_from_drawn())
        self.root.bind("D", lambda _e: self.update_view_from_drawn())
        self.root.bind("c", lambda _e: self.clear_drawing())
        self.root.bind("C", lambda _e: self.clear_drawing())
        self.root.bind("<Escape>", lambda _e: self.clear_drawing())

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
    # Layout
    # ------------------------------
    def _build_layout(self) -> None:
        """Create top-level layout containers and major UI sections."""
        outer = ttk.Frame(self.root, padding=14, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        self._build_header(outer)
        self._build_controls(outer)
        self._build_main_area(outer)
        self._build_status(outer)

    def _build_header(self, parent: ttk.Frame) -> None:
        """Create the title/subtitle area at the top of the window."""
        header = ttk.Frame(parent, style="App.TFrame")
        header.pack(fill="x", pady=(0, 10))

        ttk.Label(
            header,
            text="MNIST Neural Network Decision Explorer",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Inspect stage-by-stage activations and contributor summaries for "
                "dataset samples or your own drawn digits."
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Create controls for dataset/sample actions and drawing actions."""
        controls = ttk.LabelFrame(
            parent,
            text="Dataset Input Controls",
            padding=12,
            style="Card.TLabelframe",
        )
        controls.pack(fill="x", pady=(0, 10))
        controls.grid_columnconfigure(6, weight=1)

        self.index_var = tk.IntVar(value=0)

        ttk.Label(controls, text="MNIST test sample index:", style="Section.TLabel").grid(
            row=0,
            column=0,
            padx=(0, 6),
            pady=4,
            sticky="w",
        )
        self.index_spin = ttk.Spinbox(
            controls,
            from_=0,
            to=9999,
            textvariable=self.index_var,
            width=10,
            style="App.TSpinbox",
        )
        self.index_spin.grid(row=0, column=1, padx=(0, 8), pady=4, sticky="w")

        ttk.Button(
            controls,
            text="Run Dataset Sample",
            command=self.update_view_from_dataset,
            style="Neutral.TButton",
        ).grid(row=0, column=2, padx=(0, 8), pady=4, sticky="w")

        ttk.Button(
            controls,
            text="Random Sample",
            command=self.pick_random_sample,
            style="Neutral.TButton",
        ).grid(row=0, column=3, padx=(0, 8), pady=4, sticky="w")

        ttk.Button(
            controls,
            text="Clear Drawn Digit",
            command=self.clear_drawing,
            style="Neutral.TButton",
        ).grid(row=0, column=4, padx=(0, 8), pady=4, sticky="w")

        ttk.Label(
            controls,
            text=(
                "Shortcuts: Enter=run sample, R=random sample, "
                "D=run drawn, C/Esc=clear drawing. Draw panel predicts in real time."
            ),
            style="Body.TLabel",
        ).grid(row=1, column=0, columnspan=7, pady=(6, 0), sticky="w")

    def _build_main_area(self, parent: ttk.Frame) -> None:
        """Split UI into left input panel and right explanation panel."""
        results = ttk.Frame(parent, style="App.TFrame")
        results.pack(fill="both", expand=True)

        left = ttk.LabelFrame(results, text="Input Sources", padding=10, style="Card.TLabelframe")
        left.pack(side="left", fill="y", padx=(0, 10))
        self._build_left_panel(left)

        right = ttk.Frame(results, style="App.TFrame")
        right.pack(side="left", fill="both", expand=True)

        decision_frame = ttk.LabelFrame(
            right,
            text="Decision Stages",
            padding=10,
            style="Card.TLabelframe",
        )
        decision_frame.pack(fill="both", expand=True)

        self.stages_canvas = tk.Canvas(
            decision_frame,
            width=1040,
            height=560,
            bg="#f4f8ff",
            highlightthickness=1,
            highlightbackground=COLOR_EDGE,
        )
        self.stages_canvas.pack(fill="both", expand=True)

        contrib_frame = ttk.LabelFrame(
            right,
            text="Top Contributing Neurons (activation x weight)",
            padding=8,
            style="Card.TLabelframe",
        )
        contrib_frame.pack(fill="both", expand=False, pady=(8, 0))

        self.contrib_text = tk.Text(
            contrib_frame,
            width=102,
            height=11,
            wrap="word",
            bg="#fcfdff",
            fg=COLOR_INK,
            font=("Menlo", 11),
            highlightthickness=1,
            highlightbackground=COLOR_EDGE,
            relief="flat",
            padx=10,
            pady=10,
        )
        contrib_scroll = ttk.Scrollbar(contrib_frame, orient="vertical", command=self.contrib_text.yview)
        self.contrib_text.configure(yscrollcommand=contrib_scroll.set)
        self.contrib_text.pack(side="left", fill="both", expand=True)
        contrib_scroll.pack(side="right", fill="y")
        self.contrib_text.configure(state="disabled")

    def _build_status(self, parent: ttk.Frame) -> None:
        """Bottom status strip used for info/warning feedback."""
        status_frame = ttk.Frame(parent, style="App.TFrame")
        status_frame.pack(fill="x", pady=(8, 0))

        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            bg=COLOR_STATUS_INFO_BG,
            fg=COLOR_STATUS_INFO_FG,
            font=("Avenir Next", 10, "bold"),
            padx=10,
            pady=8,
            anchor="w",
            relief="flat",
        )
        self.status_label.pack(fill="x", anchor="w")

    def _build_left_panel(self, container: ttk.LabelFrame) -> None:
        """Create input visualization panels for preprocessed input and drawing."""
        sample_frame = ttk.LabelFrame(
            container,
            text="Current Input (Preprocessed 28x28)",
            padding=8,
            style="Card.TLabelframe",
        )
        sample_frame.pack(fill="x")

        self.input_canvas = tk.Canvas(
            sample_frame,
            width=360,
            height=360,
            bg="#111827",
            highlightthickness=1,
            highlightbackground=COLOR_EDGE,
        )
        self.input_canvas.pack()

        ttk.Label(sample_frame, textvariable=self.truth_var, style="Section.TLabel").pack(
            anchor="w", pady=(10, 2)
        )
        ttk.Label(sample_frame, textvariable=self.prediction_var, style="Section.TLabel").pack(anchor="w")

        draw_frame = ttk.LabelFrame(
            container,
            text="Draw Your Own Digit",
            padding=8,
            style="Card.TLabelframe",
        )
        draw_frame.pack(fill="x", pady=(10, 0))

        self.draw_canvas = tk.Canvas(
            draw_frame,
            width=DRAW_CANVAS_SIZE,
            height=DRAW_CANVAS_SIZE,
            bg="#111827",
            highlightthickness=1,
            highlightbackground=COLOR_EDGE,
            cursor="crosshair",
            takefocus=1,
        )
        self.draw_canvas.pack()
        self.draw_canvas.bind("<Button-1>", self._on_draw)
        self.draw_canvas.bind("<B1-Motion>", self._on_draw)
        self.draw_canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_last_draw_cell", None))

        ttk.Label(
            draw_frame,
            text=(
                "Draw in white on black. The top input panel now shows the "
                "preprocessed model input in real time."
            ),
            style="Body.TLabel",
        ).pack(anchor="w", pady=(8, 0))

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

        dense1, dropout_out, dense2, probs = run_probe_prediction(self.probe_model, x)
        y_pred = int(np.argmax(probs))
        confidence = float(probs[y_pred])

        self.truth_var.set("Ground Truth: N/A (drawn input)")
        self.prediction_var.set(
            f"Live prediction: {y_pred}  (confidence {confidence * 100:.2f}%)"
        )

        image_2d = x.reshape(28, 28)
        # Lightweight live refresh: always keep current preprocessed input current.
        self._draw_input_image(image_2d)

        # Heavy refresh (network graph + contributor text) is throttled to
        # reduce draw lag. We still force refresh when predicted class changes.
        now = self._last_live_predict_ts
        heavy_interval_s = LIVE_DRAW_HEAVY_REFRESH_INTERVAL_MS / 1000.0
        class_changed = self._last_live_heavy_pred != y_pred
        due_for_heavy = (now - self._last_live_heavy_render_ts) >= heavy_interval_s
        if class_changed or due_for_heavy:
            self._draw_decision_stages(
                x,
                dense1,
                dropout_out,
                dense2,
                probs,
                y_true=None,
                y_pred=y_pred,
            )
            self._render_contributor_text(x, dense1, dropout_out, dense2, probs, y_pred)
            self._last_live_heavy_render_ts = now
            self._last_live_heavy_pred = y_pred

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
