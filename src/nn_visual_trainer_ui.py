from __future__ import annotations

"""Interactive MNIST explanation UI with staged 'thinking' animation.

The UI keeps the same core features:
- dataset sample inference
- drawn digit inference with preprocessing
- neuron-style stage visualization
- contributor text panel

Additionally, the visualizer simulates reasoning by animating the strongest
decision path (black line) layer-by-layer while keeping the full network
diagram visible for context.
"""

import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import tensorflow as tf
from tensorflow import keras

from basic_nn import MODEL_PATH, build_model, load_data


# ------------------------------
# Configuration
# ------------------------------
DRAW_GRID_SIZE = 28
DRAW_CANVAS_SIZE = 280

WINDOW_SIZE = "1500x980"
WINDOW_MIN_SIZE = (1300, 860)

COLOR_BG = "#f3f7fb"
COLOR_CARD = "#ffffff"
COLOR_INK = "#0f172a"
COLOR_SUB = "#475569"
COLOR_EDGE = "#cbd5e1"

COLOR_STATUS_INFO_BG = "#e0f2fe"
COLOR_STATUS_INFO_FG = "#0c4a6e"
COLOR_STATUS_WARN_BG = "#fff7ed"
COLOR_STATUS_WARN_FG = "#9a3412"

COLOR_NEUTRAL_BTN = "#e5e7eb"
COLOR_NEUTRAL_BTN_HOVER = "#d1d5db"
COLOR_NEUTRAL_BTN_PRESS = "#c7ccd4"

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

THINK_STEP_MS = 320
THINK_STAGE_LABELS = [
    "Reading the input pixels",
    "Activating hidden layer 1",
    "Applying dropout path (inference mode)",
    "Activating hidden layer 2",
    "Computing output probabilities",
]


# ------------------------------
# Model bootstrap
# ------------------------------
def load_or_train_model(model_path: Path) -> keras.Model:
    """Load saved model or train a quick bootstrap model if missing."""
    if model_path.exists():
        return keras.models.load_model(model_path)

    x_train, y_train, x_test, y_test = load_data()
    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=3,
        batch_size=128,
        verbose=2,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(
        "UI bootstrap training complete - "
        f"test loss: {loss:.4f}, test accuracy: {accuracy:.4f}"
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    return model


class NNTrainingToolUI:
    """Main application class for model visualization and interaction."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MNIST Neural Network Decision Explorer")
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*WINDOW_MIN_SIZE)
        self.root.configure(bg=COLOR_BG)

        self.status_var = tk.StringVar(value="Loading data and model...")
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.truth_var = tk.StringVar(value="Ground Truth: -")

        self.draw_buffer = np.zeros((DRAW_GRID_SIZE, DRAW_GRID_SIZE), dtype=np.float32)

        self._thinking_job: str | None = None
        self._thinking_context: dict[str, object] | None = None

        self._configure_styles()
        self._build_layout()
        self._bind_shortcuts()

        self.x_train, self.y_train, self.x_test, self.y_test = load_data()
        self.model = load_or_train_model(MODEL_PATH)
        self._ensure_model_callable()

        self.probe_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.layers[0].output,
                self.model.layers[1].output,
                self.model.layers[2].output,
                self.model.layers[3].output,
            ],
        )

        self.max_index = len(self.x_test) - 1
        self.index_var.set(0)
        self.index_spin.configure(to=self.max_index)

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
        self.root.bind("<Return>", lambda _e: self.update_view_from_dataset())
        self.root.bind("<KP_Enter>", lambda _e: self.update_view_from_dataset())
        self.root.bind("r", lambda _e: self.pick_random_sample())
        self.root.bind("R", lambda _e: self.pick_random_sample())
        self.root.bind("d", lambda _e: self.update_view_from_drawn())
        self.root.bind("D", lambda _e: self.update_view_from_drawn())
        self.root.bind("c", lambda _e: self.clear_drawing())
        self.root.bind("C", lambda _e: self.clear_drawing())
        self.root.bind("<Escape>", lambda _e: self.clear_drawing())

    def _ensure_model_callable(self) -> None:
        if not self.model.built:
            self.model.build((None, 784))
        _ = self.model(np.zeros((1, 784), dtype=np.float32), training=False)

    def _cache_layer_weights(self) -> None:
        self.w_dense1, self.b_dense1 = self.model.layers[0].get_weights()
        self.w_dense2, self.b_dense2 = self.model.layers[2].get_weights()
        self.w_out, self.b_out = self.model.layers[3].get_weights()

    def _set_status(self, message: str, level: str = "info") -> None:
        self.status_var.set(message)
        if level == "warn":
            self.status_label.configure(bg=COLOR_STATUS_WARN_BG, fg=COLOR_STATUS_WARN_FG)
        else:
            self.status_label.configure(bg=COLOR_STATUS_INFO_BG, fg=COLOR_STATUS_INFO_FG)

    # ------------------------------
    # Layout
    # ------------------------------
    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=14, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        self._build_header(outer)
        self._build_controls(outer)
        self._build_main_area(outer)
        self._build_status(outer)

    def _build_header(self, parent: ttk.Frame) -> None:
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
            text="Run Drawn Digit",
            command=self.update_view_from_drawn,
            style="Neutral.TButton",
        ).grid(row=0, column=4, padx=(0, 8), pady=4, sticky="w")

        ttk.Button(
            controls,
            text="Clear Drawn Digit",
            command=self.clear_drawing,
            style="Neutral.TButton",
        ).grid(row=0, column=5, padx=(0, 8), pady=4, sticky="w")

        ttk.Label(
            controls,
            text=(
                "Shortcuts: Enter=run sample, R=random sample, "
                "D=run drawn, C/Esc=clear drawing."
            ),
            style="Body.TLabel",
        ).grid(row=1, column=0, columnspan=7, pady=(6, 0), sticky="w")

    def _build_main_area(self, parent: ttk.Frame) -> None:
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
        sample_frame = ttk.LabelFrame(
            container,
            text="Current Input (28x28)",
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

        ttk.Label(
            draw_frame,
            text=(
                "Draw in white on black. Use top controls (or D key) to run "
                "inference. Clear with top control, C key, or Esc."
            ),
            style="Body.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        preview_frame = ttk.LabelFrame(
            container,
            text="Model Input Preview (28x28 after preprocessing)",
            padding=8,
            style="Card.TLabelframe",
        )
        preview_frame.pack(fill="x", pady=(10, 0))

        self.model_input_canvas = tk.Canvas(
            preview_frame,
            width=196,
            height=196,
            bg="#111827",
            highlightthickness=1,
            highlightbackground=COLOR_EDGE,
        )
        self.model_input_canvas.pack()

        ttk.Label(
            preview_frame,
            text="This is exactly what the model receives for prediction.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(8, 0))

    # ------------------------------
    # Input interactions
    # ------------------------------
    def _on_draw(self, event: tk.Event) -> None:
        cell_size = DRAW_CANVAS_SIZE / DRAW_GRID_SIZE
        col = int(event.x / cell_size)
        row = int(event.y / cell_size)
        self._paint_to_draw_buffer(row, col)
        self._draw_digit_canvas()

    def _paint_to_draw_buffer(self, row: int, col: int) -> None:
        if row < 0 or row >= DRAW_GRID_SIZE or col < 0 or col >= DRAW_GRID_SIZE:
            return

        for dr, dc, strength in DRAW_BRUSH:
            rr = row + dr
            cc = col + dc
            if 0 <= rr < DRAW_GRID_SIZE and 0 <= cc < DRAW_GRID_SIZE:
                self.draw_buffer[rr, cc] = min(1.0, self.draw_buffer[rr, cc] + strength)

    def _draw_digit_canvas(self) -> None:
        self._draw_pixel_grid(self.draw_canvas, self.draw_buffer, margin=0, size=DRAW_CANVAS_SIZE)

    def clear_drawing(self) -> None:
        self._cancel_thinking_animation()
        self.draw_buffer.fill(0.0)
        self._draw_digit_canvas()
        self._set_status("Drawing cleared.", level="info")

    def pick_random_sample(self) -> None:
        self.index_var.set(random.randint(0, self.max_index))
        self.update_view_from_dataset()

    # ------------------------------
    # Entry points
    # ------------------------------
    def update_view_from_dataset(self) -> None:
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
        x = self._preprocess_drawn_digit(self.draw_buffer)
        if float(np.max(x)) <= 0.0:
            self._set_status("Draw a digit first, then run inference.", level="warn")
            return

        self._run_inference_and_render(
            x=x,
            y_true=None,
            input_label="Drawn input (preprocessed to MNIST style)",
        )

    # ------------------------------
    # Preprocessing
    # ------------------------------
    def _shift_image(self, image: np.ndarray, shift_r: int, shift_c: int) -> np.ndarray:
        shifted = np.zeros_like(image)
        h, w = image.shape

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

    def _preprocess_drawn_digit(self, draw_img: np.ndarray) -> np.ndarray:
        """Normalize user drawing into a model-ready 784-length vector."""
        img = draw_img.astype(np.float32).copy()
        if float(np.max(img)) <= 0.0:
            return img.reshape(-1)

        mask = img > 0.10
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return np.zeros((28 * 28,), dtype=np.float32)

        crop = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        crop /= max(float(np.max(crop)), 1e-8)

        h, w = crop.shape
        scale = 20.0 / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = tf.image.resize(
            crop[..., np.newaxis],
            size=(new_h, new_w),
            method="bilinear",
        ).numpy()[..., 0]

        canvas = np.zeros((28, 28), dtype=np.float32)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized

        mass = float(np.sum(canvas))
        if mass > 1e-8:
            rr, cc = np.indices(canvas.shape)
            cy = float(np.sum(rr * canvas) / mass)
            cx = float(np.sum(cc * canvas) / mass)
            shift_r = int(round(13.5 - cy))
            shift_c = int(round(13.5 - cx))
            canvas = self._shift_image(canvas, shift_r, shift_c)

        return np.clip(canvas, 0.0, 1.0).astype(np.float32).reshape(-1)

    # ------------------------------
    # Inference + animation
    # ------------------------------
    def _run_inference_and_render(self, x: np.ndarray, y_true: int | None, input_label: str) -> None:
        dense1, dropout_out, dense2, probs = self.probe_model.predict(
            np.expand_dims(x, axis=0),
            verbose=0,
        )

        dense1 = dense1[0]
        dropout_out = dropout_out[0]
        dense2 = dense2[0]
        probs = probs[0]

        y_pred = int(np.argmax(probs))
        confidence = float(probs[y_pred])

        self.truth_var.set(
            "Ground Truth: N/A (drawn input)" if y_true is None else f"Ground Truth: {y_true}"
        )
        self.prediction_var.set(f"Prediction: {y_pred}  (confidence {confidence * 100:.2f}%)")

        image_2d = x.reshape(28, 28)
        self._draw_input_image(image_2d)
        self._draw_model_input_preview(image_2d)

        self._start_thinking_animation(
            x=x,
            dense1=dense1,
            dropout_out=dropout_out,
            dense2=dense2,
            probs=probs,
            y_true=y_true,
            y_pred=y_pred,
            confidence=confidence,
            input_label=input_label,
        )

    def _cancel_thinking_animation(self) -> None:
        if self._thinking_job is not None:
            self.root.after_cancel(self._thinking_job)
            self._thinking_job = None
        self._thinking_context = None

    def _set_contributor_text(self, text: str) -> None:
        self.contrib_text.configure(state="normal")
        self.contrib_text.delete("1.0", "end")
        self.contrib_text.insert("1.0", text)
        self.contrib_text.configure(state="disabled")

    def _start_thinking_animation(
        self,
        x: np.ndarray,
        dense1: np.ndarray,
        dropout_out: np.ndarray,
        dense2: np.ndarray,
        probs: np.ndarray,
        y_true: int | None,
        y_pred: int,
        confidence: float,
        input_label: str,
    ) -> None:
        self._cancel_thinking_animation()

        self._thinking_context = {
            "x": x,
            "dense1": dense1,
            "dropout_out": dropout_out,
            "dense2": dense2,
            "probs": probs,
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidence,
            "input_label": input_label,
        }

        self._set_contributor_text("Thinking... contribution details appear after final output.")
        self._advance_thinking_stage(0)

    def _advance_thinking_stage(self, stage_idx: int) -> None:
        ctx = self._thinking_context
        if ctx is None:
            return

        final_stage = len(THINK_STAGE_LABELS) - 1
        stage = min(stage_idx, final_stage)

        self._draw_decision_stages(
            ctx["x"],  # type: ignore[arg-type]
            ctx["dense1"],  # type: ignore[arg-type]
            ctx["dropout_out"],  # type: ignore[arg-type]
            ctx["dense2"],  # type: ignore[arg-type]
            ctx["probs"],  # type: ignore[arg-type]
            ctx["y_true"],  # type: ignore[arg-type]
            ctx["y_pred"],  # type: ignore[arg-type]
            thinking_stage=stage,
        )

        self._set_status(
            f"Thinking {stage + 1}/{final_stage + 1}: {THINK_STAGE_LABELS[stage]}",
            level="info",
        )

        if stage < final_stage:
            self._thinking_job = self.root.after(
                THINK_STEP_MS,
                lambda: self._advance_thinking_stage(stage + 1),
            )
            return

        # Final stage reached.
        y_pred = int(ctx["y_pred"])
        confidence = float(ctx["confidence"])
        input_label = str(ctx["input_label"])
        self._set_status(
            f"{input_label} processed. Final decision: {y_pred} "
            f"(confidence {confidence * 100:.2f}%).",
            level="info",
        )

        self._render_contributor_text(
            ctx["x"],  # type: ignore[arg-type]
            ctx["dense1"],  # type: ignore[arg-type]
            ctx["dropout_out"],  # type: ignore[arg-type]
            ctx["dense2"],  # type: ignore[arg-type]
            ctx["probs"],  # type: ignore[arg-type]
            y_pred,
        )
        self._thinking_job = None

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
        canvas.delete("all")
        cell = size / 28

        canvas.create_rectangle(
            margin - 1,
            margin - 1,
            margin + size + 1,
            margin + size + 1,
            outline="#334155",
            width=2,
        )

        for r in range(28):
            for c in range(28):
                value = float(image_2d[r, c])
                gray = int(np.clip(value, 0.0, 1.0) * 255)
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                x0 = margin + c * cell
                y0 = margin + r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

    def _draw_input_image(self, image_2d: np.ndarray) -> None:
        self._draw_pixel_grid(self.input_canvas, image_2d, margin=12, size=336)

    def _draw_model_input_preview(self, image_2d: np.ndarray) -> None:
        self._draw_pixel_grid(self.model_input_canvas, image_2d, margin=8, size=180)

    def _pick_indices(self, n: int, k: int) -> np.ndarray:
        if k >= n:
            return np.arange(n, dtype=np.int32)
        if k <= 1:
            return np.array([n // 2], dtype=np.int32)
        return np.unique(np.linspace(0, n - 1, num=k, dtype=np.int32))

    def _mix_with_white(self, hex_color: str, intensity: float) -> str:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        rr = int((255 * (1.0 - intensity)) + (r * intensity))
        gg = int((255 * (1.0 - intensity)) + (g * intensity))
        bb = int((255 * (1.0 - intensity)) + (b * intensity))
        return f"#{rr:02x}{gg:02x}{bb:02x}"

    def _layer_y_positions(self, top: float, bottom: float, count: int) -> np.ndarray:
        if count <= 1:
            return np.array([(top + bottom) / 2.0], dtype=np.float32)
        return np.linspace(top, bottom, num=count, dtype=np.float32)

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
        thinking_stage: int = 4,
    ) -> None:
        """Draw full network view, with staged black-path thinking animation."""
        canvas = self.stages_canvas
        canvas.delete("all")
        canvas.update_idletasks()

        width = max(canvas.winfo_width(), 1000)
        height = max(canvas.winfo_height(), 540)

        canvas.create_rectangle(0, 0, width, height, fill="#f8fafc", outline="")
        canvas.create_rectangle(0, 0, width, 52, fill="#eef2ff", outline="")
        canvas.create_text(
            14,
            16,
            text="Neuron-Level Decision Flow",
            anchor="nw",
            font=("Helvetica", 14, "bold"),
            fill="#0f172a",
        )
        canvas.create_text(
            14,
            34,
            text="Input -> Dense(128) -> Dropout(0.2) -> Dense(64) -> Softmax(10)",
            anchor="nw",
            font=("Helvetica", 10),
            fill="#334155",
        )

        left = 50
        right_network = int(width * 0.73)
        network_top = 88
        network_bottom = height - 56
        x_positions = np.linspace(left, right_network, num=5)

        layer_defs = [
            ("Input", 784, 28, x_input, "#2563eb"),
            ("Dense 1", 128, 24, dense1, "#059669"),
            ("Dropout", 128, 24, dropout_out, "#d97706"),
            ("Dense 2", 64, 18, dense2, "#7c3aed"),
            ("Output", 10, 10, probs, "#dc2626"),
        ]

        thinking_stage = max(0, min(thinking_stage, len(layer_defs) - 1))
        thinking_segments = min(thinking_stage, len(layer_defs) - 1)

        sampled_indices: list[np.ndarray] = []
        sampled_values: list[np.ndarray] = []
        y_positions: list[np.ndarray] = []

        for i, (_name, total_n, shown_n, values, _color) in enumerate(layer_defs):
            idx = self._pick_indices(total_n, shown_n)
            vals = values[idx]
            ys = self._layer_y_positions(network_top, network_bottom, len(idx))
            sampled_indices.append(idx)
            sampled_values.append(vals)
            y_positions.append(ys)

            x = float(x_positions[i])
            frame_pad = 12
            canvas.create_rectangle(
                x - frame_pad,
                network_top - 14,
                x + frame_pad,
                network_bottom + 14,
                fill="",
                outline="#cbd5e1",
                width=1,
            )

        # Representative inter-layer edges (always visible for full context).
        for li in range(len(layer_defs) - 1):
            x1 = float(x_positions[li])
            x2 = float(x_positions[li + 1])
            ys1 = y_positions[li]
            ys2 = y_positions[li + 1]
            edge_i = self._pick_indices(len(ys1), min(8, len(ys1)))
            edge_j = self._pick_indices(len(ys2), min(8, len(ys2)))
            for i in edge_i:
                for j in edge_j:
                    canvas.create_line(
                        x1,
                        float(ys1[i]),
                        x2,
                        float(ys2[j]),
                        fill="#cbd5e1",
                        width=1,
                    )

        # Strongest activation path, revealed gradually as the "thinking" signal.
        strongest_full = [
            int(np.argmax(x_input)),
            int(np.argmax(dense1)),
            int(np.argmax(dropout_out)),
            int(np.argmax(dense2)),
            y_pred,
        ]
        for li in range(len(layer_defs) - 1):
            if li >= thinking_segments:
                continue
            i_idx = int(np.argmin(np.abs(sampled_indices[li] - strongest_full[li])))
            j_idx = int(np.argmin(np.abs(sampled_indices[li + 1] - strongest_full[li + 1])))
            canvas.create_line(
                float(x_positions[li]),
                float(y_positions[li][i_idx]),
                float(x_positions[li + 1]),
                float(y_positions[li + 1][j_idx]),
                fill="#0f172a",
                width=2,
            )

        # Neurons.
        for i, (name, total_n, _shown_n, _values, base_color) in enumerate(layer_defs):
            x = float(x_positions[i])
            vals = sampled_values[i]
            ys = y_positions[i]
            vmax = max(float(np.max(vals)), 1e-8)
            radius = 3.6 if len(ys) <= 18 else 3.0

            for vi, y in enumerate(ys):
                intensity = float(vals[vi]) / vmax
                fill = self._mix_with_white(base_color, 0.20 + (0.80 * intensity))
                outline = base_color
                canvas.create_oval(
                    x - radius,
                    float(y) - radius,
                    x + radius,
                    float(y) + radius,
                    fill=fill,
                    outline=outline,
                    width=1,
                )

            canvas.create_text(
                x,
                network_bottom + 28,
                text=name,
                anchor="n",
                font=("Helvetica", 10, "bold"),
                fill="#0f172a",
            )
            canvas.create_text(
                x,
                network_bottom + 44,
                text=f"{total_n} neurons",
                anchor="n",
                font=("Helvetica", 9),
                fill="#475569",
            )

        # Probability panel.
        bar_x0 = right_network + 40
        bar_x1 = width - 28
        canvas.create_rectangle(
            bar_x0 - 12,
            network_top - 18,
            bar_x1 + 8,
            network_bottom + 12,
            fill="#ffffff",
            outline="#cbd5e1",
            width=1,
        )
        panel_title = "Softmax Probabilities"
        canvas.create_text(
            bar_x0 - 2,
            network_top - 10,
            text=panel_title,
            anchor="w",
            font=("Helvetica", 11, "bold"),
            fill="#0f172a",
        )

        bar_h = (network_bottom - network_top - 40) / 10.0
        for digit in range(10):
            p = float(probs[digit])
            y0 = network_top + 12 + digit * bar_h
            y1 = y0 + (bar_h - 4)
            canvas.create_rectangle(bar_x0, y0, bar_x1, y1, fill="#f1f5f9", outline="#e2e8f0")

            fill_color = "#64748b"
            if digit == y_pred:
                fill_color = "#16a34a"
            elif y_true is not None and digit == y_true:
                fill_color = "#2563eb"
            marker = ""
            if y_true is not None and digit == y_true:
                marker += " true"
            if digit == y_pred:
                marker += " pred"
            pct_text = f"{p * 100:.1f}%"

            canvas.create_rectangle(
                bar_x0,
                y0,
                bar_x0 + ((bar_x1 - bar_x0) * p),
                y1,
                fill=fill_color,
                outline=fill_color,
            )
            canvas.create_text(
                bar_x0 + 4,
                (y0 + y1) / 2.0,
                text=f"{digit}{marker}",
                anchor="w",
                font=("Helvetica", 9),
                fill="#0f172a",
            )
            canvas.create_text(
                bar_x1 - 4,
                (y0 + y1) / 2.0,
                text=pct_text,
                anchor="e",
                font=("Helvetica", 9),
                fill="#0f172a",
            )

    # ------------------------------
    # Contributions
    # ------------------------------
    def _format_top_contribs(self, values: np.ndarray, prefix: str, top_n: int = 5) -> str:
        pos_idx = np.argsort(values)[-top_n:][::-1]
        neg_idx = np.argsort(values)[:top_n]
        pos_lines = [f"  + {prefix}{int(i):>3}: {float(values[i]): .5f}" for i in pos_idx]
        neg_lines = [f"  - {prefix}{int(i):>3}: {float(values[i]): .5f}" for i in neg_idx]
        return "\n".join(["Top positive contributors:", *pos_lines, "Top negative contributors:", *neg_lines])

    def _render_contributor_text(
        self,
        x: np.ndarray,
        dense1: np.ndarray,
        dropout_out: np.ndarray,
        dense2: np.ndarray,
        probs: np.ndarray,
        y_pred: int,
    ) -> None:
        out_contrib = dense2 * self.w_out[:, y_pred]

        top_h2 = int(np.argmax(dense2))
        h1_to_h2_contrib = dropout_out * self.w_dense2[:, top_h2]

        top_h1 = int(np.argmax(dense1))
        input_to_h1_contrib = x * self.w_dense1[:, top_h1]

        sorted_probs = np.argsort(probs)[::-1]
        top3 = ", ".join([f"{int(i)}={float(probs[i]) * 100:.2f}%" for i in sorted_probs[:3]])

        blocks = [
            "Prediction Summary",
            f"  Predicted class: {y_pred}",
            f"  Top probabilities: {top3}",
            "",
            "A) Hidden2 -> Output contributors for predicted class",
            "  contribution_i = a2[i] * W_out[i, pred_class]",
            self._format_top_contribs(out_contrib, prefix="h2_"),
            "",
            "B) Hidden1 -> strongest Hidden2 neuron contributors",
            f"  target strongest hidden2 neuron: h2_{top_h2}",
            "  contribution_j = a1_dropout[j] * W_dense2[j, h2_target]",
            self._format_top_contribs(h1_to_h2_contrib, prefix="h1_"),
            "",
            "C) Input pixel -> strongest Hidden1 neuron contributors",
            f"  target strongest hidden1 neuron: h1_{top_h1}",
            "  contribution_p = x[p] * W_dense1[p, h1_target]",
            self._format_top_contribs(input_to_h1_contrib, prefix="px_"),
        ]

        self._set_contributor_text("\n".join(blocks))


def main() -> None:
    root = tk.Tk()
    NNTrainingToolUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
