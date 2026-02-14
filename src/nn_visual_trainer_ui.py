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
from tkinter import ttk

import numpy as np

from basic_nn import MODEL_PATH, load_data
from ui_constants import (
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
    LIVE_DRAW_PREDICT_DEBOUNCE_MS,
    THINK_STAGE_LABELS,
    THINK_STEP_MS,
    WINDOW_MIN_SIZE,
    WINDOW_SIZE,
)
from ui_model import (
    build_probe_model,
    ensure_model_callable,
    extract_dense_weights,
    load_or_train_model,
    run_probe_prediction,
)
from ui_preprocessing import preprocess_drawn_digit
from ui_render_utils import format_top_contribs, layer_y_positions, mix_with_white, pick_indices


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

        # Animation state:
        # _thinking_job stores the active "after(...)" callback id.
        # _thinking_context stores values needed across animation frames.
        self._thinking_job: str | None = None
        self._thinking_context: dict[str, object] | None = None
        # Debounced callback id for real-time prediction while drawing.
        self._live_predict_job: str | None = None

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
        """Create input visualization panels: sample, drawing, preview."""
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
        """Handle mouse draw events and paint into the 28x28 buffer."""
        # Convert from screen pixels to 28x28 logical coordinates.
        cell_size = DRAW_CANVAS_SIZE / DRAW_GRID_SIZE
        col = int(event.x / cell_size)
        row = int(event.y / cell_size)
        self._paint_to_draw_buffer(row, col)
        self._draw_digit_canvas()
        self._schedule_live_draw_prediction()

    def _schedule_live_draw_prediction(self) -> None:
        """Debounce live inference so prediction updates feel real-time.

        We delay by a short window to avoid running model inference for every
        mouse event while still keeping the UI responsive.
        """
        if self._live_predict_job is not None:
            self.root.after_cancel(self._live_predict_job)
            self._live_predict_job = None

        # While user is drawing, stop any staged thinking animation from a
        # previous run so visual state stays consistent with live updates.
        self._cancel_thinking_animation()
        self._live_predict_job = self.root.after(
            LIVE_DRAW_PREDICT_DEBOUNCE_MS,
            self._run_live_draw_prediction,
        )

    def _run_live_draw_prediction(self) -> None:
        """Run inference against current drawing and update UI immediately."""
        self._live_predict_job = None

        x = preprocess_drawn_digit(self.draw_buffer)
        if float(np.max(x)) <= 0.0:
            self.truth_var.set("Ground Truth: N/A (drawn input)")
            self.prediction_var.set("Prediction: -")
            self._draw_model_input_preview(x.reshape(28, 28))
            return

        dense1, dropout_out, dense2, probs = run_probe_prediction(self.probe_model, x)
        y_pred = int(np.argmax(probs))
        confidence = float(probs[y_pred])

        self.truth_var.set("Ground Truth: N/A (drawn input)")
        self.prediction_var.set(
            f"Live prediction: {y_pred}  (confidence {confidence * 100:.2f}%)"
        )

        image_2d = x.reshape(28, 28)
        self._draw_input_image(image_2d)
        self._draw_model_input_preview(image_2d)
        # Keep full network visible and draw final-stage path for live updates.
        self._draw_decision_stages(
            x,
            dense1,
            dropout_out,
            dense2,
            probs,
            y_true=None,
            y_pred=y_pred,
            thinking_stage=4,
        )
        self._render_contributor_text(x, dense1, dropout_out, dense2, probs, y_pred)

    def _paint_to_draw_buffer(self, row: int, col: int) -> None:
        """Paint a soft brush stamp centered at (row, col)."""
        if row < 0 or row >= DRAW_GRID_SIZE or col < 0 or col >= DRAW_GRID_SIZE:
            return

        # Apply the brush kernel around target point for smoother lines.
        for dr, dc, strength in DRAW_BRUSH:
            rr = row + dr
            cc = col + dc
            if 0 <= rr < DRAW_GRID_SIZE and 0 <= cc < DRAW_GRID_SIZE:
                self.draw_buffer[rr, cc] = min(1.0, self.draw_buffer[rr, cc] + strength)

    def _draw_digit_canvas(self) -> None:
        """Render current draw_buffer to the visible draw canvas."""
        self._draw_pixel_grid(self.draw_canvas, self.draw_buffer, margin=0, size=DRAW_CANVAS_SIZE)

    def clear_drawing(self) -> None:
        """Clear drawing buffer and stop active thinking animation."""
        self._cancel_thinking_animation()
        if self._live_predict_job is not None:
            self.root.after_cancel(self._live_predict_job)
            self._live_predict_job = None
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
    # Inference + animation
    # ------------------------------
    def _run_inference_and_render(self, x: np.ndarray, y_true: int | None, input_label: str) -> None:
        """Run model forward pass, update labels, then start thinking animation."""
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
        """Stop any scheduled animation callback and clear context."""
        if self._thinking_job is not None:
            self.root.after_cancel(self._thinking_job)
            self._thinking_job = None
        self._thinking_context = None

    def _set_contributor_text(self, text: str) -> None:
        """Replace contents of the read-only contributor text box."""
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
        """Initialize staged animation context and start from stage 0."""
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

        # During animation we temporarily show a placeholder message.
        self._set_contributor_text("Thinking... contribution details appear after final output.")
        self._advance_thinking_stage(0)

    def _advance_thinking_stage(self, stage_idx: int) -> None:
        """Advance visualization by one stage until final output is reached."""
        ctx = self._thinking_context
        if ctx is None:
            return

        # Clamp stage to valid range to avoid out-of-bounds issues.
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

        # Schedule next stage if not finished yet.
        if stage < final_stage:
            self._thinking_job = self.root.after(
                THINK_STEP_MS,
                lambda: self._advance_thinking_stage(stage + 1),
            )
            return

        # Final stage reached: show final status + detailed contributor text.
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
        """Draw a 28x28 grayscale image as a scaled pixel grid on a canvas."""
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

        # Draw each logical pixel as a rectangle.
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
        """Render main input preview (larger panel)."""
        self._draw_pixel_grid(self.input_canvas, image_2d, margin=12, size=336)

    def _draw_model_input_preview(self, image_2d: np.ndarray) -> None:
        """Render compact model-input preview panel."""
        self._draw_pixel_grid(self.model_input_canvas, image_2d, margin=8, size=180)

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
        """Draw network visualization and animate only the strongest path.

        Important behavior:
        - Full network stays visible the whole time.
        - Black "strongest path" line grows stage-by-stage to imply thinking.
        """
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

        # Compute x positions for 5 conceptual layers.
        left = 50
        right_network = int(width * 0.73)
        network_top = 88
        network_bottom = height - 56
        x_positions = np.linspace(left, right_network, num=5)

        # Each tuple describes: (label, total_neurons, shown_neurons, values, color).
        layer_defs = [
            ("Input", 784, 28, x_input, "#2563eb"),
            ("Dense 1", 128, 24, dense1, "#059669"),
            ("Dropout", 128, 24, dropout_out, "#d97706"),
            ("Dense 2", 64, 18, dense2, "#7c3aed"),
            ("Output", 10, 10, probs, "#dc2626"),
        ]

        # thinking_segments determines how many black links are currently visible.
        thinking_stage = max(0, min(thinking_stage, len(layer_defs) - 1))
        thinking_segments = min(thinking_stage, len(layer_defs) - 1)

        # Pre-sample neuron indices so large layers remain readable on screen.
        sampled_indices: list[np.ndarray] = []
        sampled_values: list[np.ndarray] = []
        y_positions: list[np.ndarray] = []

        for i, (_name, total_n, shown_n, values, _color) in enumerate(layer_defs):
            idx = pick_indices(total_n, shown_n)
            vals = values[idx]
            ys = layer_y_positions(network_top, network_bottom, len(idx))
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

        # Draw a sparse "background mesh" of gray edges for context.
        for li in range(len(layer_defs) - 1):
            x1 = float(x_positions[li])
            x2 = float(x_positions[li + 1])
            ys1 = y_positions[li]
            ys2 = y_positions[li + 1]
            edge_i = pick_indices(len(ys1), min(8, len(ys1)))
            edge_j = pick_indices(len(ys2), min(8, len(ys2)))
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

        # Draw strongest path (black) progressively to show "thinking".
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

        # Draw sampled neuron circles, colored by activation intensity.
        for i, (name, total_n, _shown_n, _values, base_color) in enumerate(layer_defs):
            x = float(x_positions[i])
            vals = sampled_values[i]
            ys = y_positions[i]
            vmax = max(float(np.max(vals)), 1e-8)
            radius = 3.6 if len(ys) <= 18 else 3.0

            for vi, y in enumerate(ys):
                intensity = float(vals[vi]) / vmax
                fill = mix_with_white(base_color, 0.20 + (0.80 * intensity))
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

        # Right panel: final class probabilities (softmax output).
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
        # Hidden2 -> output contribution for predicted class.
        out_contrib = dense2 * self.w_out[:, y_pred]

        # Find strongest hidden neurons and trace what drove them.
        top_h2 = int(np.argmax(dense2))
        h1_to_h2_contrib = dropout_out * self.w_dense2[:, top_h2]

        top_h1 = int(np.argmax(dense1))
        input_to_h1_contrib = x * self.w_dense1[:, top_h1]

        # Top-3 probability summary is easier to read than all 10 values.
        sorted_probs = np.argsort(probs)[::-1]
        top3 = ", ".join([f"{int(i)}={float(probs[i]) * 100:.2f}%" for i in sorted_probs[:3]])

        blocks = [
            "Prediction Summary",
            f"  Predicted class: {y_pred}",
            f"  Top probabilities: {top3}",
            "",
            "A) Hidden2 -> Output contributors for predicted class",
            "  contribution_i = a2[i] * W_out[i, pred_class]",
            format_top_contribs(out_contrib, prefix="h2_"),
            "",
            "B) Hidden1 -> strongest Hidden2 neuron contributors",
            f"  target strongest hidden2 neuron: h2_{top_h2}",
            "  contribution_j = a1_dropout[j] * W_dense2[j, h2_target]",
            format_top_contribs(h1_to_h2_contrib, prefix="h1_"),
            "",
            "C) Input pixel -> strongest Hidden1 neuron contributors",
            f"  target strongest hidden1 neuron: h1_{top_h1}",
            "  contribution_p = x[p] * W_dense1[p, h1_target]",
            format_top_contribs(input_to_h1_contrib, prefix="px_"),
        ]

        self._set_contributor_text("\n".join(blocks))


def main() -> None:
    root = tk.Tk()
    NNTrainingToolUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
