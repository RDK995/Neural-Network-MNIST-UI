from __future__ import annotations

import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from tensorflow import keras

from basic_nn import MODEL_PATH, build_model, load_data


DRAW_GRID_SIZE = 28
DRAW_CANVAS_SIZE = 280


def load_or_train_model(model_path: Path) -> keras.Model:
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
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MNIST Neural Network Decision Explorer")
        self.root.geometry("1500x980")

        self.status_var = tk.StringVar(value="Loading data and model...")
        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.truth_var = tk.StringVar(value="Ground Truth: -")

        self.draw_buffer = np.zeros((DRAW_GRID_SIZE, DRAW_GRID_SIZE), dtype=np.float32)

        self._build_layout()

        self.x_train, self.y_train, self.x_test, self.y_test = load_data()
        self.model = load_or_train_model(MODEL_PATH)

        self.probe_model = keras.Model(
            inputs=self.model.input,
            outputs=[
                self.model.layers[0].output,  # Dense(128)
                self.model.layers[1].output,  # Dropout(0.2)
                self.model.layers[2].output,  # Dense(64)
                self.model.layers[3].output,  # Dense(10, softmax)
            ],
        )

        self.max_index = len(self.x_test) - 1
        self.index_var.set(0)
        self.index_spin.configure(to=self.max_index)

        self._cache_layer_weights()

        self.status_var.set(
            "Ready. Use test samples or draw your own digit and run decision flow."
        )
        self.update_view_from_dataset()
        self._redraw_draw_canvas()

    def _cache_layer_weights(self) -> None:
        # Dense1: input(784) -> hidden1(128)
        self.w_dense1, self.b_dense1 = self.model.layers[0].get_weights()
        # Dense2: hidden1(128) -> hidden2(64)
        self.w_dense2, self.b_dense2 = self.model.layers[2].get_weights()
        # Output: hidden2(64) -> classes(10)
        self.w_out, self.b_out = self.model.layers[3].get_weights()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))

        ttk.Label(
            header,
            text="MNIST Neural Network Decision Explorer",
            font=("Helvetica", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Training tool: inspect stage-by-stage activations, then review "
                "which neurons contributed most to the final decision."
            ),
            font=("Helvetica", 11),
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.LabelFrame(outer, text="Dataset Input Controls", padding=10)
        controls.pack(fill="x", pady=(0, 10))

        self.index_var = tk.IntVar(value=0)

        ttk.Label(controls, text="MNIST test sample index:").grid(
            row=0, column=0, padx=(0, 6), pady=4, sticky="w"
        )
        self.index_spin = ttk.Spinbox(
            controls,
            from_=0,
            to=9999,
            textvariable=self.index_var,
            width=10,
        )
        self.index_spin.grid(row=0, column=1, padx=(0, 8), pady=4, sticky="w")

        ttk.Button(
            controls, text="Run Dataset Sample", command=self.update_view_from_dataset
        ).grid(row=0, column=2, padx=(0, 8), pady=4, sticky="w")

        ttk.Button(
            controls, text="Random Sample", command=self.pick_random_sample
        ).grid(row=0, column=3, padx=(0, 8), pady=4, sticky="w")

        ttk.Button(
            controls, text="Run Drawn Digit", command=self.update_view_from_drawn
        ).grid(row=0, column=4, padx=(0, 8), pady=4, sticky="w")

        ttk.Label(
            controls,
            text=(
                "Tip: use low-confidence outputs and compare contributor lists to "
                "teach how decision boundaries behave."
            ),
        ).grid(row=1, column=0, columnspan=5, pady=(2, 0), sticky="w")

        results = ttk.Frame(outer)
        results.pack(fill="both", expand=True)

        left = ttk.LabelFrame(results, text="Input Sources", padding=10)
        left.pack(side="left", fill="y", padx=(0, 8))

        self._build_left_panel(left)

        right = ttk.Frame(results)
        right.pack(side="left", fill="both", expand=True)

        decision_frame = ttk.LabelFrame(right, text="Decision Stages", padding=10)
        decision_frame.pack(fill="both", expand=True)

        self.stages_canvas = tk.Canvas(
            decision_frame,
            width=1040,
            height=560,
            bg="#f8fafc",
            highlightthickness=1,
            highlightbackground="#94a3b8",
        )
        self.stages_canvas.pack(fill="both", expand=True)

        contrib_frame = ttk.LabelFrame(
            right,
            text="Top Contributing Neurons (activation x weight)",
            padding=8,
        )
        contrib_frame.pack(fill="both", expand=False, pady=(8, 0))

        self.contrib_text = tk.Text(
            contrib_frame,
            width=110,
            height=11,
            wrap="word",
            bg="#ffffff",
            fg="#0f172a",
            font=("Menlo", 11),
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.contrib_text.pack(fill="both", expand=True)
        self.contrib_text.configure(state="disabled")

        status_frame = ttk.Frame(outer)
        status_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")

    def _build_left_panel(self, container: ttk.LabelFrame) -> None:
        sample_frame = ttk.LabelFrame(container, text="Current Input (28x28)", padding=8)
        sample_frame.pack(fill="x")

        self.input_canvas = tk.Canvas(
            sample_frame,
            width=360,
            height=360,
            bg="#111827",
            highlightthickness=1,
            highlightbackground="#94a3b8",
        )
        self.input_canvas.pack()

        ttk.Label(
            sample_frame,
            textvariable=self.truth_var,
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w", pady=(10, 2))
        ttk.Label(
            sample_frame,
            textvariable=self.prediction_var,
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w")

        draw_frame = ttk.LabelFrame(container, text="Draw Your Own Digit", padding=8)
        draw_frame.pack(fill="x", pady=(10, 0))

        self.draw_canvas = tk.Canvas(
            draw_frame,
            width=DRAW_CANVAS_SIZE,
            height=DRAW_CANVAS_SIZE,
            bg="#111827",
            highlightthickness=1,
            highlightbackground="#94a3b8",
        )
        self.draw_canvas.pack()

        self.draw_canvas.bind("<Button-1>", self._on_draw)
        self.draw_canvas.bind("<B1-Motion>", self._on_draw)

        draw_buttons = ttk.Frame(draw_frame)
        draw_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(draw_buttons, text="Clear Drawing", command=self.clear_drawing).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(
            draw_buttons, text="Run Drawn Digit", command=self.update_view_from_drawn
        ).pack(side="left")

        ttk.Label(
            draw_frame,
            text="Draw in white on black. Use 'Run Drawn Digit' for inference.",
        ).pack(anchor="w", pady=(8, 0))

    def _on_draw(self, event: tk.Event) -> None:
        cell_size = DRAW_CANVAS_SIZE / DRAW_GRID_SIZE
        col = int(event.x / cell_size)
        row = int(event.y / cell_size)
        self._paint_to_draw_buffer(row, col)
        self._redraw_draw_canvas()

    def _paint_to_draw_buffer(self, row: int, col: int) -> None:
        if row < 0 or row >= DRAW_GRID_SIZE or col < 0 or col >= DRAW_GRID_SIZE:
            return

        # Paint a small soft brush to make strokes less pixelated.
        brush = [
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

        for dr, dc, strength in brush:
            rr = row + dr
            cc = col + dc
            if 0 <= rr < DRAW_GRID_SIZE and 0 <= cc < DRAW_GRID_SIZE:
                self.draw_buffer[rr, cc] = min(1.0, self.draw_buffer[rr, cc] + strength)

    def _redraw_draw_canvas(self) -> None:
        self.draw_canvas.delete("all")

        cell = DRAW_CANVAS_SIZE / DRAW_GRID_SIZE
        for r in range(DRAW_GRID_SIZE):
            for c in range(DRAW_GRID_SIZE):
                value = float(self.draw_buffer[r, c])
                gray = int(value * 255)
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                x0 = c * cell
                y0 = r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                self.draw_canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=color,
                    outline=color,
                )

    def clear_drawing(self) -> None:
        self.draw_buffer.fill(0.0)
        self._redraw_draw_canvas()
        self.status_var.set("Drawing cleared.")

    def pick_random_sample(self) -> None:
        self.index_var.set(random.randint(0, self.max_index))
        self.update_view_from_dataset()

    def update_view_from_dataset(self) -> None:
        try:
            idx = int(self.index_var.get())
        except (ValueError, tk.TclError):
            self.status_var.set("Invalid index. Please enter a number.")
            return

        if idx < 0 or idx > self.max_index:
            self.status_var.set(f"Index out of range. Use 0 to {self.max_index}.")
            return

        x = self.x_test[idx]
        y_true = int(self.y_test[idx])
        self._run_inference_and_render(
            x=x,
            y_true=y_true,
            input_label=f"Dataset sample {idx}",
        )

    def update_view_from_drawn(self) -> None:
        x = self.draw_buffer.reshape(-1).astype(np.float32)
        if float(np.max(x)) <= 0.0:
            self.status_var.set("Draw a digit first, then run inference.")
            return

        self._run_inference_and_render(
            x=x,
            y_true=None,
            input_label="Drawn input",
        )

    def _run_inference_and_render(
        self,
        x: np.ndarray,
        y_true: int | None,
        input_label: str,
    ) -> None:
        dense1, dropout_out, dense2, probs = self.probe_model.predict(
            np.expand_dims(x, axis=0), verbose=0
        )

        dense1 = dense1[0]
        dropout_out = dropout_out[0]
        dense2 = dense2[0]
        probs = probs[0]

        y_pred = int(np.argmax(probs))
        confidence = float(probs[y_pred])

        if y_true is None:
            self.truth_var.set("Ground Truth: N/A (drawn input)")
        else:
            self.truth_var.set(f"Ground Truth: {y_true}")

        self.prediction_var.set(
            f"Prediction: {y_pred}  (confidence {confidence * 100:.2f}%)"
        )
        self.status_var.set(
            f"{input_label} processed. Review activation stages and contributor lists."
        )

        self._draw_input_image(x.reshape(28, 28))
        self._draw_decision_stages(dense1, dropout_out, dense2, probs, y_true, y_pred)
        self._render_contributor_text(x, dense1, dropout_out, dense2, probs, y_pred)

    def _draw_input_image(self, image_2d: np.ndarray) -> None:
        self.input_canvas.delete("all")

        margin = 12
        size = 336
        cell = size / 28

        self.input_canvas.create_rectangle(
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
                gray = int(value * 255)
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                x0 = margin + c * cell
                y0 = margin + r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                self.input_canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=color,
                    outline=color,
                )

    def _draw_activation_strip(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        activations: np.ndarray,
        title: str,
        color: str,
    ) -> None:
        canvas = self.stages_canvas
        canvas.create_rectangle(
            x,
            y,
            x + width,
            y + height,
            fill="#ffffff",
            outline="#cbd5e1",
            width=1,
        )

        canvas.create_text(
            x + 12,
            y + 18,
            text=title,
            anchor="w",
            font=("Helvetica", 12, "bold"),
            fill="#0f172a",
        )

        n = len(activations)
        plot_x = x + 12
        plot_y = y + 36
        plot_w = width - 24
        plot_h = height - 70

        max_val = max(float(np.max(activations)), 1e-8)
        bar_w = plot_w / n

        for i, val in enumerate(activations):
            norm = float(val) / max_val
            x0 = plot_x + i * bar_w
            y1 = plot_y + plot_h
            y0 = y1 - (norm * plot_h)
            canvas.create_line(x0, y1, x0, y0, fill=color, width=1)

        top_k = np.argsort(activations)[-5:][::-1]
        top_text = ", ".join(
            [f"n{int(i)}={float(activations[i]):.3f}" for i in top_k]
        )
        canvas.create_text(
            x + 12,
            y + height - 18,
            text=f"Top activations: {top_text}",
            anchor="w",
            font=("Helvetica", 10),
            fill="#334155",
        )

    def _draw_output_probabilities(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        probs: np.ndarray,
        y_true: int | None,
        y_pred: int,
    ) -> None:
        canvas = self.stages_canvas
        canvas.create_rectangle(
            x,
            y,
            x + width,
            y + height,
            fill="#ffffff",
            outline="#cbd5e1",
            width=1,
        )

        canvas.create_text(
            x + 12,
            y + 18,
            text="Output Layer (Softmax) - Class Probabilities",
            anchor="w",
            font=("Helvetica", 12, "bold"),
            fill="#0f172a",
        )

        bar_area_x = x + 16
        bar_area_y = y + 42
        bar_area_w = width - 32
        gap = 4
        available_h = height - 52
        bar_h = max(8, int((available_h - (9 * gap)) / 10))

        for digit in range(10):
            p = float(probs[digit])
            y0 = bar_area_y + digit * (bar_h + gap)
            y1 = y0 + bar_h

            canvas.create_rectangle(
                bar_area_x,
                y0,
                bar_area_x + bar_area_w,
                y1,
                fill="#f1f5f9",
                outline="#e2e8f0",
            )

            fill_w = bar_area_w * p
            if digit == y_pred:
                fill = "#16a34a"
            elif y_true is not None and digit == y_true:
                fill = "#2563eb"
            else:
                fill = "#64748b"

            canvas.create_rectangle(
                bar_area_x,
                y0,
                bar_area_x + fill_w,
                y1,
                fill=fill,
                outline=fill,
            )

            label = f"Digit {digit}"
            marker = ""
            if y_true is not None and digit == y_true:
                marker += " (true)"
            if digit == y_pred:
                marker += " (pred)"

            canvas.create_text(
                bar_area_x + 6,
                y0 + bar_h / 2,
                text=label + marker,
                anchor="w",
                font=("Helvetica", 10),
                fill="#0f172a",
            )
            canvas.create_text(
                bar_area_x + bar_area_w - 6,
                y0 + bar_h / 2,
                text=f"{p * 100:.2f}%",
                anchor="e",
                font=("Helvetica", 10),
                fill="#0f172a",
            )

    def _draw_decision_stages(
        self,
        dense1: np.ndarray,
        dropout_out: np.ndarray,
        dense2: np.ndarray,
        probs: np.ndarray,
        y_true: int | None,
        y_pred: int,
    ) -> None:
        canvas = self.stages_canvas
        canvas.delete("all")

        canvas.create_text(
            14,
            14,
            text=(
                "Decision Flow: Input -> Dense(128) -> Dropout(0.2) -> "
                "Dense(64) -> Softmax(10)"
            ),
            anchor="nw",
            font=("Helvetica", 13, "bold"),
            fill="#0f172a",
        )

        self._draw_activation_strip(
            x=12,
            y=44,
            width=1008,
            height=120,
            activations=dense1,
            title="Stage 1: Dense(128) activations after ReLU",
            color="#059669",
        )

        self._draw_activation_strip(
            x=12,
            y=174,
            width=1008,
            height=120,
            activations=dropout_out,
            title="Stage 2: Dropout layer output (inference mode)",
            color="#d97706",
        )

        self._draw_activation_strip(
            x=12,
            y=304,
            width=1008,
            height=120,
            activations=dense2,
            title="Stage 3: Dense(64) activations after ReLU",
            color="#7c3aed",
        )

        self._draw_output_probabilities(
            x=12,
            y=434,
            width=1008,
            height=114,
            probs=probs,
            y_true=y_true,
            y_pred=y_pred,
        )

    def _format_top_contribs(
        self,
        values: np.ndarray,
        prefix: str,
        top_n: int = 5,
    ) -> str:
        pos_idx = np.argsort(values)[-top_n:][::-1]
        neg_idx = np.argsort(values)[:top_n]

        pos_lines = [
            f"  + {prefix}{int(i):>3}: {float(values[i]): .5f}" for i in pos_idx
        ]
        neg_lines = [
            f"  - {prefix}{int(i):>3}: {float(values[i]): .5f}" for i in neg_idx
        ]

        return "\n".join(
            ["Top positive contributors:"] + pos_lines + ["Top negative contributors:"] + neg_lines
        )

    def _render_contributor_text(
        self,
        x: np.ndarray,
        dense1: np.ndarray,
        dropout_out: np.ndarray,
        dense2: np.ndarray,
        probs: np.ndarray,
        y_pred: int,
    ) -> None:
        # Contribution to predicted class from hidden2 neurons.
        # Shape: (64,), each term is activation_i * weight_i_to_pred_class
        out_contrib = dense2 * self.w_out[:, y_pred]

        # For learning context, inspect the strongest hidden2 neuron and
        # decompose which hidden1 neurons drove it.
        top_h2 = int(np.argmax(dense2))
        h1_to_h2_contrib = dropout_out * self.w_dense2[:, top_h2]

        # Also expose input-pixel influence for one top hidden1 neuron.
        top_h1 = int(np.argmax(dense1))
        input_to_h1_contrib = x * self.w_dense1[:, top_h1]

        sorted_probs = np.argsort(probs)[::-1]
        top3 = ", ".join(
            [f"{int(i)}={float(probs[i]) * 100:.2f}%" for i in sorted_probs[:3]]
        )

        blocks = [
            "Prediction Summary",
            f"  Predicted class: {y_pred}",
            f"  Top probabilities: {top3}",
            "",
            "A) Hidden2 -> Output contributors for predicted class",
            "  Formula per neuron i: contribution_i = a2[i] * W_out[i, pred_class]",
            self._format_top_contribs(out_contrib, prefix="h2_"),
            "",
            "B) Hidden1 -> strongest Hidden2 neuron contributors",
            f"  Target strongest hidden2 neuron: h2_{top_h2}",
            "  Formula per neuron j: contribution_j = a1_dropout[j] * W_dense2[j, h2_target]",
            self._format_top_contribs(h1_to_h2_contrib, prefix="h1_"),
            "",
            "C) Input pixel -> strongest Hidden1 neuron contributors",
            f"  Target strongest hidden1 neuron: h1_{top_h1}",
            "  Formula per pixel p: contribution_p = x[p] * W_dense1[p, h1_target]",
            self._format_top_contribs(input_to_h1_contrib, prefix="px_"),
        ]

        self.contrib_text.configure(state="normal")
        self.contrib_text.delete("1.0", "end")
        self.contrib_text.insert("1.0", "\n".join(blocks))
        self.contrib_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    NNTrainingToolUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
