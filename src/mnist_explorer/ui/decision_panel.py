"""Rendering and explainability helpers for the decision panel."""

from __future__ import annotations

import tkinter as tk

import numpy as np

from mnist_explorer.ui.render import format_top_contribs, layer_y_positions, mix_with_white, pick_indices


def render_decision_stages(
    canvas: tk.Canvas,
    x_input: np.ndarray,
    dense1: np.ndarray,
    dropout_out: np.ndarray,
    dense2: np.ndarray,
    probs: np.ndarray,
    y_true: int | None,
    y_pred: int,
) -> None:
    """Draw network visualization and strongest decision path."""
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

    strongest_full = [
        int(np.argmax(x_input)),
        int(np.argmax(dense1)),
        int(np.argmax(dropout_out)),
        int(np.argmax(dense2)),
        y_pred,
    ]
    for li in range(len(layer_defs) - 1):
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
    canvas.create_text(
        bar_x0 - 2,
        network_top - 10,
        text="Softmax Probabilities",
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


def build_contributor_text(
    x: np.ndarray,
    dense1: np.ndarray,
    dropout_out: np.ndarray,
    dense2: np.ndarray,
    probs: np.ndarray,
    y_pred: int,
    w_dense1: np.ndarray,
    w_dense2: np.ndarray,
    w_out: np.ndarray,
) -> str:
    """Build human-readable contributor report for the prediction."""
    out_contrib = dense2 * w_out[:, y_pred]

    top_h2 = int(np.argmax(dense2))
    h1_to_h2_contrib = dropout_out * w_dense2[:, top_h2]

    top_h1 = int(np.argmax(dense1))
    input_to_h1_contrib = x * w_dense1[:, top_h1]

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
    return "\n".join(blocks)
