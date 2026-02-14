"""Layout, style, and key-binding helpers for the MNIST explorer UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

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
    COLOR_SUB,
    DRAW_CANVAS_SIZE,
)


def configure_styles(root: tk.Tk) -> None:
    """Define ttk style rules so widgets share one visual language."""
    style = ttk.Style(root)
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


def bind_shortcuts(ui: object) -> None:
    """Register keyboard shortcuts for fast interaction."""
    ui.root.bind("<Return>", lambda _e: ui.update_view_from_dataset())
    ui.root.bind("<KP_Enter>", lambda _e: ui.update_view_from_dataset())
    ui.root.bind("r", lambda _e: ui.pick_random_sample())
    ui.root.bind("R", lambda _e: ui.pick_random_sample())
    ui.root.bind("d", lambda _e: ui.update_view_from_drawn())
    ui.root.bind("D", lambda _e: ui.update_view_from_drawn())
    ui.root.bind("c", lambda _e: ui.clear_drawing())
    ui.root.bind("C", lambda _e: ui.clear_drawing())
    ui.root.bind("<Escape>", lambda _e: ui.clear_drawing())


def build_layout(ui: object) -> None:
    """Create top-level layout containers and major UI sections."""
    outer = ttk.Frame(ui.root, padding=14, style="App.TFrame")
    outer.pack(fill="both", expand=True)

    _build_header(outer)
    _build_controls(ui, outer)
    _build_main_area(ui, outer)
    _build_status(ui, outer)


def _build_header(parent: ttk.Frame) -> None:
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


def _build_controls(ui: object, parent: ttk.Frame) -> None:
    controls = ttk.LabelFrame(
        parent,
        text="Dataset Input Controls",
        padding=12,
        style="Card.TLabelframe",
    )
    controls.pack(fill="x", pady=(0, 10))
    controls.grid_columnconfigure(6, weight=1)

    ui.index_var = tk.IntVar(value=0)

    ttk.Label(controls, text="MNIST test sample index:", style="Section.TLabel").grid(
        row=0,
        column=0,
        padx=(0, 6),
        pady=4,
        sticky="w",
    )
    ui.index_spin = ttk.Spinbox(
        controls,
        from_=0,
        to=9999,
        textvariable=ui.index_var,
        width=10,
        style="App.TSpinbox",
    )
    ui.index_spin.grid(row=0, column=1, padx=(0, 8), pady=4, sticky="w")

    ttk.Button(
        controls,
        text="Run Dataset Sample",
        command=ui.update_view_from_dataset,
        style="Neutral.TButton",
    ).grid(row=0, column=2, padx=(0, 8), pady=4, sticky="w")

    ttk.Button(
        controls,
        text="Random Sample",
        command=ui.pick_random_sample,
        style="Neutral.TButton",
    ).grid(row=0, column=3, padx=(0, 8), pady=4, sticky="w")

    ttk.Button(
        controls,
        text="Clear Drawn Digit",
        command=ui.clear_drawing,
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


def _build_main_area(ui: object, parent: ttk.Frame) -> None:
    results = ttk.Frame(parent, style="App.TFrame")
    results.pack(fill="both", expand=True)

    left = ttk.LabelFrame(results, text="Input Sources", padding=10, style="Card.TLabelframe")
    left.pack(side="left", fill="y", padx=(0, 10))
    _build_left_panel(ui, left)

    right = ttk.Frame(results, style="App.TFrame")
    right.pack(side="left", fill="both", expand=True)

    decision_frame = ttk.LabelFrame(
        right,
        text="Decision Stages",
        padding=10,
        style="Card.TLabelframe",
    )
    decision_frame.pack(fill="both", expand=True)

    ui.stages_canvas = tk.Canvas(
        decision_frame,
        width=1040,
        height=560,
        bg="#f4f8ff",
        highlightthickness=1,
        highlightbackground=COLOR_EDGE,
    )
    ui.stages_canvas.pack(fill="both", expand=True)

    contrib_frame = ttk.LabelFrame(
        right,
        text="Top Contributing Neurons (activation x weight)",
        padding=8,
        style="Card.TLabelframe",
    )
    contrib_frame.pack(fill="both", expand=False, pady=(8, 0))

    ui.contrib_text = tk.Text(
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
    contrib_scroll = ttk.Scrollbar(contrib_frame, orient="vertical", command=ui.contrib_text.yview)
    ui.contrib_text.configure(yscrollcommand=contrib_scroll.set)
    ui.contrib_text.pack(side="left", fill="both", expand=True)
    contrib_scroll.pack(side="right", fill="y")
    ui.contrib_text.configure(state="disabled")


def _build_status(ui: object, parent: ttk.Frame) -> None:
    status_frame = ttk.Frame(parent, style="App.TFrame")
    status_frame.pack(fill="x", pady=(8, 0))

    ui.status_label = tk.Label(
        status_frame,
        textvariable=ui.status_var,
        bg=COLOR_STATUS_INFO_BG,
        fg=COLOR_STATUS_INFO_FG,
        font=("Avenir Next", 10, "bold"),
        padx=10,
        pady=8,
        anchor="w",
        relief="flat",
    )
    ui.status_label.pack(fill="x", anchor="w")


def _build_left_panel(ui: object, container: ttk.LabelFrame) -> None:
    sample_frame = ttk.LabelFrame(
        container,
        text="Current Input (Preprocessed 28x28)",
        padding=8,
        style="Card.TLabelframe",
    )
    sample_frame.pack(fill="x")

    ui.input_canvas = tk.Canvas(
        sample_frame,
        width=360,
        height=360,
        bg="#111827",
        highlightthickness=1,
        highlightbackground=COLOR_EDGE,
    )
    ui.input_canvas.pack()

    ttk.Label(sample_frame, textvariable=ui.truth_var, style="Section.TLabel").pack(
        anchor="w", pady=(10, 2)
    )
    ttk.Label(sample_frame, textvariable=ui.prediction_var, style="Section.TLabel").pack(anchor="w")

    draw_frame = ttk.LabelFrame(
        container,
        text="Draw Your Own Digit",
        padding=8,
        style="Card.TLabelframe",
    )
    draw_frame.pack(fill="x", pady=(10, 0))

    ui.draw_canvas = tk.Canvas(
        draw_frame,
        width=DRAW_CANVAS_SIZE,
        height=DRAW_CANVAS_SIZE,
        bg="#111827",
        highlightthickness=1,
        highlightbackground=COLOR_EDGE,
        cursor="crosshair",
        takefocus=1,
    )
    ui.draw_canvas.pack()
    ui.draw_canvas.bind("<Button-1>", ui._on_draw)
    ui.draw_canvas.bind("<B1-Motion>", ui._on_draw)
    ui.draw_canvas.bind("<ButtonRelease-1>", lambda _e: setattr(ui, "_last_draw_cell", None))

    ttk.Label(
        draw_frame,
        text=(
            "Draw in white on black. The top input panel now shows the "
            "preprocessed model input in real time."
        ),
        style="Body.TLabel",
    ).pack(anchor="w", pady=(8, 0))
