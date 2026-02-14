"""Unit tests for recent live-drawing behavior in the UI module.

These tests focus on logic (not full Tk rendering) so they stay fast and stable.
"""

from __future__ import annotations

import numpy as np

import nn_visual_trainer_ui as ui_mod


class _DummyVar:
    """Simple stand-in for tkinter.StringVar used in logic tests."""

    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class _DummyRoot:
    """Minimal stand-in for tk root after/cancel scheduling APIs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def after(self, delay_ms: int, _fn):  # noqa: ANN001 - callback type not important for this unit test
        self.calls.append(("after", delay_ms))
        return "job-1"

    def after_cancel(self, _job):  # noqa: ANN001 - job id type not important for this unit test
        self.calls.append(("cancel", 0))


def _make_ui_shell() -> ui_mod.NNTrainingToolUI:
    """Create a lightweight UI object without constructing full Tk widgets."""
    ui = ui_mod.NNTrainingToolUI.__new__(ui_mod.NNTrainingToolUI)
    ui.root = _DummyRoot()
    ui.draw_buffer = np.zeros((28, 28), dtype=np.float32)
    ui.truth_var = _DummyVar()
    ui.prediction_var = _DummyVar()
    ui.probe_model = object()
    ui._live_predict_job = None
    ui._last_live_predict_ts = 0.0
    ui._last_live_heavy_render_ts = 0.0
    ui._last_live_heavy_pred = None
    ui._cancel_thinking_animation = lambda: None
    return ui


def test_live_prediction_empty_input_updates_top_input_panel(monkeypatch) -> None:
    """When drawing is empty, UI should reset labels and draw preprocessed top input."""
    ui = _make_ui_shell()

    called = {"input_draws": 0}

    def _draw_input_image(image_2d: np.ndarray) -> None:
        called["input_draws"] += 1
        assert image_2d.shape == (28, 28)

    ui._draw_input_image = _draw_input_image

    monkeypatch.setattr(ui_mod, "preprocess_drawn_digit", lambda _img: np.zeros((784,), dtype=np.float32))
    monkeypatch.setattr(ui_mod.time, "monotonic", lambda: 1.0)

    ui._run_live_draw_prediction()

    assert ui.truth_var.value == "Ground Truth: N/A (drawn input)"
    assert ui.prediction_var.value == "Prediction: -"
    assert called["input_draws"] == 1


def test_live_prediction_heavy_refresh_runs_only_when_due_or_class_changes(monkeypatch) -> None:
    """Heavy panels should be throttled to reduce drawing lag."""
    ui = _make_ui_shell()

    x = np.ones((784,), dtype=np.float32) * 0.5
    dense1 = np.ones((128,), dtype=np.float32) * 0.1
    dropout_out = np.ones((128,), dtype=np.float32) * 0.1
    dense2 = np.ones((64,), dtype=np.float32) * 0.1
    probs = np.zeros((10,), dtype=np.float32)
    probs[3] = 0.9

    draws = {"input": 0, "stages": 0, "contrib": 0}

    ui._draw_input_image = lambda _image: draws.__setitem__("input", draws["input"] + 1)
    ui._draw_decision_stages = lambda *_args, **_kwargs: draws.__setitem__("stages", draws["stages"] + 1)
    ui._render_contributor_text = lambda *_args, **_kwargs: draws.__setitem__("contrib", draws["contrib"] + 1)

    monkeypatch.setattr(ui_mod, "preprocess_drawn_digit", lambda _img: x)
    monkeypatch.setattr(ui_mod, "run_probe_prediction", lambda _model, _x: (dense1, dropout_out, dense2, probs))

    # First call: class changed from None -> 3, so heavy refresh must run.
    monkeypatch.setattr(ui_mod.time, "monotonic", lambda: 10.0)
    ui._run_live_draw_prediction()

    assert draws["input"] >= 1
    assert draws["stages"] == 1
    assert draws["contrib"] == 1

    # Second call shortly after with same class: heavy refresh should be skipped.
    prev_stages = draws["stages"]
    prev_contrib = draws["contrib"]
    monkeypatch.setattr(ui_mod.time, "monotonic", lambda: 10.05)
    ui._run_live_draw_prediction()

    assert draws["stages"] == prev_stages
    assert draws["contrib"] == prev_contrib
    # Lightweight input refresh still happens on each live update.
    assert draws["input"] >= 2


def test_schedule_live_prediction_runs_immediately_when_interval_elapsed(monkeypatch) -> None:
    """Scheduler should run prediction immediately when enough time has passed."""
    ui = _make_ui_shell()
    ui._last_live_predict_ts = 1.0

    called = {"live": 0}
    ui._run_live_draw_prediction = lambda: called.__setitem__("live", called["live"] + 1)

    monkeypatch.setattr(ui_mod.time, "monotonic", lambda: 2.0)

    ui._schedule_live_draw_prediction()

    assert called["live"] == 1
    assert ui._live_predict_job is None
