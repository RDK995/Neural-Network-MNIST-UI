"""Tests for the background live-inference service."""

from __future__ import annotations

import time

import numpy as np

from mnist_explorer.services.live_inference import LiveInferenceWorker


def _predict_fn(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Simulate a deterministic model response shape.
    dense1 = np.ones((128,), dtype=np.float32) * float(np.sum(x))
    dropout_out = dense1.copy()
    dense2 = np.ones((64,), dtype=np.float32) * 0.25
    probs = np.zeros((10,), dtype=np.float32)
    probs[3] = 0.9
    return dense1, dropout_out, dense2, probs


def test_worker_processes_latest_submission() -> None:
    worker = LiveInferenceWorker(predict_fn=_predict_fn)
    worker.start()

    # Submit two frames quickly; worker should eventually expose the latest.
    x1 = np.zeros((784,), dtype=np.float32)
    x2 = np.ones((784,), dtype=np.float32)
    assert worker.submit(x1)
    assert worker.submit(x2)

    result = None
    deadline = time.time() + 1.0
    while time.time() < deadline:
        result = worker.poll_latest()
        if result is not None:
            break
        time.sleep(0.01)

    worker.stop()

    assert result is not None
    assert result.x.shape == (784,)
    # Because x2 is all ones, dense1 should be sum(x2)=784 for every neuron.
    assert float(result.dense1[0]) == 784.0
    assert int(np.argmax(result.probs)) == 3


def test_worker_stop_flag_updates_state() -> None:
    worker = LiveInferenceWorker(predict_fn=_predict_fn)
    assert not worker.is_stopped()
    worker.stop()
    assert worker.is_stopped()
