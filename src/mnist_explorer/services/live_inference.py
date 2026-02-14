"""Background live-inference worker.

This module owns all threading/queue behavior used for "live while drawing"
predictions. Keeping it isolated from Tkinter controller code makes the UI
class easier to reason about and easier to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LiveInferenceResult:
    """Single completed inference bundle produced by the worker.

    The UI needs both raw input (`x`) and all intermediate outputs to decide
    whether expensive panels should refresh.
    """

    x: np.ndarray
    dense1: np.ndarray
    dropout_out: np.ndarray
    dense2: np.ndarray
    probs: np.ndarray


class LiveInferenceWorker:
    """Threaded worker that computes live inference off the Tk main thread.

    Design goals:
    - keep queue size at 1 so old frames are dropped automatically
    - allow cheap non-blocking `submit(...)` from mouse-move handlers
    - expose `poll_latest(...)` so UI can apply only the freshest result
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        self._predict_fn = predict_fn
        self._input_queue: Queue[np.ndarray] = Queue(maxsize=1)
        self._result_queue: Queue[LiveInferenceResult] = Queue(maxsize=1)
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        """Start the worker thread once."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal worker shutdown."""
        self._stop_event.set()

    def submit(self, x: np.ndarray) -> bool:
        """Submit latest drawn input for inference.

        Returns `True` when queued successfully. Returns `False` only in the
        unlikely case queue operations fail.
        """
        try:
            # Drain stale input first so only latest drawing state is processed.
            while True:
                self._input_queue.get_nowait()
        except Empty:
            pass

        try:
            self._input_queue.put_nowait(np.array(x, copy=True))
        except Exception:
            return False
        return True

    def poll_latest(self) -> LiveInferenceResult | None:
        """Return the newest completed result, dropping stale older ones."""
        latest: LiveInferenceResult | None = None
        try:
            while True:
                latest = self._result_queue.get_nowait()
        except Empty:
            return latest

    def is_stopped(self) -> bool:
        """Check whether worker has been requested to stop."""
        return self._stop_event.is_set()

    def _run_loop(self) -> None:
        """Internal worker loop.

        This function intentionally avoids any Tk calls. It only transforms
        queued inputs into model outputs.
        """
        while not self._stop_event.is_set():
            try:
                x = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            dense1, dropout_out, dense2, probs = self._predict_fn(x)
            result = LiveInferenceResult(
                x=x,
                dense1=dense1,
                dropout_out=dropout_out,
                dense2=dense2,
                probs=probs,
            )

            try:
                # Keep only freshest result to minimize UI lag.
                while True:
                    self._result_queue.get_nowait()
            except Empty:
                pass

            self._result_queue.put(result)
