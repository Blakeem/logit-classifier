"""The status line under a node and the progress bars core's generate builds inside a node's run."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


def send_status(node_id: str | None, text: str) -> None:
    """Show text under the node's progress bar, or do nothing without a node id, a server or a server instance."""
    instance: Any = None

    if node_id is None:
        return
    try:
        from server import PromptServer
    except ImportError:
        return
    # Core sets PromptServer.instance only when it constructs a server.
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return
    try:
        instance.send_progress_text(text, node_id)
    except Exception:
        # The status line is decoration, so a failed send must never fail the run.
        return


def token_fill(value: float, total: float, ratio: float) -> float:
    """Return a token bar's progress as a fraction, held at 1 once value reaches ratio of total."""
    if total <= 0:
        return 1.0
    # Core sizes its token bar to the token budget and a greedy reply stops early.
    return min(1.0, max(0.0, value / (ratio * total)))


@contextmanager
def routed_progress_bars(on_update: Callable[[float, float, Any], None]) -> Iterator[type]:
    """Route every comfy.utils.ProgressBar built inside the block to on_update, and yield core's own class.

    Core's clip.generate builds its own bar per call and each new bar resets the node's display,
    so the caller builds its one bar from the yielded class and folds the routed updates into it.
    """
    import comfy.model_management
    import comfy.utils

    core_bar: type = comfy.utils.ProgressBar

    class RoutedProgressBar:
        def __init__(self, total: float, node_id: Any = None) -> None:
            self.total = total
            self.current: float = 0
            self.node_id = node_id

        def update_absolute(self, value: float, total: float | None = None, preview: Any = None) -> None:
            # Core's bar hook is the only Cancel check inside a generate, and a routed update never reaches it.
            comfy.model_management.throw_exception_if_processing_interrupted()
            if total is not None:
                self.total = total
            self.current = value
            on_update(value, self.total, preview)

        def update(self, value: float) -> None:
            self.update_absolute(self.current + value)

    comfy.utils.ProgressBar = RoutedProgressBar
    try:
        yield core_bar
    finally:
        comfy.utils.ProgressBar = core_bar


def run_outcome(exc_type: type[BaseException] | None, done_text: str) -> str:
    """Return the last status line of a run: cancelled, failed or done_text.

    Frontend 1.53.6 ignores an empty progress text, so a run must end on a line that says how it ended.
    """
    import comfy.model_management

    if exc_type is None:
        return done_text
    if issubclass(exc_type, comfy.model_management.InterruptProcessingException):
        return "cancelled"
    return "failed"
