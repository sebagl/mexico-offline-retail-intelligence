"""Tracks recent health of the optional generation provider for ``/health``."""

import threading
import time
from collections.abc import Callable

FAILURE_MEMORY_SECONDS = 300.0


class ProviderStatus:
    """Remembers the most recent provider failure for a short window.

    A configured Gemini that keeps failing (quota, outage) is reported as
    ``degraded`` rather than ``ok`` so operators can see it without reading
    logs. State is process-local and intentionally tiny.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._last_failure_at: float | None = None
        self._last_failure_category: str | None = None

    def record_success(self) -> None:
        with self._lock:
            self._last_failure_at = None
            self._last_failure_category = None

    def record_failure(self, category: str) -> None:
        with self._lock:
            self._last_failure_at = self._clock()
            self._last_failure_category = category

    @property
    def recent_failure(self) -> str | None:
        with self._lock:
            if self._last_failure_at is None:
                return None
            if self._clock() - self._last_failure_at > FAILURE_MEMORY_SECONDS:
                return None
            return self._last_failure_category
