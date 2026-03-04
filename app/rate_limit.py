from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class InMemorySlidingWindowRateLimiter:
    def __init__(self, window_seconds: int, max_requests: int) -> None:
        self.window_seconds = max(1, int(window_seconds))
        self.max_requests = max(1, int(max_requests))
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> RateLimitDecision:
        now = time.monotonic()
        boundary = now - self.window_seconds

        with self._lock:
            queue = self._events[key]
            while queue and queue[0] <= boundary:
                queue.popleft()

            if len(queue) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - queue[0])))
                return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)

            queue.append(now)
            return RateLimitDecision(allowed=True, retry_after_seconds=0)
