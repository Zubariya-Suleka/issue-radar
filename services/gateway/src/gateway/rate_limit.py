"""Per-client token buckets (D-016).

Each client has a bucket holding up to `burst` tokens, refilled continuously at
`per_minute / 60` tokens per second; a request spends one. A client can burst briefly,
but its long-run rate is capped. Unlike a fixed window ("30 per calendar minute"), there
is no boundary where a client can fire 2x the limit back to back.
"""

import time
from collections.abc import Callable


class TokenBucket:
    def __init__(self, rate_per_s: float, capacity: float, now: float) -> None:
        self._rate = rate_per_s
        self._capacity = capacity
        self._tokens = capacity
        self._updated = now

    def take(self, now: float) -> float:
        """Spend one token. Returns 0 if allowed, else seconds until a token is available."""
        # Refill lazily for the time since the last call: no background timer needed.
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
        self._updated = now
        if self._tokens >= 1:
            self._tokens -= 1
            return 0.0
        return (1 - self._tokens) / self._rate


class RateLimiter:
    """One bucket per client. Clients are configured API keys, so the dict stays bounded.

    In-memory: correct for one gateway process. Several processes would each allow the
    full rate; shared state (Valkey, D-005) would be needed then.
    """

    def __init__(
        self, per_minute: float, burst: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._rate = per_minute / 60
        self._burst = burst
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, client: str) -> float:
        """0 if the request may proceed, else seconds the client should wait."""
        now = self._clock()
        bucket = self._buckets.get(client)
        if bucket is None:
            bucket = self._buckets[client] = TokenBucket(self._rate, self._burst, now)
        return bucket.take(now)
