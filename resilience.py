import asyncio
import random
import time

import anthropic
import httpx


class TokenBucket:
    """Rate limiter. The bucket refills at `rate` tokens/sec up to `capacity`.

    Each API call takes a token. If the bucket is empty, the caller waits until
    enough tokens have refilled. `capacity` is the burst size allowed at once.
    """

    def __init__(self, rate: float, capacity: float | None = None):
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n: float = 1) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            await asyncio.sleep(wait)  # sleep outside the lock so others can check


def is_retryable(exc: Exception) -> bool:
    """Transient errors are worth retrying. Bad requests and bad keys are not."""
    if isinstance(
        exc,
        (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError),
    ):
        return True
    if isinstance(exc, httpx.TransportError):  # connect errors, timeouts
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


async def retry(fn, *args, attempts: int = 5, base: float = 1.0, cap: float = 20.0):
    """Call `await fn(*args)`, retrying transient failures with exponential backoff + jitter.

    Wait before retry n is a random time in [0, min(cap, base * 2**n)] ("full jitter").
    The randomness stops 100 failed calls from all retrying at the same instant.
    """
    for attempt in range(attempts):
        try:
            return await fn(*args)
        except Exception as exc:
            if attempt == attempts - 1 or not is_retryable(exc):
                raise
            await asyncio.sleep(random.uniform(0, min(cap, base * 2**attempt)))
