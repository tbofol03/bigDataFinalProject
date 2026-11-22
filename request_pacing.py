"""Coordinate request start times and cooldowns across concurrent workers."""

from math import isfinite
import threading
import time
from types import SimpleNamespace


class RequestPacer:
    def __init__(self, requests_per_minute: float, cooldown_seconds: float = 30.0):
        if (not isfinite(requests_per_minute) or requests_per_minute <= 0
                or not isfinite(cooldown_seconds) or cooldown_seconds <= 0):
            raise ValueError("Request pacing values must be finite and positive")
        self.interval = 60 / requests_per_minute
        self.cooldown_seconds = cooldown_seconds
        self.lock = threading.Lock()
        self.next_start = 0.0
        self.cooldown_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                remaining = max(self.next_start, self.cooldown_until) - now
                if remaining <= 0:
                    self.next_start = now + self.interval
                    return
            time.sleep(min(remaining, 1.0))

    def defer(self, error):
        duration = self.cooldown_seconds
        response = getattr(error, "response", None)
        if response is not None:
            try:
                value = float(response.headers.get("retry-after", ""))
                if isfinite(value) and value > 0:
                    duration = max(duration, value)
            except (TypeError, ValueError):
                pass
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + duration)


class PacedClient:
    def __init__(self, client, requests_per_minute, cooldown_seconds=30.0):
        self.pacer = RequestPacer(requests_per_minute, cooldown_seconds)
        self.create_request = client.chat.completions.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **options):
        self.pacer.wait()
        try:
            return self.create_request(**options)
        except Exception as error:
            if getattr(error, "status_code", None) == 429 or type(error).__name__ == "RateLimitError":
                self.pacer.defer(error)
            raise
