"""Retry Manager - Configurable retry logic with backoff."""

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class ExponentialBackoff:
    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self.attempt = 0

    def next_delay(self) -> float:
        delay = min(
            self.config.initial_delay * (self.config.exponential_base**self.attempt),
            self.config.max_delay,
        )
        self.attempt += 1
        return delay

    def reset(self):
        self.attempt = 0


class RetryManager:
    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def execute(self, func: Callable, *args, **kwargs):
        backoff = ExponentialBackoff(self.config)
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    time.sleep(backoff.next_delay())

        if last_error is not None:
            raise last_error
        raise RuntimeError("No attempts made")


def retry_with_backoff(
    config: RetryConfig | None = None, exceptions: tuple[type[Exception], ...] = (Exception,)
):
    """Decorator for automatic retry with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = RetryManager(config)
            return manager.execute(func, *args, **kwargs)

        return wrapper

    return decorator
