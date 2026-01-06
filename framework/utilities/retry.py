import asyncio
import functools
import logging
import random
from collections.abc import Callable
from typing import Any

# Simple logger for this module
logger = logging.getLogger("retry")


def async_retry(
    exceptions: type[Exception] | tuple[type[Exception], ...] = Exception,
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: tuple[float, float] = (0.5, 1.5),
    logger: logging.Logger = logger,
) -> Callable:
    """
    Decorator for retrying async functions with exponential backoff and jitter.

    :param exceptions: Exception or tuple of exceptions to check.
    :param tries: Total number of attempts (default 3).
    :param delay: Initial delay between retries in seconds (default 1).
    :param backoff: Multiplier applied to delay between attempts (default 2).
    :param jitter: Tuple of (min_mult, max_mult) to randomize delay (default (0.5, 1.5)).
    :param logger: Logger to use for warnings (default 'retry').
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            attempt = 1
            current_delay = delay
            while attempt <= tries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == tries:
                        logger.error(
                            f"Function {func.__name__} failed after {tries} attempts. Error: {e}"
                        )
                        raise e

                    # Calculate wait time with jitter
                    jitter_mult = random.uniform(*jitter)
                    sleep_time = current_delay * jitter_mult

                    logger.warning(
                        f"Attempt {attempt}/{tries} for {func.__name__} failed: {e}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )

                    await asyncio.sleep(sleep_time)
                    current_delay *= backoff
                    attempt += 1

        return wrapper

    return decorator
