import asyncio
import functools
import logging
import random
from typing import Type, Tuple, Union, Callable, Any

logger = logging.getLogger(__name__)

def async_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (Exception,),
    jitter: bool = True
) -> Callable:
    """
    A dependency-free decorator to retry async functions with exponential backoff.

    Args:
        max_retries (int): Maximum number of retries before giving up.
        initial_delay (float): Initial delay in seconds.
        max_delay (float): Maximum delay in seconds to wait between retries.
        backoff_factor (float): Multiplier for the delay after each failure.
        exceptions (Union[Type[Exception], Tuple[Type[Exception], ...]]): Exceptions to trigger a retry.
        jitter (bool): Whether to add random jitter to the delay to prevent thundering herd.

    Returns:
        Callable: The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = initial_delay
            attempt = 0

            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries. Last error: {str(e)}")
                        raise e

                    delay = current_delay
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Function {func.__name__} failed with {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s... (Attempt {attempt}/{max_retries})"
                    )

                    await asyncio.sleep(delay)

                    current_delay = min(current_delay * backoff_factor, max_delay)
        return wrapper
    return decorator
