import pytest
import asyncio
import time
from framework.utilities.retry import async_retry

@pytest.mark.asyncio
async def test_async_retry_success():
    """Test that the function returns correctly on success."""
    @async_retry(tries=3, delay=0.01)
    async def successful_func():
        return "success"

    result = await successful_func()
    assert result == "success"

@pytest.mark.asyncio
async def test_async_retry_eventually_succeeds():
    """Test that the function retries and eventually succeeds."""

    # Using a list to hold the mock because python closures are weird
    attempts = 0

    @async_retry(tries=3, delay=0.01, jitter=(1.0, 1.0))
    async def flaky_func():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("fail")
        return "success"

    result = await flaky_func()
    assert result == "success"
    assert attempts == 3

@pytest.mark.asyncio
async def test_async_retry_fails():
    """Test that the function raises exception after max retries."""
    attempts = 0

    @async_retry(tries=3, delay=0.01, jitter=(1.0, 1.0))
    async def failing_func():
        nonlocal attempts
        attempts += 1
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        await failing_func()

    assert attempts == 3
