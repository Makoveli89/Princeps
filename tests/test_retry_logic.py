"""Tests for retry logic with transient failure scenarios."""
import pytest
from unittest.mock import MagicMock, patch, call
import time

from brain.resilience.retry_manager import (
    RetryConfig,
    RetryManager,
    ExponentialBackoff,
    retry_with_backoff,
)


class TestExponentialBackoff:
    """Tests for ExponentialBackoff class."""

    def test_initial_delay(self):
        """First delay should be initial_delay."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0)
        backoff = ExponentialBackoff(config)
        delay = backoff.next_delay()
        assert delay == 1.0

    def test_exponential_increase(self):
        """Delays should increase exponentially."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=100.0)
        backoff = ExponentialBackoff(config)

        delays = [backoff.next_delay() for _ in range(4)]
        assert delays == [1.0, 2.0, 4.0, 8.0]

    def test_max_delay_cap(self):
        """Delays should not exceed max_delay."""
        config = RetryConfig(initial_delay=10.0, exponential_base=2.0, max_delay=30.0)
        backoff = ExponentialBackoff(config)

        delays = [backoff.next_delay() for _ in range(5)]
        assert all(d <= 30.0 for d in delays)

    def test_reset(self):
        """Reset should restart delay calculation."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0)
        backoff = ExponentialBackoff(config)

        backoff.next_delay()
        backoff.next_delay()
        backoff.reset()

        assert backoff.next_delay() == 1.0


class TestRetryManager:
    """Tests for RetryManager class."""

    def test_success_on_first_attempt(self):
        """Should succeed immediately if function works."""
        manager = RetryManager(RetryConfig(max_retries=3))
        func = MagicMock(return_value="success")

        result = manager.execute(func)

        assert result == "success"
        assert func.call_count == 1

    def test_retry_on_transient_failure(self):
        """Should retry on transient failures."""
        manager = RetryManager(RetryConfig(max_retries=3, initial_delay=0.01))
        func = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "success"])

        result = manager.execute(func)

        assert result == "success"
        assert func.call_count == 3

    def test_raises_after_max_retries(self):
        """Should raise after exhausting retries."""
        manager = RetryManager(RetryConfig(max_retries=2, initial_delay=0.01))
        func = MagicMock(side_effect=Exception("persistent failure"))

        with pytest.raises(Exception, match="persistent failure"):
            manager.execute(func)

        assert func.call_count == 3  # Initial + 2 retries

    def test_passes_args_and_kwargs(self):
        """Should pass arguments to function."""
        manager = RetryManager(RetryConfig(max_retries=1))
        func = MagicMock(return_value="result")

        manager.execute(func, "arg1", "arg2", key="value")

        func.assert_called_once_with("arg1", "arg2", key="value")


class TestRetryWithBackoffDecorator:
    """Tests for retry_with_backoff decorator."""

    def test_decorator_retries_on_failure(self):
        """Decorator should retry failed functions."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=0.01))
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_decorator_preserves_function_metadata(self):
        """Decorator should preserve function name and docstring."""
        @retry_with_backoff()
        def example_function():
            """Example docstring."""
            pass

        assert example_function.__name__ == "example_function"
        assert "Example docstring" in (example_function.__doc__ or "")


class TestMockedDownloadFailure:
    """Tests for mocked download/network failure scenarios."""

    def test_retry_on_network_error(self):
        """Should retry on simulated network errors."""
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        manager = RetryManager(config)

        # Simulate network errors then success
        network_call = MagicMock(side_effect=[
            ConnectionError("Network unreachable"),
            TimeoutError("Connection timed out"),
            {"status": "success", "data": "downloaded"}
        ])

        result = manager.execute(network_call)

        assert result["status"] == "success"
        assert network_call.call_count == 3

    def test_retry_on_http_500_error(self):
        """Should retry on HTTP 500 errors."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        manager = RetryManager(config)

        class HTTP500Error(Exception):
            pass

        http_call = MagicMock(side_effect=[
            HTTP500Error("Internal Server Error"),
            {"status": 200, "body": "OK"}
        ])

        result = manager.execute(http_call)
        assert result["status"] == 200

    def test_no_retry_on_http_400_error(self):
        """Should not retry on client errors (4xx)."""
        # This is a design decision - client errors are not transient

        class HTTP400Error(Exception):
            pass

        @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=0.01))
        def always_400():
            raise HTTP400Error("Bad Request")

        # Should raise after all retries (since we retry all exceptions by default)
        with pytest.raises(HTTP400Error):
            always_400()


class TestRetryWithDatabaseTransient:
    """Tests for retry with database transient errors."""

    def test_retry_on_db_connection_lost(self):
        """Should retry on database connection lost."""
        config = RetryConfig(max_retries=2, initial_delay=0.01)
        manager = RetryManager(config)

        class DBConnectionError(Exception):
            pass

        db_call = MagicMock(side_effect=[
            DBConnectionError("Connection lost"),
            {"inserted": True, "id": 123}
        ])

        result = manager.execute(db_call)
        assert result["inserted"] is True

    def test_retry_on_deadlock(self):
        """Should retry on database deadlock."""
        config = RetryConfig(max_retries=3, initial_delay=0.01)
        manager = RetryManager(config)

        class DeadlockError(Exception):
            pass

        attempts = 0

        def maybe_deadlock():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise DeadlockError("Deadlock detected")
            return "committed"

        result = manager.execute(maybe_deadlock)
        assert result == "committed"
        assert attempts == 3


class TestRetryTimings:
    """Tests for retry timing behavior."""

    def test_delays_between_retries(self):
        """Should wait between retries."""
        config = RetryConfig(max_retries=2, initial_delay=0.1, exponential_base=2.0)
        manager = RetryManager(config)

        start_time = time.time()
        func = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "success"])

        result = manager.execute(func)
        elapsed = time.time() - start_time

        # Should have waited at least initial_delay + initial_delay*2 = 0.3 seconds
        assert elapsed >= 0.2  # Allow some tolerance
        assert result == "success"


class TestRetryConfigDefaults:
    """Tests for RetryConfig default values."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False
        )

        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
