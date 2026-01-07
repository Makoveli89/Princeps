"""Error Handler - Centralized error handling."""

import functools


class BrainError(Exception):
    """Base exception for Brain layer."""

    pass


class IngestionError(BrainError):
    """Error during document/repo ingestion."""

    pass


class DistillationError(BrainError):
    """Error during knowledge distillation."""

    pass


class DatabaseError(BrainError):
    """Database operation error."""

    pass


class ErrorHandler:
    def __init__(self, logger=None, raise_on_error=True):
        self.logger = logger
        self.raise_on_error = raise_on_error

    def handle(self, error: Exception, context: dict | None = None):
        if self.logger:
            self.logger.error(f"Error: {error}", extra={"context": context})
        if self.raise_on_error:
            raise error


def handle_errors(error_class=BrainError, logger=None):
    """Decorator for error handling."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"Error in {func.__name__}: {e}")
                raise error_class(str(e)) from e

        return wrapper

    return decorator
