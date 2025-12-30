"""Batch Processing Utilities."""
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchConfig:
    batch_size: int = 100
    max_errors: int = 10
    continue_on_error: bool = True

@dataclass
class BatchResult:
    success: bool = True
    processed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

class BatchProcessor:
    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()

    def process(self, items: Iterable, processor: Callable[[Any], Any]) -> BatchResult:
        result = BatchResult()
        error_count = 0

        for batch in self._batch(items):
            for item in batch:
                try:
                    processor(item)
                    result.processed += 1
                except Exception as e:
                    result.failed += 1
                    result.errors.append(str(e))
                    error_count += 1

                    if error_count >= self.config.max_errors:
                        result.success = False
                        return result

                    if not self.config.continue_on_error:
                        result.success = False
                        return result

        result.success = result.failed == 0
        return result

    def _batch(self, items: Iterable) -> Generator[list, None, None]:
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def process_in_batches(items: Iterable, processor: Callable, batch_size: int = 100) -> BatchResult:
    """Convenience function for batch processing."""
    config = BatchConfig(batch_size=batch_size)
    return BatchProcessor(config).process(items, processor)
