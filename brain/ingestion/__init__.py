"""
Brain Ingestion Module
======================

Document and repository ingestion pipeline.

Exports:
    - IngestService: Main ingestion service
    - IngestConfig: Configuration dataclass
    - IngestResult: Result dataclass
    - TextExtractor: Extract text from various formats
    - TextChunker: Split text into chunks
    - EmbeddingService: Generate embeddings
"""

from .ingest_service import (
    DependencyParser,
    EmbeddingService,
    GitUtil,
    IngestConfig,
    IngestResult,
    IngestService,
    SecurityScanner,
    TextChunker,
    TextExtractor,
)

__all__ = [
    "IngestService",
    "IngestConfig",
    "IngestResult",
    "TextExtractor",
    "TextChunker",
    "EmbeddingService",
    "DependencyParser",
    "SecurityScanner",
    "GitUtil",
]
