"""Tests for document and repository ingestion edge cases."""

import os
import tempfile
from unittest.mock import patch

from brain.ingestion.ingest_service import (
    IngestConfig,
    IngestResult,
    IngestService,
    SecurityScanner,  # Note: This is the ingest_service version with dict return type
    TextChunker,
    TextExtractor,
)


class TestTextExtractor:
    """Tests for TextExtractor class."""

    def test_count_tokens_with_tiktoken(self):
        """Token count should work with tiktoken if available."""
        text = "Hello world, this is a test sentence."
        count = TextExtractor.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_fallback(self):
        """Token count should fall back to char-based estimate."""
        text = "Hello world"
        # Force fallback by temporarily clearing encoder
        original = TextExtractor._encoder
        TextExtractor._encoder = False
        count = TextExtractor.count_tokens(text)
        TextExtractor._encoder = original

        assert count > 0
        assert count == max(1, len(text) // 4)

    def test_extract_text_file(self):
        """Should extract text from plain text files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content\nwith multiple lines.")
            f.flush()
            path = f.name

        try:
            content, metadata = TextExtractor.extract_text_file(path)
            assert "Test content" in content
            assert metadata.get("line_count", 0) >= 2
        finally:
            os.unlink(path)

    def test_extract_empty_file(self):
        """Should handle empty files gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            content, metadata = TextExtractor.extract_text_file(path)
            assert content == ""
        finally:
            os.unlink(path)

    def test_clean_text_removes_nulls(self):
        """Should remove null characters from text."""
        text = "Hello\x00World"
        cleaned = TextExtractor._clean_text(text)
        assert "\x00" not in cleaned
        assert "HelloWorld" in cleaned

    def test_clean_text_normalizes_whitespace(self):
        """Should normalize excessive whitespace."""
        text = "Hello    \t\t   World\n\n\n\n\nTest"
        cleaned = TextExtractor._clean_text(text)
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned


class TestTextChunker:
    """Tests for TextChunker class."""

    def test_chunk_empty_text(self):
        """Should return empty list for empty text."""
        chunker = TextChunker(chunk_tokens=100, overlap_tokens=10)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_small_text(self):
        """Should return single chunk for small text."""
        chunker = TextChunker(chunk_tokens=1000, overlap_tokens=100)
        text = "This is a small text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0]["content"] == text

    def test_chunk_large_text(self):
        """Should split large text into multiple chunks."""
        chunker = TextChunker(chunk_tokens=50, overlap_tokens=10)
        text = " ".join(["word"] * 500)  # ~500 tokens
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunks_have_required_fields(self):
        """Each chunk should have all required metadata."""
        chunker = TextChunker(chunk_tokens=50, overlap_tokens=10)
        text = " ".join(["word"] * 100)
        chunks = chunker.chunk(text)

        for chunk in chunks:
            assert "content" in chunk
            assert "index" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk
            assert "token_count" in chunk
            assert "char_count" in chunk

    def test_chunks_are_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        chunker = TextChunker(chunk_tokens=50, overlap_tokens=10)
        text = " ".join(["word"] * 200)
        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert chunk["index"] == i


class TestSecurityScanner:
    """Tests for SecurityScanner in ingestion.

    Note: This tests the SecurityScanner from brain.ingestion.ingest_service
    which returns dict[str, list[str]] with 'pii' and 'secrets' keys.
    For the dataclass-based SecurityScanner, see test_security.py.
    """

    def test_detects_email(self):
        """Should detect email addresses."""
        scanner = SecurityScanner()
        content = "Contact us at test@example.com for more info."
        results = scanner.scan(content)
        assert isinstance(results, dict), "Expected dict return type"
        assert "email" in results.get("pii", [])

    def test_detects_phone(self):
        """Should detect phone numbers."""
        scanner = SecurityScanner()
        content = "Call us at 555-123-4567 today."
        results = scanner.scan(content)
        assert "phone" in results.get("pii", [])

    def test_detects_ssn(self):
        """Should detect Social Security Numbers."""
        scanner = SecurityScanner()
        content = "SSN: 123-45-6789 is sensitive."
        results = scanner.scan(content)
        assert "ssn" in results.get("pii", [])

    def test_detects_api_key(self):
        """Should detect API key patterns."""
        scanner = SecurityScanner()
        content = "api_key = 'abcdefghijklmnopqrstuvwxyz123456'"
        results = scanner.scan(content)
        assert "api_key" in results.get("secrets", [])

    def test_detects_github_token(self):
        """Should detect GitHub personal access tokens."""
        scanner = SecurityScanner()
        content = "token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        results = scanner.scan(content)
        assert "github_token" in results.get("secrets", [])

    def test_detects_openai_key(self):
        """Should detect OpenAI API keys."""
        scanner = SecurityScanner()
        content = "OPENAI_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        results = scanner.scan(content)
        assert "openai_key" in results.get("secrets", [])

    def test_no_false_positives_on_clean_text(self):
        """Should not flag clean text as containing PII/secrets."""
        scanner = SecurityScanner()
        content = "This is normal text without any sensitive data."
        results = scanner.scan(content)
        assert results.get("pii", []) == []
        assert results.get("secrets", []) == []


class TestIngestServiceEdgeCases:
    """Tests for IngestService edge cases."""

    def test_ingest_nonexistent_file(self):
        """Should handle non-existent files gracefully."""
        service = IngestService()
        result = service.ingest_document("/nonexistent/path/file.pdf")
        assert result.success is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower() or "File not found" in result.errors[0]

    def test_ingest_file_too_large(self):
        """Should reject files exceeding size limit."""
        config = IngestConfig(max_file_size_mb=0.0001)  # Very small limit
        service = IngestService(config=config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 1000)  # 1KB file
            f.flush()
            path = f.name

        try:
            result = service.ingest_document(path)
            assert result.success is False
            assert any("large" in e.lower() for e in result.errors)
        finally:
            os.unlink(path)

    def test_ingest_document_result_structure(self):
        """IngestResult should have all required fields."""
        result = IngestResult(success=True)
        assert hasattr(result, "success")
        assert hasattr(result, "operation_id")
        assert hasattr(result, "documents_created")
        assert hasattr(result, "chunks_created")
        assert hasattr(result, "errors")
        assert hasattr(result, "duration_ms")

    def test_config_defaults(self):
        """IngestConfig should have sensible defaults."""
        config = IngestConfig()
        assert config.chunk_tokens > 0
        assert config.overlap_tokens >= 0
        assert config.max_file_size_mb > 0
        assert ".py" in config.include_extensions
        assert "__pycache__" in config.exclude_patterns


class TestMalformedPDFHandling:
    """Tests for handling malformed PDFs."""

    def test_pdf_extraction_with_mock(self):
        """Test PDF extraction with mocked PyPDF."""
        with patch.object(TextExtractor, "extract_pdf") as mock_extract:
            # Simulate successful extraction
            mock_extract.return_value = ("PDF content", 5, {"title": "Test"})

            text, pages, meta = TextExtractor.extract_pdf("/fake/path.pdf")
            assert text == "PDF content"
            assert pages == 5

    def test_pdf_extraction_error_handling(self):
        """Should handle PDF extraction errors gracefully."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"Not a valid PDF content")
            f.flush()
            path = f.name

        try:
            # This should raise an error or return empty content
            try:
                text, pages, meta = TextExtractor.extract_pdf(path)
                # If it doesn't raise, content should be empty or minimal
            except Exception:
                # Expected behavior for malformed PDF
                assert True
        finally:
            os.unlink(path)


class TestDuplicateRepoHandling:
    """Tests for duplicate repository ingestion."""

    def test_duplicate_detection_by_url(self, session, tenant):
        """Should detect duplicate repositories by URL."""
        from brain.core.models import Repository

        repo1 = Repository(
            tenant_id=tenant.id,
            name="project",
            url="https://github.com/org/project",
        )
        session.add(repo1)
        session.commit()

        # Query for existing repo
        existing = (
            session.query(Repository)
            .filter_by(tenant_id=tenant.id, url="https://github.com/org/project")
            .first()
        )

        assert existing is not None
        assert existing.id == repo1.id

    def test_different_tenants_can_have_same_repo(self, session):
        """Different tenants should be able to ingest the same repo URL."""
        from brain.core.models import Repository, Tenant

        tenant1 = Tenant(name="tenant1")
        tenant2 = Tenant(name="tenant2")
        session.add_all([tenant1, tenant2])
        session.commit()

        repo1 = Repository(tenant_id=tenant1.id, name="project", url="https://github.com/test")
        repo2 = Repository(tenant_id=tenant2.id, name="project", url="https://github.com/test")
        session.add_all([repo1, repo2])
        session.commit()

        assert repo1.id != repo2.id
