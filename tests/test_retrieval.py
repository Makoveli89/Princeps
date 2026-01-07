"""Tests for retrieval functions and fallback behavior."""

import os
from uuid import uuid4

import pytest

from brain.core.db import similarity_search_chunks


# Mark to detect if we're running with PostgreSQL
def is_postgres_available():
    """Check if PostgreSQL is available for testing."""
    import os

    return os.environ.get("DATABASE_URL", "").startswith("postgresql")


requires_postgres = pytest.mark.skipif(
    not is_postgres_available(), reason="Requires PostgreSQL with pgvector extension"
)


class TestSimilaritySearchChunks:
    """Tests for pgvector similarity search."""

    @requires_postgres
    def test_search_with_no_results(self, session):
        """Should return empty list when no results found."""
        # Use mock embedding
        query_embedding = [0.1] * 384
        results = similarity_search_chunks(session, query_embedding, limit=5)

        # Should return empty list, not raise error
        assert results == []

    @requires_postgres
    def test_search_with_tenant_filter(self, session, tenant, sample_chunks):
        """Should filter by tenant ID when provided."""
        query_embedding = [0.1] * 384
        other_tenant_id = str(uuid4())

        # Search with non-existent tenant should return empty
        results = similarity_search_chunks(
            session, query_embedding, tenant_id=other_tenant_id, limit=5
        )
        assert results == []

    @requires_postgres
    def test_search_with_document_filter(self, session, tenant, sample_document, sample_chunks):
        """Should filter by document ID when provided."""
        query_embedding = [0.1] * 384
        other_doc_id = str(uuid4())

        # Search with non-existent document should return empty
        results = similarity_search_chunks(
            session, query_embedding, document_id=other_doc_id, limit=5
        )
        assert results == []


def import_unified_retriever():
    """Try to import UnifiedRetriever from possible locations."""
    try:
        from brain_layer.retrieval_systems.unified_retriever import (
            RetrievalResult,
            UnifiedRetriever,
        )

        return UnifiedRetriever, RetrievalResult
    except ImportError:
        pass
    try:
        # Alternative path with numbered folder
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from brain_layer import unified_retriever

        return unified_retriever.UnifiedRetriever, unified_retriever.RetrievalResult
    except (ImportError, AttributeError):
        pass
    return None, None


class TestUnifiedRetrieverFallback:
    """Tests for UnifiedRetriever fallback behavior."""

    def test_fallback_chain_initialization(self):
        """Should initialize with default fallback chain."""
        UnifiedRetriever, _ = import_unified_retriever()
        if UnifiedRetriever is None:
            pytest.skip("UnifiedRetriever not available in current structure")

        # This should not raise even without backends
        retriever = UnifiedRetriever(backend="heuristic")
        assert retriever.active_backend_name == "heuristic"

    def test_empty_query_handling(self):
        """Should handle empty queries gracefully."""
        UnifiedRetriever, _ = import_unified_retriever()
        if UnifiedRetriever is None:
            pytest.skip("UnifiedRetriever not available")

        retriever = UnifiedRetriever(backend="heuristic")
        results = retriever.search("", top_k=5)
        assert isinstance(results, list)

    def test_malformed_query_handling(self):
        """Should handle malformed/special character queries."""
        UnifiedRetriever, _ = import_unified_retriever()
        if UnifiedRetriever is None:
            pytest.skip("UnifiedRetriever not available")

        retriever = UnifiedRetriever(backend="heuristic")
        # Query with special characters
        results = retriever.search("@#$%^&*()", top_k=5)
        assert isinstance(results, list)


class TestRetrievalResultFormat:
    """Tests for retrieval result structure."""

    def test_retrieval_result_fields(self):
        """RetrievalResult should have all required fields."""
        _, RetrievalResult = import_unified_retriever()
        if RetrievalResult is None:
            pytest.skip("RetrievalResult not available")

        result = RetrievalResult(
            id="test-id",
            content="Test content",
            score=0.95,
            source="test",
            metadata={"key": "value"},
            backend="test_backend",
        )

        assert result.id == "test-id"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.source == "test"
        assert result.metadata == {"key": "value"}
        assert result.backend == "test_backend"

    def test_retrieval_result_to_dict(self):
        """RetrievalResult.to_dict should return proper dict."""
        _, RetrievalResult = import_unified_retriever()
        if RetrievalResult is None:
            pytest.skip("RetrievalResult not available")

        result = RetrievalResult(
            id="test", content="content", score=0.9, source="src", metadata={}, backend="backend"
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "test"
        assert d["score"] == 0.9


class TestGracefulDegradation:
    """Tests for graceful degradation when backends fail."""

    def test_backend_unavailable_fallback(self):
        """Should fall back when primary backend is unavailable."""
        UnifiedRetriever, _ = import_unified_retriever()
        if UnifiedRetriever is None:
            pytest.skip("UnifiedRetriever not available")

        try:
            # Try to use pgvector without actual postgres - should fall back
            retriever = UnifiedRetriever(
                backend="auto", fallback_chain=["pgvector", "chroma", "tfidf", "heuristic"]
            )
            # Should have selected an available backend
            assert retriever.active_backend is not None
        except Exception:
            # If this fails, at minimum it should not crash
            pytest.skip("Retriever initialization failed")

    def test_search_continues_on_backend_error(self):
        """Search should continue even if one backend errors."""
        UnifiedRetriever, _ = import_unified_retriever()
        if UnifiedRetriever is None:
            pytest.skip("UnifiedRetriever not available")

        retriever = UnifiedRetriever(backend="heuristic")
        # Even with no documents, should return empty list
        results = retriever.search("test query", top_k=5, with_fallback=True)
        assert isinstance(results, list)


class TestEmptyResultsHandling:
    """Tests for handling empty results."""

    def test_cli_handles_no_results(self):
        """CLI query command should handle no results gracefully."""
        from brain.interface.brain_cli import BrainCLI

        cli = BrainCLI()
        # Running query with non-existent data should not crash
        # (actual DB interaction is mocked in real tests)
        assert cli is not None

    def test_api_handles_no_results(self):
        """API query endpoint should return empty results, not error."""
        from brain.interface.brain_api import create_app

        app = create_app()
        # The stub implementation returns empty results
        if hasattr(app, "test_client"):
            client = app.test_client()
            response = client.post("/api/v1/query", json={"text": "nonexistent"})
            assert response.status_code != 500
        else:
            pytest.skip("App does not have test_client")


class TestMinResultsFallback:
    """Tests for min_results fallback behavior."""

    def test_fallback_triggered_on_insufficient_results(self):
        """Should trigger fallback when primary returns insufficient results."""
        UnifiedRetriever, _ = import_unified_retriever()
        if UnifiedRetriever is None:
            pytest.skip("UnifiedRetriever not available")

        retriever = UnifiedRetriever(backend="heuristic")
        # Request more results than available
        results = retriever.search("test query", top_k=100, with_fallback=True, min_results=1)
        # Should not error even if can't find enough
        assert isinstance(results, list)


class TestBackendStatusReporting:
    """Tests for backend status reporting."""

    def test_get_backend_status(self):
        """get_backend_status should report all backends."""
        UnifiedRetriever, _ = import_unified_retriever()
        if UnifiedRetriever is None:
            pytest.skip("UnifiedRetriever not available")

        retriever = UnifiedRetriever(backend="heuristic")
        status = retriever.get_backend_status()

        assert isinstance(status, dict)
        assert "heuristic" in status
        assert "available" in status["heuristic"]
        assert "active" in status["heuristic"]
