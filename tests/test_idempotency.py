"""Tests for idempotency logic and Operation.input_hash."""
import pytest
from uuid import uuid4
from datetime import datetime, timedelta

from brain.core.models import Operation, OperationTypeEnum, OperationStatusEnum, compute_input_hash
from brain.core.db import get_or_create_operation, mark_operation_started, mark_operation_success, mark_operation_failed
from brain.resilience.idempotency_service import (
    IdempotencyManager,
    IdempotencyConfig,
    compute_input_hash as idem_compute_hash,
    IdempotentOperationScope,
)


class TestInputHashComputation:
    """Tests for input hash computation."""

    def test_same_inputs_produce_same_hash(self):
        """Same inputs should always produce the same hash."""
        inputs = {"path": "/path/to/file.pdf", "tenant": "test"}
        hash1 = compute_input_hash("ingest_document", inputs)
        hash2 = compute_input_hash("ingest_document", inputs)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_different_inputs_produce_different_hash(self):
        """Different inputs should produce different hashes."""
        inputs1 = {"path": "/path/to/file1.pdf"}
        inputs2 = {"path": "/path/to/file2.pdf"}
        hash1 = compute_input_hash("ingest_document", inputs1)
        hash2 = compute_input_hash("ingest_document", inputs2)
        assert hash1 != hash2

    def test_different_op_types_produce_different_hash(self):
        """Different operation types should produce different hashes."""
        inputs = {"path": "/path/to/file.pdf"}
        hash1 = compute_input_hash("ingest_document", inputs)
        hash2 = compute_input_hash("ingest_repo", inputs)
        assert hash1 != hash2

    def test_key_order_does_not_affect_hash(self):
        """Hash should be deterministic regardless of key order."""
        inputs1 = {"a": 1, "b": 2, "c": 3}
        inputs2 = {"c": 3, "a": 1, "b": 2}
        hash1 = compute_input_hash("test", inputs1)
        hash2 = compute_input_hash("test", inputs2)
        assert hash1 == hash2

    def test_path_normalization_in_idempotency_config(self):
        """Path normalization should handle Windows/Unix paths."""
        config = IdempotencyConfig(normalize_paths=True)
        inputs1 = {"path": "C:\\Users\\test\\file.pdf"}
        inputs2 = {"path": "C:/Users/test/file.pdf"}
        hash1 = idem_compute_hash("test", inputs1, config)
        hash2 = idem_compute_hash("test", inputs2, config)
        assert hash1 == hash2


class TestGetOrCreateOperation:
    """Tests for get_or_create_operation function."""

    def test_creates_new_operation(self, session, tenant):
        """Should create a new operation when none exists."""
        inputs = {"path": "/test/new.pdf"}
        operation, created = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        assert created is True
        assert operation is not None
        assert operation.status == OperationStatusEnum.PENDING
        assert operation.input_hash is not None

    def test_returns_existing_operation(self, session, tenant):
        """Should return existing operation with same inputs."""
        inputs = {"path": "/test/existing.pdf"}
        op1, created1 = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()

        op2, created2 = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        assert created1 is True
        assert created2 is False
        assert op1.id == op2.id

    def test_different_op_types_create_separate_operations(self, session, tenant):
        """Different op types with same inputs should create separate operations."""
        inputs = {"path": "/test/file.pdf"}
        op1, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()

        op2, created = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.ANALYSIS, inputs
        )
        assert created is True
        assert op1.id != op2.id


class TestOperationLifecycle:
    """Tests for operation lifecycle management."""

    def test_mark_operation_started(self, session, tenant):
        """Test marking an operation as started."""
        inputs = {"path": "/test/lifecycle.pdf"}
        operation, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()

        mark_operation_started(session, operation.id)
        session.commit()
        session.refresh(operation)

        assert operation.status == OperationStatusEnum.IN_PROGRESS
        assert operation.started_at is not None

    def test_mark_operation_success(self, session, tenant):
        """Test marking an operation as successful."""
        inputs = {"path": "/test/success.pdf"}
        operation, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()

        mark_operation_started(session, operation.id)
        session.commit()

        outputs = {"documents_created": 1, "chunks": 5}
        mark_operation_success(session, operation.id, outputs)
        session.commit()
        session.refresh(operation)

        assert operation.status == OperationStatusEnum.SUCCESS
        assert operation.completed_at is not None
        assert operation.outputs == outputs

    def test_mark_operation_failed(self, session, tenant):
        """Test marking an operation as failed."""
        inputs = {"path": "/test/failed.pdf"}
        operation, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()

        mark_operation_started(session, operation.id)
        session.commit()

        mark_operation_failed(session, operation.id, "File not found", "Traceback...")
        session.commit()
        session.refresh(operation)

        assert operation.status == OperationStatusEnum.FAILED
        assert operation.error_message == "File not found"
        assert operation.retry_count == 1


class TestIdempotencyManager:
    """Tests for IdempotencyManager class."""

    def test_check_operation_allows_new(self, session, tenant):
        """Should allow new operations to run."""
        manager = IdempotencyManager(session, str(tenant.id))
        inputs = {"path": "/test/manager_new.pdf"}
        result = manager.check_operation(OperationTypeEnum.INGEST_DOCUMENT, inputs)

        assert result.should_run is True
        assert result.was_skipped is False
        assert result.existing_operation is None

    def test_check_operation_skips_success(self, session, tenant):
        """Should skip operations that already succeeded."""
        manager = IdempotencyManager(session, str(tenant.id))
        inputs = {"path": "/test/manager_success.pdf"}

        # Create and complete operation
        operation, _ = manager.create_operation(OperationTypeEnum.INGEST_DOCUMENT, inputs)
        manager.start_operation(operation.id)
        manager.complete_operation(operation.id, {"result": "ok"}, success=True)
        session.commit()

        # Check if it should run again
        result = manager.check_operation(OperationTypeEnum.INGEST_DOCUMENT, inputs)
        assert result.should_run is False
        assert result.skip_reason == "Already completed successfully"
        assert result.cached_result == {"result": "ok"}

    def test_check_operation_skips_in_progress(self, session, tenant):
        """Should skip operations that are in progress."""
        manager = IdempotencyManager(session, str(tenant.id))
        inputs = {"path": "/test/manager_progress.pdf"}

        # Create and start operation
        operation, _ = manager.create_operation(OperationTypeEnum.INGEST_DOCUMENT, inputs)
        manager.start_operation(operation.id)
        session.commit()

        # Check if it should run again
        result = manager.check_operation(OperationTypeEnum.INGEST_DOCUMENT, inputs)
        assert result.should_run is False
        assert result.skip_reason == "Already in progress"

    def test_stale_operation_retry(self, session, tenant):
        """Should retry stale in-progress operations."""
        config = IdempotencyConfig(stale_in_progress_minutes=0, auto_retry_stale=True)
        manager = IdempotencyManager(session, str(tenant.id), config)
        inputs = {"path": "/test/manager_stale.pdf"}

        # Create and start operation with old timestamp
        operation, _ = manager.create_operation(OperationTypeEnum.INGEST_DOCUMENT, inputs)
        operation.status = OperationStatusEnum.IN_PROGRESS
        operation.started_at = datetime.utcnow() - timedelta(hours=2)
        session.commit()

        # Check if it should run again
        result = manager.check_operation(OperationTypeEnum.INGEST_DOCUMENT, inputs)
        assert result.should_run is True  # Should retry stale operation


class TestAllOperationTypesGenerateHashes:
    """Verify all operation types correctly generate and use input hashes."""

    def test_ingest_document_hash(self, session, tenant):
        """INGEST_DOCUMENT should generate consistent hash."""
        inputs = {"path": "/docs/test.pdf", "content_hash": "abc123"}
        op, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        assert op.input_hash is not None
        expected_hash = compute_input_hash(OperationTypeEnum.INGEST_DOCUMENT.value, inputs)
        assert op.input_hash == expected_hash

    def test_ingest_repo_hash(self, session, tenant):
        """INGEST_REPO should generate consistent hash."""
        inputs = {"path": "/repos/project", "url": "https://github.com/test/project"}
        op, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_REPO, inputs
        )
        assert op.input_hash is not None
        expected_hash = compute_input_hash(OperationTypeEnum.INGEST_REPO.value, inputs)
        assert op.input_hash == expected_hash

    def test_chunk_document_hash(self, session, tenant):
        """CHUNK_DOCUMENT should generate consistent hash."""
        inputs = {"document_id": str(uuid4()), "chunk_size": 800}
        op, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.CHUNK_DOCUMENT, inputs
        )
        assert op.input_hash is not None

    def test_generate_embedding_hash(self, session, tenant):
        """GENERATE_EMBEDDING should generate consistent hash."""
        inputs = {"chunk_id": str(uuid4()), "model": "all-MiniLM-L6-v2"}
        op, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.GENERATE_EMBEDDING, inputs
        )
        assert op.input_hash is not None

    def test_analysis_hash(self, session, tenant):
        """ANALYSIS should generate consistent hash."""
        inputs = {"document_id": str(uuid4()), "summary": True, "entities": True}
        op, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.ANALYSIS, inputs
        )
        assert op.input_hash is not None

    def test_retrieval_hash(self, session, tenant):
        """RETRIEVAL should generate consistent hash."""
        inputs = {"query": "machine learning pipeline", "top_k": 5}
        op, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.RETRIEVAL, inputs
        )
        assert op.input_hash is not None
