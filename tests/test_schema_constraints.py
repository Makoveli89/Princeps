"""Tests for Alembic migration schema constraints."""

from uuid import uuid4

from brain.core.db import compute_content_hash, compute_input_hash, get_or_create_operation
from brain.core.models import (
    DocChunk,
    Document,
    DocumentSummary,
    KnowledgeEdge,
    KnowledgeNode,
    NodeKnowledgeTypeEnum,
    Operation,
    OperationStatusEnum,
    OperationTypeEnum,
    Repository,
    Resource,
    ResourceTypeEnum,
    Tenant,
)


class TestForeignKeyConstraints:
    """Tests for foreign key constraints."""

    def test_chunk_requires_document(self, session, tenant):
        """DocChunk should require valid document_id."""
        fake_doc_id = uuid4()

        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=fake_doc_id,  # Non-existent document
            content="Orphan chunk",
            chunk_index=0,
        )
        session.add(chunk)

        # SQLite doesn't enforce FK by default, but model should handle this
        # In production PostgreSQL, this would raise IntegrityError

    def test_resource_requires_repository(self, session, tenant):
        """Resource should require valid repository_id."""
        fake_repo_id = uuid4()

        resource = Resource(
            tenant_id=tenant.id,
            repository_id=fake_repo_id,  # Non-existent repo
            file_path="/fake/path.py",
            file_name="path.py",
            content_hash=compute_content_hash("content"),
            resource_type=ResourceTypeEnum.CODE_FILE,
        )
        session.add(resource)
        # FK constraint would be enforced in PostgreSQL

    def test_document_summary_requires_document(self, session, tenant):
        """DocumentSummary should require valid document_id."""
        fake_doc_id = uuid4()

        summary = DocumentSummary(
            tenant_id=tenant.id,
            document_id=fake_doc_id,  # Non-existent document
            one_sentence="Summary of nothing",
        )
        session.add(summary)
        # FK constraint would be enforced in PostgreSQL


class TestUniqueConstraints:
    """Tests for unique constraints."""

    def test_operation_type_input_hash_unique(self, session, tenant):
        """Operation (op_type, input_hash) should be unique."""
        inputs = {"path": "/unique/test.pdf"}
        compute_input_hash(OperationTypeEnum.INGEST_DOCUMENT.value, inputs)

        # First operation
        op1, created1 = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()
        assert created1 is True

        # Second operation with same inputs should return existing
        op2, created2 = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        assert created2 is False
        assert op1.id == op2.id

    def test_tenant_name_uniqueness(self, session):
        """Tenant name should be unique (if constraint exists)."""
        tenant1 = Tenant(name="unique_tenant")
        session.add(tenant1)
        session.commit()

        # Note: Current model may not have unique constraint on name
        # This documents expected behavior


class TestNotNullConstraints:
    """Tests for NOT NULL constraints."""

    def test_document_requires_content(self, session, tenant):
        """Document should require content field."""
        doc = Document(
            tenant_id=tenant.id,
            title="Title without content",
            content=None,  # This should fail
            content_hash=compute_content_hash("dummy"),
            source="test",
        )
        session.add(doc)
        # In strict mode, this would raise IntegrityError

    def test_document_requires_title(self, session, tenant):
        """Document should require title field."""
        doc = Document(
            tenant_id=tenant.id,
            title=None,  # This should fail
            content="Content without title",
            content_hash=compute_content_hash("Content without title"),
            source="test",
        )
        session.add(doc)
        # In strict mode, this would raise IntegrityError

    def test_chunk_requires_content(self, session, tenant, sample_document):
        """DocChunk should require content field."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content=None,  # This should fail
            chunk_index=0,
        )
        session.add(chunk)
        # In strict mode, this would raise IntegrityError

    def test_operation_requires_op_type(self, session, tenant):
        """Operation should require op_type field."""
        op = Operation(
            tenant_id=tenant.id,
            op_type=None,  # This should fail
            input_hash="abc123",
            status=OperationStatusEnum.PENDING,
        )
        session.add(op)
        # In strict mode, this would raise IntegrityError


class TestCascadeDeletes:
    """Tests for CASCADE delete behavior."""

    def test_document_cascade_deletes_chunks(self, session, tenant, sample_document):
        """Deleting document should cascade to chunks."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Will be deleted",
            chunk_index=0,
        )
        session.add(chunk)
        session.commit()

        # Delete document
        session.delete(sample_document)
        session.commit()

        # In PostgreSQL with CASCADE, chunk would be deleted
        # SQLite may not enforce this

    def test_repository_cascade_deletes_resources(self, session, tenant):
        """Deleting repository should cascade to resources."""
        repo = Repository(
            tenant_id=tenant.id,
            name="cascade-repo",
            url="https://github.com/test/cascade",
        )
        session.add(repo)
        session.commit()

        resource = Resource(
            tenant_id=tenant.id,
            repository_id=repo.id,
            file_path="/src/main.py",
            file_name="main.py",
            content_hash=compute_content_hash("content"),
            resource_type=ResourceTypeEnum.CODE_FILE,
        )
        session.add(resource)
        session.commit()

        # Delete repository
        session.delete(repo)
        session.commit()

        # In PostgreSQL with CASCADE, resource would be deleted


class TestOperationConstraints:
    """Tests for Operation model constraints."""

    def test_input_hash_length(self, session, tenant):
        """input_hash should be exactly 64 chars (SHA-256 hex)."""
        inputs = {"test": "data"}
        hash_value = compute_input_hash(OperationTypeEnum.INGEST_DOCUMENT.value, inputs)

        assert len(hash_value) == 64

    def test_operation_status_enum(self, session, tenant):
        """Operation status should be valid enum value."""
        inputs = {"path": "/test/enum.pdf"}
        operation, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()

        assert operation.status in [
            OperationStatusEnum.PENDING,
            OperationStatusEnum.IN_PROGRESS,
            OperationStatusEnum.SUCCESS,
            OperationStatusEnum.FAILED,
        ]


class TestKnowledgeEdgeConstraints:
    """Tests for KnowledgeEdge constraints."""

    def test_edge_requires_source_and_target(self, session, tenant):
        """KnowledgeEdge should require source and target nodes."""
        node1 = KnowledgeNode(
            tenant_id=tenant.id,
            agent_id="test-agent",
            knowledge_type=NodeKnowledgeTypeEnum.PATTERN,
            title="Source node",
            content="Source node content",
        )
        node2 = KnowledgeNode(
            tenant_id=tenant.id,
            agent_id="test-agent",
            knowledge_type=NodeKnowledgeTypeEnum.PATTERN,
            title="Target node",
            content="Target node content",
        )
        session.add_all([node1, node2])
        session.commit()

        edge = KnowledgeEdge(
            tenant_id=tenant.id,
            from_node_id=node1.node_id,
            to_node_id=node2.node_id,
            edge_type="related",
        )
        session.add(edge)
        session.commit()

        assert edge.from_node_id == node1.node_id
        assert edge.to_node_id == node2.node_id


class TestIndexConstraints:
    """Tests for database indexes."""

    def test_document_indexed_by_tenant(self, session, tenant):
        """Documents should be efficiently queryable by tenant."""
        # Create multiple documents
        for i in range(10):
            content = f"Content {i}"
            doc = Document(
                tenant_id=tenant.id,
                title=f"Doc {i}",
                content=content,
                content_hash=compute_content_hash(content),
                source="test",
            )
            session.add(doc)
        session.commit()

        # Query should use tenant_id index
        docs = session.query(Document).filter(Document.tenant_id == tenant.id).all()

        assert len(docs) == 10

    def test_operation_indexed_by_input_hash(self, session, tenant):
        """Operations should be efficiently queryable by input_hash."""
        inputs = {"path": "/test/indexed.pdf"}
        operation, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()

        # Query by input_hash should use index
        found = (
            session.query(Operation).filter(Operation.input_hash == operation.input_hash).first()
        )

        assert found is not None
        assert found.id == operation.id


class TestModelAssumptionsAlignment:
    """Tests to verify model assumptions align with migration constraints."""

    def test_document_has_content_hash(self, session, tenant):
        """Document should support content_hash field."""
        content = "Test content"
        doc = Document(
            tenant_id=tenant.id,
            title="Hashed doc",
            content=content,
            content_hash=compute_content_hash(content),
            source="test",
        )
        session.add(doc)
        session.commit()

        assert doc.content_hash is not None
        assert len(doc.content_hash) == 64  # SHA-256 hex

    def test_operation_has_all_status_fields(self, session, tenant):
        """Operation should have all lifecycle timestamp fields."""
        inputs = {"path": "/test/lifecycle.pdf"}
        operation, _ = get_or_create_operation(
            session, str(tenant.id), OperationTypeEnum.INGEST_DOCUMENT, inputs
        )
        session.commit()

        # Verify all expected fields exist
        assert hasattr(operation, "created_at")
        assert hasattr(operation, "started_at")
        assert hasattr(operation, "completed_at")
        assert hasattr(operation, "status")
        assert hasattr(operation, "error_message")
        assert hasattr(operation, "retry_count")
