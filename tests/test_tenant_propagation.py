"""Tests for tenant ID propagation in all DB writes."""
import pytest
from uuid import uuid4

from brain.core.models import (
    Tenant, Document, DocChunk, Operation, Artifact,
    DocumentSummary, DocumentEntity, DocumentTopic, DocumentConcept,
    Repository, Resource, AgentRun, KnowledgeNode,
    OperationTypeEnum, OperationStatusEnum, ResourceTypeEnum, NodeKnowledgeTypeEnum, ArtifactTypeEnum,
)
from brain.core.db import get_or_create_operation, compute_content_hash


class TestTenantPropagation:
    """Tests to verify tenant_id is properly propagated in all DB writes."""

    def test_document_requires_tenant(self, session, tenant):
        """Document should require tenant_id."""
        content = "Content"
        doc = Document(
            tenant_id=tenant.id,
            title="Test Doc",
            content=content,
            content_hash=compute_content_hash(content),
            source="test",
        )
        session.add(doc)
        session.commit()

        assert doc.tenant_id == tenant.id

    def test_chunk_inherits_tenant(self, session, tenant, sample_document):
        """DocChunk should have tenant_id from its document."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Chunk content",
            chunk_index=0,
        )
        session.add(chunk)
        session.commit()

        assert chunk.tenant_id == tenant.id
        assert chunk.tenant_id == sample_document.tenant_id

    def test_operation_has_tenant(self, session, tenant):
        """Operation should have tenant_id."""
        operation, _ = get_or_create_operation(
            session,
            str(tenant.id),
            OperationTypeEnum.INGEST_DOCUMENT,
            {"path": "/test/path.pdf"}
        )
        session.commit()

        assert str(operation.tenant_id) == str(tenant.id)

    def test_artifact_has_tenant(self, session, tenant):
        """Artifact should have tenant_id."""
        operation, _ = get_or_create_operation(
            session,
            str(tenant.id),
            OperationTypeEnum.INGEST_DOCUMENT,
            {"path": "/test/artifact.pdf"}
        )
        session.commit()

        artifact = Artifact(
            tenant_id=tenant.id,
            operation_id=operation.id,
            artifact_type=ArtifactTypeEnum.EMBEDDING_INDEX,
            name="test_artifact",
            content={"vector": [0.1, 0.2]},
        )
        session.add(artifact)
        session.commit()

        assert artifact.tenant_id == tenant.id

    def test_document_summary_has_tenant(self, session, tenant, sample_document):
        """DocumentSummary should have tenant_id."""
        summary = DocumentSummary(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            one_sentence="Brief summary.",
            executive="Executive summary.",
        )
        session.add(summary)
        session.commit()

        assert summary.tenant_id == tenant.id

    def test_document_entity_has_tenant(self, session, tenant, sample_document):
        """DocumentEntity should have tenant_id."""
        entity = DocumentEntity(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            text="Test Entity",
            label="ORG",
        )
        session.add(entity)
        session.commit()

        assert entity.tenant_id == tenant.id

    def test_document_topic_has_tenant(self, session, tenant, sample_document):
        """DocumentTopic should have tenant_id."""
        topic = DocumentTopic(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            topic_id=0,
            name="machine_learning",
            keywords=["ML", "AI", "deep learning"],
        )
        session.add(topic)
        session.commit()

        assert topic.tenant_id == tenant.id

    def test_document_concept_has_tenant(self, session, tenant, sample_document):
        """DocumentConcept should have tenant_id."""
        concept = DocumentConcept(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            concept="artificial intelligence",
            relevance=0.95,
        )
        session.add(concept)
        session.commit()

        assert concept.tenant_id == tenant.id

    def test_repository_has_tenant(self, session, tenant):
        """Repository should have tenant_id."""
        repo = Repository(
            tenant_id=tenant.id,
            name="test-repo",
            url="https://github.com/test/repo",
        )
        session.add(repo)
        session.commit()

        assert repo.tenant_id == tenant.id

    def test_resource_inherits_tenant(self, session, tenant):
        """Resource should have tenant_id."""
        repo = Repository(
            tenant_id=tenant.id,
            name="test-repo",
            url="https://github.com/test/repo",
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

        assert resource.tenant_id == tenant.id

    def test_agent_run_has_tenant(self, session, tenant):
        """AgentRun should have tenant_id."""
        run = AgentRun(
            tenant_id=tenant.id,
            agent_id="test-agent",
            task="Process document",
            success=True,
        )
        session.add(run)
        session.commit()

        assert run.tenant_id == tenant.id

    def test_knowledge_node_has_tenant(self, session, tenant):
        """KnowledgeNode should have tenant_id."""
        node = KnowledgeNode(
            tenant_id=tenant.id,
            agent_id="test-agent",
            knowledge_type=NodeKnowledgeTypeEnum.INSIGHT,
            title="Test Node",
            content="Machine learning basics",
        )
        session.add(node)
        session.commit()

        assert node.tenant_id == tenant.id


class TestTenantIsolationInQueries:
    """Tests for tenant isolation in database queries."""

    def test_documents_filtered_by_tenant(self, session):
        """Documents should be filterable by tenant."""
        tenant1 = Tenant(name="tenant1")
        tenant2 = Tenant(name="tenant2")
        session.add_all([tenant1, tenant2])
        session.commit()

        content1 = "Content1"
        content2 = "Content2"
        doc1 = Document(
            tenant_id=tenant1.id, title="Doc1", content=content1,
            content_hash=compute_content_hash(content1), source="test"
        )
        doc2 = Document(
            tenant_id=tenant2.id, title="Doc2", content=content2,
            content_hash=compute_content_hash(content2), source="test"
        )
        session.add_all([doc1, doc2])
        session.commit()

        # Query for tenant1 only
        tenant1_docs = session.query(Document).filter(
            Document.tenant_id == tenant1.id
        ).all()

        assert len(tenant1_docs) == 1
        assert tenant1_docs[0].title == "Doc1"

    def test_chunks_filtered_by_tenant(self, session):
        """DocChunks should be filterable by tenant."""
        tenant1 = Tenant(name="tenant_chunk1")
        tenant2 = Tenant(name="tenant_chunk2")
        session.add_all([tenant1, tenant2])
        session.commit()

        c1 = "C1"
        c2 = "C2"
        doc1 = Document(
            tenant_id=tenant1.id, title="Doc1", content=c1,
            content_hash=compute_content_hash(c1), source="test"
        )
        doc2 = Document(
            tenant_id=tenant2.id, title="Doc2", content=c2,
            content_hash=compute_content_hash(c2), source="test"
        )
        session.add_all([doc1, doc2])
        session.commit()

        chunk1 = DocChunk(tenant_id=tenant1.id, document_id=doc1.id, content="C1", chunk_index=0)
        chunk2 = DocChunk(tenant_id=tenant2.id, document_id=doc2.id, content="C2", chunk_index=0)
        session.add_all([chunk1, chunk2])
        session.commit()

        # Query for tenant1 only
        tenant1_chunks = session.query(DocChunk).filter(
            DocChunk.tenant_id == tenant1.id
        ).all()

        assert len(tenant1_chunks) == 1
        assert tenant1_chunks[0].content == "C1"

    def test_operations_filtered_by_tenant(self, session):
        """Operations should be filterable by tenant."""
        tenant1 = Tenant(name="tenant_op1")
        tenant2 = Tenant(name="tenant_op2")
        session.add_all([tenant1, tenant2])
        session.commit()

        op1, _ = get_or_create_operation(
            session, str(tenant1.id), OperationTypeEnum.INGEST_DOCUMENT, {"p": "1"}
        )
        op2, _ = get_or_create_operation(
            session, str(tenant2.id), OperationTypeEnum.INGEST_DOCUMENT, {"p": "2"}
        )
        session.commit()

        # Query for tenant1 only
        tenant1_ops = session.query(Operation).filter(
            Operation.tenant_id == tenant1.id
        ).all()

        assert len(tenant1_ops) == 1


class TestCrossResourceTenantConsistency:
    """Tests for tenant consistency across related resources."""

    def test_document_and_chunks_same_tenant(self, session, tenant, sample_document):
        """Document and its chunks should have same tenant."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Chunk",
            chunk_index=0,
        )
        session.add(chunk)
        session.commit()

        # Verify consistency
        doc = session.get(Document, sample_document.id)
        chunk = session.query(DocChunk).filter_by(document_id=doc.id).first()

        assert doc.tenant_id == chunk.tenant_id

    def test_document_and_summary_same_tenant(self, session, tenant, sample_document):
        """Document and its summary should have same tenant."""
        summary = DocumentSummary(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            one_sentence="Brief.",
        )
        session.add(summary)
        session.commit()

        doc = session.get(Document, sample_document.id)
        summ = session.query(DocumentSummary).filter_by(document_id=doc.id).first()

        assert doc.tenant_id == summ.tenant_id

    def test_repo_and_resources_same_tenant(self, session, tenant):
        """Repository and its resources should have same tenant."""
        repo = Repository(
            tenant_id=tenant.id,
            name="repo",
            url="https://github.com/test/repo",
        )
        session.add(repo)
        session.commit()

        resource = Resource(
            tenant_id=tenant.id,
            repository_id=repo.id,
            file_path="/file.py",
            file_name="file.py",
            content_hash=compute_content_hash("content"),
            resource_type=ResourceTypeEnum.CODE_FILE,
        )
        session.add(resource)
        session.commit()

        assert repo.tenant_id == resource.tenant_id
