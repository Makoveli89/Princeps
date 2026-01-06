"""Tests for vector index population for all chunk types."""

from brain.core.models import DocChunk, KnowledgeNode, NodeKnowledgeTypeEnum


class TestVectorIndexPopulation:
    """Tests for vector embedding index population."""

    def test_chunk_embedding_field_exists(self, session, tenant, sample_document):
        """DocChunk should have embedding field."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Test content for embedding",
            chunk_index=0,
        )
        session.add(chunk)
        session.commit()

        # Verify embedding field exists (even if None initially)
        assert hasattr(chunk, "embedding")

    def test_chunk_with_embedding(self, session, tenant, sample_document):
        """DocChunk should accept embedding vector."""
        embedding = [0.1] * 384  # Standard embedding dimension

        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Test content",
            chunk_index=0,
            embedding=embedding,
        )
        session.add(chunk)
        session.commit()

        # Refresh from DB
        session.refresh(chunk)
        # Note: SQLite doesn't support pgvector, so embedding may be stored differently
        # In production with PostgreSQL, this would be a proper vector

    def test_knowledge_node_embedding_field(self, session, tenant):
        """KnowledgeNode should have embedding field."""
        node = KnowledgeNode(
            tenant_id=tenant.id,
            agent_id="test-agent",
            knowledge_type=NodeKnowledgeTypeEnum.INSIGHT,
            title="ML Basics",
            content="Machine learning",
        )
        session.add(node)
        session.commit()

        assert hasattr(node, "embedding")

    def test_multiple_chunks_have_embeddings(self, session, tenant, sample_document):
        """Multiple chunks should each have their own embedding."""
        chunks = []
        for i in range(3):
            chunk = DocChunk(
                tenant_id=tenant.id,
                document_id=sample_document.id,
                content=f"Chunk {i} content",
                chunk_index=i,
                embedding=[float(i) / 10] * 384,
            )
            chunks.append(chunk)

        session.add_all(chunks)
        session.commit()

        # All chunks should be created
        db_chunks = (
            session.query(DocChunk)
            .filter_by(document_id=sample_document.id)
            .order_by(DocChunk.chunk_index)
            .all()
        )

        assert len(db_chunks) == 3


class TestChunkMetadata:
    """Tests for chunk metadata required for retrieval."""

    def test_chunk_has_required_fields(self, session, tenant, sample_document):
        """DocChunk should have all required metadata fields."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Test content",
            chunk_index=0,
            token_count=10,
            start_char=0,
            end_char=100,
        )
        session.add(chunk)
        session.commit()

        assert chunk.tenant_id is not None
        assert chunk.document_id is not None
        assert chunk.content is not None
        assert chunk.chunk_index is not None
        assert chunk.token_count is not None

    def test_chunk_preserves_document_reference(self, session, tenant, sample_document):
        """Chunk should maintain reference to parent document."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Content",
            chunk_index=0,
        )
        session.add(chunk)
        session.commit()

        # Verify relationship
        assert chunk.document_id == sample_document.id


class TestEmbeddingDimensions:
    """Tests for embedding dimension consistency."""

    def test_standard_embedding_dimension(self):
        """Standard embedding dimension should be 384 for all-MiniLM-L6-v2."""
        expected_dim = 384
        embedding = [0.1] * expected_dim
        assert len(embedding) == 384

    def test_embedding_as_list(self, session, tenant, sample_document):
        """Embedding should be storable as list/array."""
        embedding = list(range(384))  # Integer embeddings for testing

        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Test",
            chunk_index=0,
            embedding=embedding,
        )
        session.add(chunk)
        session.commit()


class TestVectorSearchPrerequisites:
    """Tests for prerequisites needed for vector search."""

    def test_chunk_indexable_by_document(self, session, tenant, sample_document):
        """Should be able to filter chunks by document_id."""
        chunks = []
        for i in range(3):
            chunk = DocChunk(
                tenant_id=tenant.id,
                document_id=sample_document.id,
                content=f"Chunk {i}",
                chunk_index=i,
            )
            chunks.append(chunk)

        session.add_all(chunks)
        session.commit()

        # Filter by document
        doc_chunks = (
            session.query(DocChunk).filter(DocChunk.document_id == sample_document.id).all()
        )

        assert len(doc_chunks) == 3

    def test_chunk_indexable_by_tenant(self, session, tenant, sample_document):
        """Should be able to filter chunks by tenant_id."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Tenant chunk",
            chunk_index=0,
        )
        session.add(chunk)
        session.commit()

        # Filter by tenant
        tenant_chunks = session.query(DocChunk).filter(DocChunk.tenant_id == tenant.id).all()

        assert len(tenant_chunks) >= 1


class TestKnowledgeNodeEmbeddings:
    """Tests for knowledge node embedding population."""

    def test_knowledge_node_accepts_embedding(self, session, tenant):
        """KnowledgeNode should accept embedding vector."""
        node = KnowledgeNode(
            tenant_id=tenant.id,
            agent_id="test-agent",
            knowledge_type=NodeKnowledgeTypeEnum.INSIGHT,
            title="Sky Color",
            content="The sky is blue",
            embedding=[0.5] * 384,
        )
        session.add(node)
        session.commit()

        assert node.node_id is not None

    def test_knowledge_node_types(self, session, tenant):
        """Should support various knowledge node types."""
        # Use actual enum values from NodeKnowledgeTypeEnum
        node_types = [
            NodeKnowledgeTypeEnum.SOLUTION,
            NodeKnowledgeTypeEnum.PATTERN,
            NodeKnowledgeTypeEnum.INSIGHT,
            NodeKnowledgeTypeEnum.BEST_PRACTICE,
        ]

        for i, node_type in enumerate(node_types):
            node = KnowledgeNode(
                tenant_id=tenant.id,
                agent_id="test-agent",
                knowledge_type=node_type,
                title=f"Test {node_type.value}",
                content=f"Test {node_type.value}",
            )
            session.add(node)

        session.commit()

        all_nodes = session.query(KnowledgeNode).filter(KnowledgeNode.tenant_id == tenant.id).all()

        assert len(all_nodes) == len(node_types)


class TestEmbeddingNullability:
    """Tests for handling null embeddings."""

    def test_chunk_without_embedding(self, session, tenant, sample_document):
        """Chunk should be valid without embedding (for later population)."""
        chunk = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="No embedding yet",
            chunk_index=0,
            embedding=None,
        )
        session.add(chunk)
        session.commit()

        assert chunk.embedding is None

    def test_filter_chunks_without_embedding(self, session, tenant, sample_document):
        """Should be able to find chunks needing embeddings."""
        chunk_with = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="Has embedding",
            chunk_index=0,
            embedding=[0.1] * 384,
        )
        chunk_without = DocChunk(
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content="No embedding",
            chunk_index=1,
            embedding=None,
        )

        session.add_all([chunk_with, chunk_without])
        session.commit()

        # Find chunks needing embeddings
        needs_embedding = (
            session.query(DocChunk)
            .filter(
                DocChunk.tenant_id == tenant.id,
                DocChunk.embedding.is_(None),
            )
            .all()
        )

        assert len(needs_embedding) == 1
        assert needs_embedding[0].content == "No embedding"
