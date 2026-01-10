"""add_vector_indexes

Revision ID: a28eaabd94aa
Revises: edec7db70b5d
Create Date: 2025-05-23 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a28eaabd94aa"
down_revision: str | None = "edec7db70b5d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add HNSW indexes to vector columns if not present."""
    # Ensure pgvector extension exists
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Add index to doc_chunks.embedding
    # Using 'hnsw' (Hierarchical Navigable Small World) which is better for incremental updates
    # and doesn't require a training step like ivfflat.
    # Using 'vector_cosine_ops' for cosine distance.
    # m=16 and ef_construction=64 are reasonable defaults.
    op.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_chunk_embedding_hnsw
        ON doc_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """))

    # Add index to knowledge_nodes.embedding
    op.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_knode_embedding_hnsw
        ON knowledge_nodes
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """))


def downgrade() -> None:
    """Remove the indexes."""
    op.execute(sa.text("DROP INDEX IF EXISTS idx_chunk_embedding_hnsw"))
    op.execute(sa.text("DROP INDEX IF EXISTS idx_knode_embedding_hnsw"))
