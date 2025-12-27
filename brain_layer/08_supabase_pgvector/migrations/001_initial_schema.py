"""Initial brain layer schema

Revision ID: 001_initial
Revises: 
Create Date: 2024-12-26

Creates the complete brain layer schema including:
- Documents and chunks with pgvector embeddings
- Knowledge atoms and network
- Agent runs and decision logging
- Entity extraction tables
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")  # For text search
    
    # ==================== Documents ====================
    op.create_table(
        'documents',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('source', sa.String(1000)),
        sa.Column('source_type', sa.String(50)),  # 'pdf', 'web', 'code', 'manual'
        sa.Column('file_path', sa.String(1000)),
        sa.Column('file_hash', sa.String(64)),  # SHA256
        sa.Column('page_count', sa.Integer),
        sa.Column('word_count', sa.Integer),
        sa.Column('token_count', sa.Integer),
        sa.Column('tags', ARRAY(sa.String)),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.create_index('idx_documents_source_type', 'documents', ['source_type'])
    op.create_index('idx_documents_created_at', 'documents', ['created_at'])
    op.create_index('idx_documents_tags', 'documents', ['tags'], postgresql_using='gin')
    
    # ==================== Document Chunks ====================
    op.create_table(
        'document_chunks',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', UUID(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer, nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('token_count', sa.Integer),
        sa.Column('start_char', sa.Integer),
        sa.Column('end_char', sa.Integer),
        sa.Column('page_number', sa.Integer),
        sa.Column('embedding', sa.dialects.postgresql.ARRAY(sa.Float), nullable=True),  # Will be vector(384)
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    # Convert embedding to vector type after table creation
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)")
    
    op.create_index('idx_chunks_document', 'document_chunks', ['document_id'])
    op.create_index('idx_chunks_embedding', 'document_chunks', ['embedding'], postgresql_using='ivfflat',
                    postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    
    # ==================== Knowledge Atoms ====================
    op.create_table(
        'knowledge_atoms',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('knowledge_type', sa.String(50), nullable=False),  # 'fact', 'concept', 'procedure', 'insight'
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('source_chunk_id', UUID(), sa.ForeignKey('document_chunks.id', ondelete='SET NULL')),
        sa.Column('source_document_id', UUID(), sa.ForeignKey('documents.id', ondelete='SET NULL')),
        sa.Column('confidence', sa.Float, default=1.0),
        sa.Column('usage_count', sa.Integer, default=0),
        sa.Column('priority_score', sa.Float, default=0.0),
        sa.Column('is_promoted', sa.Boolean, default=False),
        sa.Column('embedding', sa.dialects.postgresql.ARRAY(sa.Float)),
        sa.Column('tags', ARRAY(sa.String)),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.execute("ALTER TABLE knowledge_atoms ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)")
    
    op.create_index('idx_atoms_type', 'knowledge_atoms', ['knowledge_type'])
    op.create_index('idx_atoms_promoted', 'knowledge_atoms', ['is_promoted'])
    op.create_index('idx_atoms_priority', 'knowledge_atoms', ['priority_score'])
    
    # ==================== Knowledge Relations ====================
    op.create_table(
        'knowledge_relations',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('source_id', UUID(), sa.ForeignKey('knowledge_atoms.id', ondelete='CASCADE'), nullable=False),
        sa.Column('target_id', UUID(), sa.ForeignKey('knowledge_atoms.id', ondelete='CASCADE'), nullable=False),
        sa.Column('relation_type', sa.String(50), nullable=False),  # 'supports', 'contradicts', 'builds_on', 'related'
        sa.Column('strength', sa.Float, default=1.0),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.create_index('idx_relations_source', 'knowledge_relations', ['source_id'])
    op.create_index('idx_relations_target', 'knowledge_relations', ['target_id'])
    op.create_index('idx_relations_type', 'knowledge_relations', ['relation_type'])
    
    # ==================== Document Summaries ====================
    op.create_table(
        'document_summaries',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', UUID(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('summary_type', sa.String(50), nullable=False),  # 'one_sentence', 'executive', 'detailed'
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('word_count', sa.Integer),
        sa.Column('model_used', sa.String(100)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.create_index('idx_summaries_document', 'document_summaries', ['document_id'])
    op.create_index('idx_summaries_type', 'document_summaries', ['summary_type'])
    
    # ==================== Document Entities ====================
    op.create_table(
        'document_entities',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', UUID(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('entity_text', sa.String(500), nullable=False),
        sa.Column('entity_type', sa.String(50), nullable=False),  # 'PERSON', 'ORG', 'LOCATION', etc.
        sa.Column('start_char', sa.Integer),
        sa.Column('end_char', sa.Integer),
        sa.Column('confidence', sa.Float),
        sa.Column('frequency', sa.Integer, default=1),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.create_index('idx_entities_document', 'document_entities', ['document_id'])
    op.create_index('idx_entities_type', 'document_entities', ['entity_type'])
    op.create_index('idx_entities_text', 'document_entities', ['entity_text'])
    
    # ==================== Document Concepts ====================
    op.create_table(
        'document_concepts',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', UUID(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('concept', sa.String(500), nullable=False),
        sa.Column('relevance_score', sa.Float, nullable=False),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.create_index('idx_concepts_document', 'document_concepts', ['document_id'])
    op.create_index('idx_concepts_relevance', 'document_concepts', ['relevance_score'])
    
    # ==================== Document Topics ====================
    op.create_table(
        'document_topics',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', UUID(), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('topic_label', sa.String(200)),
        sa.Column('topic_keywords', ARRAY(sa.String)),
        sa.Column('confidence', sa.Float),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.create_index('idx_topics_document', 'document_topics', ['document_id'])
    
    # ==================== Agent Runs ====================
    op.create_table(
        'agent_runs',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', sa.String(100), nullable=False),
        sa.Column('task', sa.Text, nullable=False),
        sa.Column('task_hash', sa.String(64)),
        sa.Column('success', sa.Boolean, nullable=False),
        sa.Column('score', sa.Float),
        sa.Column('solution', JSONB),
        sa.Column('feedback', sa.Text),
        sa.Column('iteration', sa.Integer, default=1),
        sa.Column('latency_ms', sa.Float),
        sa.Column('model_version', sa.String(50)),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.create_index('idx_runs_agent', 'agent_runs', ['agent_id'])
    op.create_index('idx_runs_task_hash', 'agent_runs', ['task_hash'])
    op.create_index('idx_runs_success', 'agent_runs', ['success'])
    op.create_index('idx_runs_created_at', 'agent_runs', ['created_at'])
    
    # ==================== Model Versions ====================
    op.create_table(
        'model_versions',
        sa.Column('id', UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('model_id', sa.String(100), nullable=False),
        sa.Column('version', sa.Integer, nullable=False),
        sa.Column('agent_id', sa.String(100)),
        sa.Column('accuracy', sa.Float),
        sa.Column('precision', sa.Float),
        sa.Column('recall', sa.Float),
        sa.Column('f1_score', sa.Float),
        sa.Column('training_samples', sa.Integer),
        sa.Column('is_active', sa.Boolean, default=False),
        sa.Column('model_path', sa.String(1000)),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    op.create_index('idx_models_model_id', 'model_versions', ['model_id'])
    op.create_index('idx_models_active', 'model_versions', ['is_active'])
    
    # ==================== Create vector search function ====================
    op.execute("""
        CREATE OR REPLACE FUNCTION match_document_chunks(
            query_embedding vector(384),
            match_count int DEFAULT 10,
            filter_document_id uuid DEFAULT NULL
        )
        RETURNS TABLE (
            id uuid,
            document_id uuid,
            content text,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                dc.id,
                dc.document_id,
                dc.content,
                1 - (dc.embedding <=> query_embedding) as similarity
            FROM document_chunks dc
            WHERE 
                dc.embedding IS NOT NULL
                AND (filter_document_id IS NULL OR dc.document_id = filter_document_id)
            ORDER BY dc.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
    """)
    
    op.execute("""
        CREATE OR REPLACE FUNCTION match_knowledge_atoms(
            query_embedding vector(384),
            match_count int DEFAULT 10,
            match_threshold float DEFAULT 0.0
        )
        RETURNS TABLE (
            id uuid,
            title text,
            content text,
            knowledge_type text,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                ka.id,
                ka.title,
                ka.content,
                ka.knowledge_type,
                1 - (ka.embedding <=> query_embedding) as similarity
            FROM knowledge_atoms ka
            WHERE 
                ka.embedding IS NOT NULL
                AND 1 - (ka.embedding <=> query_embedding) > match_threshold
            ORDER BY ka.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
    """)


def downgrade() -> None:
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS match_knowledge_atoms")
    op.execute("DROP FUNCTION IF EXISTS match_document_chunks")
    
    # Drop tables in reverse order
    op.drop_table('model_versions')
    op.drop_table('agent_runs')
    op.drop_table('document_topics')
    op.drop_table('document_concepts')
    op.drop_table('document_entities')
    op.drop_table('document_summaries')
    op.drop_table('knowledge_relations')
    op.drop_table('knowledge_atoms')
    op.drop_table('document_chunks')
    op.drop_table('documents')
