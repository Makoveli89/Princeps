"""
Supabase/pgvector Integration Module

Database integration layer for the brain layer, providing:
- Supabase client wrapper for REST operations
- Postgres utilities with connection pooling
- pgvector operations for similarity search
- Alembic migrations for schema management

Components:
- SupabaseClient: REST API wrapper for Supabase
- PostgresConnectionPool: psycopg2 connection pooling
- SQLAlchemyEngine: ORM integration
- PgVectorUtils: Vector similarity search utilities

Created: December 26, 2024
"""

from .supabase_client import (
    SupabaseClient,
    SupabaseConfig,
    create_supabase_client,
    SUPABASE_AVAILABLE,
)

from .pg_utils import (
    PostgresConnectionPool,
    PostgresConfig,
    SQLAlchemyEngine,
    PgVectorUtils,
    create_postgres_pool,
    create_sqlalchemy_engine,
    PSYCOPG2_AVAILABLE,
    SQLALCHEMY_AVAILABLE,
)

__all__ = [
    # Supabase
    "SupabaseClient",
    "SupabaseConfig",
    "create_supabase_client",
    "SUPABASE_AVAILABLE",
    # Postgres
    "PostgresConnectionPool",
    "PostgresConfig",
    "SQLAlchemyEngine",
    "PgVectorUtils",
    "create_postgres_pool",
    "create_sqlalchemy_engine",
    "PSYCOPG2_AVAILABLE",
    "SQLALCHEMY_AVAILABLE",
]
