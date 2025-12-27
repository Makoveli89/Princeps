"""
Postgres Utilities - Connection pooling, health checks, and helpers

Provides direct Postgres access for operations not suited to Supabase client,
including pgvector operations, connection pooling, and raw SQL execution.

Created: December 26, 2024
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Type hints only - for runtime optional imports
if TYPE_CHECKING:
    from psycopg2 import pool as psycopg2_pool

# Attempt imports - graceful fallback
pool = None
RealDictCursor = None
execute_values = None

try:
    import psycopg2 as _psycopg2  # noqa: F401
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor, execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not installed. Install with: pip install psycopg2-binary")

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not installed. Install with: pip install sqlalchemy")


@dataclass
class PostgresConfig:
    """Configuration for Postgres connection"""
    
    host: str
    port: int
    database: str
    user: str
    password: str
    
    # Connection pool settings
    pool_min_size: int = 1
    pool_max_size: int = 10
    
    # SSL settings
    ssl_mode: str = "prefer"
    
    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Load configuration from environment variables"""
        # Support both individual vars and DATABASE_URL
        database_url = os.getenv("DATABASE_URL")
        
        if database_url:
            # Parse DATABASE_URL
            # Format: postgresql://user:password@host:port/database
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            
            return cls(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") or "postgres",
                user=parsed.username or "postgres",
                password=parsed.password or "",
            )
        
        # Fall back to individual environment variables
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "postgres"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
    
    @property
    def connection_string(self) -> str:
        """Get psycopg2 connection string"""
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password} sslmode={self.ssl_mode}"
        )
    
    @property
    def sqlalchemy_url(self) -> str:
        """Get SQLAlchemy connection URL"""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class PostgresConnectionPool:
    """
    Postgres Connection Pool using psycopg2
    
    Provides connection pooling for high-throughput scenarios.
    """
    
    def __init__(self, config: PostgresConfig):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not installed")
            
        self.config = config
        self._pool: Optional[psycopg2_pool.ThreadedConnectionPool] = None

    @property
    def pool(self) -> psycopg2_pool.ThreadedConnectionPool:
        """Lazy-initialize connection pool"""
        if self._pool is None:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.config.pool_min_size,
                maxconn=self.config.pool_max_size,
                dsn=self.config.connection_string,
            )
            logger.info(f"Connection pool initialized: {self.config.pool_max_size} max connections")
        return self._pool
    
    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a connection from the pool
        
        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = True) -> Generator:
        """
        Get a cursor with automatic connection management
        
        Args:
            dict_cursor: Use RealDictCursor for dict results
            
        Usage:
            with pool.get_cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                rows = cursor.fetchall()
        """
        cursor_factory = RealDictCursor if dict_cursor else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def execute(
        self, 
        query: str, 
        params: Optional[Tuple] = None,
        fetch: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a query and optionally fetch results
        
        Args:
            query: SQL query
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            List of rows if fetch=True
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            
            if fetch:
                return [dict(row) for row in cursor.fetchall()]
            return None
    
    def execute_many(
        self,
        query: str,
        data: List[Tuple]
    ) -> int:
        """
        Execute a query with multiple parameter sets
        
        Args:
            query: SQL query with placeholders
            data: List of parameter tuples
            
        Returns:
            Number of rows affected
        """
        with self.get_cursor(dict_cursor=False) as cursor:
            cursor.executemany(query, data)
            return cursor.rowcount
    
    def execute_values(
        self,
        query: str,
        data: List[Tuple],
        template: Optional[str] = None,
        page_size: int = 1000
    ) -> int:
        """
        Efficient bulk insert using execute_values
        
        Args:
            query: INSERT query with VALUES placeholder
            data: List of value tuples
            template: Value template (optional)
            page_size: Batch size
            
        Returns:
            Number of rows inserted
        """
        with self.get_cursor(dict_cursor=False) as cursor:
            execute_values(
                cursor, 
                query, 
                data, 
                template=template,
                page_size=page_size
            )
            return cursor.rowcount
    
    def health_check(self) -> Dict[str, Any]:
        """Check database connection health"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1 as health, now() as timestamp")
                result = cursor.fetchone()
                
                return {
                    "status": "healthy",
                    "timestamp": result["timestamp"].isoformat(),
                    "pool_size": self.config.pool_max_size,
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    def close(self):
        """Close all connections in the pool"""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Connection pool closed")


class SQLAlchemyEngine:
    """
    SQLAlchemy Engine wrapper for ORM operations
    
    Provides session management and integration with SQLAlchemy models.
    """
    
    def __init__(self, config: PostgresConfig):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy not installed")
            
        self.config = config
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self):
        """Lazy-initialize SQLAlchemy engine"""
        if self._engine is None:
            self._engine = create_engine(
                self.config.sqlalchemy_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_min_size,
                max_overflow=self.config.pool_max_size - self.config.pool_min_size,
                pool_pre_ping=True,  # Verify connections are alive
            )
            logger.info(f"SQLAlchemy engine initialized: {self.config.host}")
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Get a session with automatic commit/rollback
        
        Usage:
            with engine.session() as session:
                session.add(model)
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute raw SQL"""
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result
    
    def health_check(self) -> Dict[str, Any]:
        """Check database connection health"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as health, now() as timestamp"))
                row = result.fetchone()
                
                return {
                    "status": "healthy",
                    "timestamp": str(row[1]),
                    "backend": "sqlalchemy",
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    def dispose(self):
        """Dispose engine and connections"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("SQLAlchemy engine disposed")


# ==================== pgvector Utilities ====================

class PgVectorUtils:
    """
    Utilities for pgvector operations
    
    Provides helpers for vector similarity search and index management.
    """
    
    def __init__(self, pool: PostgresConnectionPool):
        self.pool = pool
    
    def check_extension(self) -> bool:
        """Check if pgvector extension is installed"""
        result = self.pool.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
        return len(result) > 0
    
    def create_extension(self) -> None:
        """Create pgvector extension if not exists"""
        self.pool.execute(
            "CREATE EXTENSION IF NOT EXISTS vector",
            fetch=False
        )
        logger.info("pgvector extension created/verified")
    
    def create_vector_index(
        self,
        table: str,
        column: str = "embedding",
        index_name: Optional[str] = None,
        method: str = "ivfflat",  # or "hnsw"
        lists: int = 100,  # for ivfflat
        m: int = 16,  # for hnsw
        ef_construction: int = 64  # for hnsw
    ) -> None:
        """
        Create a vector index for similarity search
        
        Args:
            table: Table name
            column: Vector column name
            index_name: Custom index name
            method: Index method (ivfflat or hnsw)
            lists: Number of lists for ivfflat
            m: M parameter for hnsw
            ef_construction: ef_construction for hnsw
        """
        index_name = index_name or f"idx_{table}_{column}_vector"
        
        if method == "ivfflat":
            query = f"""
                CREATE INDEX IF NOT EXISTS {index_name} 
                ON {table} 
                USING ivfflat ({column} vector_cosine_ops)
                WITH (lists = {lists})
            """
        elif method == "hnsw":
            query = f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table}
                USING hnsw ({column} vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction})
            """
        else:
            raise ValueError(f"Unknown index method: {method}")
        
        self.pool.execute(query, fetch=False)
        logger.info(f"Created {method} index: {index_name}")
    
    def similarity_search(
        self,
        table: str,
        query_embedding: List[float],
        embedding_column: str = "embedding",
        select_columns: List[str] = ["*"],
        top_k: int = 10,
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search
        
        Args:
            table: Table name
            query_embedding: Query vector
            embedding_column: Vector column name
            select_columns: Columns to return
            top_k: Number of results
            filters: Additional WHERE clause (without WHERE)
            
        Returns:
            List of matching rows with similarity score
        """
        columns = ", ".join(select_columns)
        
        query = f"""
            SELECT {columns},
                   1 - ({embedding_column} <=> %s::vector) as similarity
            FROM {table}
            {f"WHERE {filters}" if filters else ""}
            ORDER BY {embedding_column} <=> %s::vector
            LIMIT %s
        """
        
        # Convert embedding to string format for pgvector
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        
        return self.pool.execute(query, (embedding_str, embedding_str, top_k))
    
    def insert_with_embedding(
        self,
        table: str,
        data: Dict[str, Any],
        embedding: List[float],
        embedding_column: str = "embedding"
    ) -> Optional[Dict[str, Any]]:
        """
        Insert a row with an embedding vector
        
        Args:
            table: Table name
            data: Row data (without embedding)
            embedding: Embedding vector
            embedding_column: Vector column name
            
        Returns:
            Inserted row
        """
        # Add embedding to data
        data[embedding_column] = f"[{','.join(map(str, embedding))}]"
        
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        
        query = f"""
            INSERT INTO {table} ({columns})
            VALUES ({placeholders})
            RETURNING *
        """
        
        result = self.pool.execute(query, tuple(data.values()))
        return result[0] if result else None
    
    def bulk_insert_with_embeddings(
        self,
        table: str,
        rows: List[Dict[str, Any]],
        embeddings: List[List[float]],
        embedding_column: str = "embedding"
    ) -> int:
        """
        Bulk insert rows with embeddings
        
        Args:
            table: Table name
            rows: List of row data
            embeddings: List of embedding vectors
            embedding_column: Vector column name
            
        Returns:
            Number of rows inserted
        """
        if len(rows) != len(embeddings):
            raise ValueError("rows and embeddings must have same length")
        
        if not rows:
            return 0
        
        # Add embeddings to rows
        for row, embedding in zip(rows, embeddings):
            row[embedding_column] = f"[{','.join(map(str, embedding))}]"
        
        columns = list(rows[0].keys())
        columns_str = ", ".join(columns)
        
        query = f"INSERT INTO {table} ({columns_str}) VALUES %s"
        data = [tuple(row[c] for c in columns) for row in rows]
        
        return self.pool.execute_values(query, data)


# ==================== Factory Functions ====================

def create_postgres_pool(
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> PostgresConnectionPool:
    """
    Create a Postgres connection pool
    
    Args:
        Connection parameters (or from environment)
        
    Returns:
        Configured PostgresConnectionPool
    """
    if all([host, database, user, password]):
        config = PostgresConfig(
            host=host,
            port=port or 5432,
            database=database,
            user=user,
            password=password,
        )
    else:
        config = PostgresConfig.from_env()
    
    return PostgresConnectionPool(config)


def create_sqlalchemy_engine(
    url: Optional[str] = None
) -> SQLAlchemyEngine:
    """
    Create a SQLAlchemy engine
    
    Args:
        url: Database URL (or from environment)
        
    Returns:
        Configured SQLAlchemyEngine
    """
    if url:
        # Parse URL into config
        from urllib.parse import urlparse
        parsed = urlparse(url)
        config = PostgresConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/"),
            user=parsed.username or "postgres",
            password=parsed.password or "",
        )
    else:
        config = PostgresConfig.from_env()
    
    return SQLAlchemyEngine(config)


# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("POSTGRES UTILITIES - DEMO")
    print("=" * 60)
    
    print("\nAvailable backends:")
    print(f"  psycopg2: {'✅' if PSYCOPG2_AVAILABLE else '❌'}")
    print(f"  SQLAlchemy: {'✅' if SQLALCHEMY_AVAILABLE else '❌'}")
    
    print("\nTo use:")
    print("  1. Set DATABASE_URL or individual POSTGRES_* env vars")
    print("  2. pool = create_postgres_pool()")
    print("  3. pool.execute('SELECT * FROM table')")
    
    print("\nFor pgvector:")
    print("  utils = PgVectorUtils(pool)")
    print("  utils.similarity_search('chunks', embedding, top_k=10)")
