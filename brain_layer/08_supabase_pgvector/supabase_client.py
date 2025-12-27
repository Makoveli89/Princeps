"""
Supabase Client - Wrapper for Supabase Python client

Provides a clean interface for Supabase operations with connection pooling,
authentication, and storage integration. Designed for use with the brain layer.

Created: December 26, 2024
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Attempt to import supabase - graceful fallback if not installed
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None
    logger.warning("supabase-py not installed. Install with: pip install supabase")


@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection"""
    
    url: str
    key: str  # anon key for client-side, service_role key for server-side
    schema: str = "public"
    auto_refresh_token: bool = True
    persist_session: bool = True
    
    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        """Load configuration from environment variables"""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables required"
            )
        
        return cls(url=url, key=key)


class SupabaseClient:
    """
    Supabase Client Wrapper
    
    Features:
    - Connection management with lazy initialization
    - Table operations (CRUD)
    - Vector operations via pgvector
    - Storage bucket operations
    - RPC function calls
    - Realtime subscriptions (if enabled)
    
    Usage:
        client = SupabaseClient.from_env()
        
        # Insert document
        client.insert("documents", {"title": "...", "content": "..."})
        
        # Vector search
        results = client.vector_search("chunks", query_embedding, top_k=10)
    """
    
    def __init__(self, config: SupabaseConfig):
        self.config = config
        self._client: Optional[Client] = None
        
    @classmethod
    def from_env(cls) -> "SupabaseClient":
        """Create client from environment variables"""
        config = SupabaseConfig.from_env()
        return cls(config)
    
    @property
    def client(self) -> Client:
        """Lazy-initialize Supabase client"""
        if not SUPABASE_AVAILABLE:
            raise ImportError("supabase-py not installed")
            
        if self._client is None:
            self._client = create_client(
                self.config.url,
                self.config.key,
            )
            logger.info(f"Connected to Supabase: {self.config.url}")
            
        return self._client
    
    def health_check(self) -> Dict[str, Any]:
        """Check connection health"""
        try:
            # Simple query to verify connection
            self.client.table("_health_check").select("*").limit(1).execute()
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            # Table might not exist, but connection worked
            if "does not exist" in str(e):
                return {"status": "healthy", "note": "connected", "timestamp": datetime.now().isoformat()}
            return {"status": "unhealthy", "error": str(e)}
    
    # ==================== Table Operations ====================
    
    def insert(
        self, 
        table: str, 
        data: Dict[str, Any], 
        return_data: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Insert a row into a table
        
        Args:
            table: Table name
            data: Row data as dict
            return_data: Whether to return the inserted row
            
        Returns:
            Inserted row if return_data=True
        """
        try:
            query = self.client.table(table).insert(data)
            result = query.execute()
            
            if return_data and result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Insert failed on {table}: {e}")
            raise
    
    def insert_many(
        self, 
        table: str, 
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Insert multiple rows into a table
        
        Args:
            table: Table name
            data: List of row data
            
        Returns:
            List of inserted rows
        """
        try:
            result = self.client.table(table).insert(data).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Bulk insert failed on {table}: {e}")
            raise
    
    def upsert(
        self, 
        table: str, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        on_conflict: str = "id"
    ) -> List[Dict[str, Any]]:
        """
        Upsert (insert or update) rows
        
        Args:
            table: Table name
            data: Row(s) to upsert
            on_conflict: Column to detect conflicts
            
        Returns:
            Upserted rows
        """
        try:
            if isinstance(data, dict):
                data = [data]
                
            result = self.client.table(table).upsert(
                data, 
                on_conflict=on_conflict
            ).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Upsert failed on {table}: {e}")
            raise
    
    def select(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Select rows from a table
        
        Args:
            table: Table name
            columns: Columns to select (comma-separated or *)
            filters: Dict of column=value filters (eq only)
            order_by: Column to order by (prefix with - for desc)
            limit: Max rows to return
            offset: Number of rows to skip
            
        Returns:
            List of matching rows
        """
        try:
            query = self.client.table(table).select(columns)
            
            # Apply filters
            if filters:
                for column, value in filters.items():
                    query = query.eq(column, value)
            
            # Apply ordering
            if order_by:
                desc = order_by.startswith("-")
                col = order_by.lstrip("-")
                query = query.order(col, desc=desc)
            
            # Apply pagination
            if limit:
                query = query.limit(limit)
            if offset:
                query = query.offset(offset)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Select failed on {table}: {e}")
            raise
    
    def select_by_id(
        self, 
        table: str, 
        id_value: Any,
        id_column: str = "id"
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single row by ID
        
        Args:
            table: Table name
            id_value: ID value
            id_column: Name of ID column
            
        Returns:
            Row if found, None otherwise
        """
        try:
            result = self.client.table(table).select("*").eq(id_column, id_value).limit(1).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Select by ID failed on {table}: {e}")
            raise
    
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Update rows matching filters
        
        Args:
            table: Table name
            data: Fields to update
            filters: Dict of column=value filters
            
        Returns:
            Updated rows
        """
        try:
            query = self.client.table(table).update(data)
            
            for column, value in filters.items():
                query = query.eq(column, value)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Update failed on {table}: {e}")
            raise
    
    def delete(
        self,
        table: str,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Delete rows matching filters
        
        Args:
            table: Table name
            filters: Dict of column=value filters
            
        Returns:
            Deleted rows
        """
        try:
            query = self.client.table(table).delete()
            
            for column, value in filters.items():
                query = query.eq(column, value)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Delete failed on {table}: {e}")
            raise
    
    # ==================== Vector Operations ====================
    
    def vector_search(
        self,
        table: str,
        query_embedding: List[float],
        embedding_column: str = "embedding",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using pgvector
        
        Requires RPC function 'match_{table}' to be defined in Supabase.
        
        Args:
            table: Table name
            query_embedding: Query vector
            embedding_column: Name of embedding column
            top_k: Number of results
            filters: Additional filters (table-specific)
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching rows with similarity scores
        """
        try:
            # Call RPC function for vector search
            # Function should be created in Supabase:
            # CREATE FUNCTION match_chunks(query_embedding vector(384), match_count int)
            # RETURNS TABLE (id uuid, content text, similarity float)
            
            rpc_name = f"match_{table}"
            params = {
                "query_embedding": query_embedding,
                "match_count": top_k,
            }
            
            if threshold:
                params["match_threshold"] = threshold
            
            if filters:
                params.update(filters)
            
            result = self.client.rpc(rpc_name, params).execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Vector search failed on {table}: {e}")
            raise
    
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
            data: Row data
            embedding: Embedding vector
            embedding_column: Name of embedding column
            
        Returns:
            Inserted row
        """
        data[embedding_column] = embedding
        return self.insert(table, data)
    
    # ==================== RPC Functions ====================
    
    def rpc(
        self,
        function_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Call a Supabase RPC function
        
        Args:
            function_name: Name of the function
            params: Function parameters
            
        Returns:
            Function result
        """
        try:
            result = self.client.rpc(function_name, params or {}).execute()
            return result.data
            
        except Exception as e:
            logger.error(f"RPC call failed for {function_name}: {e}")
            raise
    
    # ==================== Storage Operations ====================
    
    def upload_file(
        self,
        bucket: str,
        path: str,
        file_content: bytes,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a file to Supabase Storage
        
        Args:
            bucket: Bucket name
            path: File path within bucket
            file_content: File bytes
            content_type: MIME type
            
        Returns:
            Public URL of uploaded file
        """
        try:
            self.client.storage.from_(bucket).upload(
                path,
                file_content,
                {"content-type": content_type}
            )

            # Get public URL
            url = self.client.storage.from_(bucket).get_public_url(path)
            return url
            
        except Exception as e:
            logger.error(f"Upload failed to {bucket}/{path}: {e}")
            raise
    
    def download_file(self, bucket: str, path: str) -> bytes:
        """
        Download a file from Supabase Storage
        
        Args:
            bucket: Bucket name
            path: File path within bucket
            
        Returns:
            File content as bytes
        """
        try:
            result = self.client.storage.from_(bucket).download(path)
            return result
            
        except Exception as e:
            logger.error(f"Download failed from {bucket}/{path}: {e}")
            raise
    
    def delete_file(self, bucket: str, paths: List[str]) -> None:
        """
        Delete files from Supabase Storage
        
        Args:
            bucket: Bucket name
            paths: List of file paths to delete
        """
        try:
            self.client.storage.from_(bucket).remove(paths)
            
        except Exception as e:
            logger.error(f"Delete failed from {bucket}: {e}")
            raise


# Factory function
def create_supabase_client(
    url: Optional[str] = None,
    key: Optional[str] = None
) -> SupabaseClient:
    """
    Create a Supabase client
    
    Args:
        url: Supabase project URL (or from env)
        key: Supabase API key (or from env)
        
    Returns:
        Configured SupabaseClient
    """
    if url and key:
        config = SupabaseConfig(url=url, key=key)
    else:
        config = SupabaseConfig.from_env()
    
    return SupabaseClient(config)


# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("SUPABASE CLIENT - DEMO")
    print("=" * 60)
    
    if not SUPABASE_AVAILABLE:
        print("\n⚠️  supabase-py not installed")
        print("Install with: pip install supabase")
    else:
        print("\n✅ supabase-py available")
        print("\nTo use:")
        print("  1. Set SUPABASE_URL and SUPABASE_KEY environment variables")
        print("  2. client = SupabaseClient.from_env()")
        print("  3. client.insert('table', {'column': 'value'})")
