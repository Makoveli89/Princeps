"""
Ingestion Service - Handles document ingestion, chunking, and storage.
"""

import hashlib
import logging
import os
import uuid
from typing import Any, BinaryIO

from brain.core.db import get_session
from brain.core.models import (
    DocChunk,
    Document,
    KnowledgeTypeEnum,
    Operation,
    OperationStatusEnum,
    OperationTypeEnum,
    Resource,
    ResourceTypeEnum,
    SecurityLevelEnum,
)
from framework.retrieval.vector_search import (
    PgVectorIndex,
    get_embedding_service,
)

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        # Initialize pgvector index
        # Note: Connection string usually from env or config.
        # For this setup we rely on environment variables used by the brain module.
        db_url = os.getenv("DATABASE_URL")
        self.vector_index = (
            PgVectorIndex(connection_string=db_url, table_name="doc_chunks") if db_url else None
        )

        # In a real app we'd verify connection, but for now we assume it's setup or will fail gracefully

    async def ingest_file(
        self, file_obj: BinaryIO, filename: str, workspace_id: str, content_type: str = "text/plain"
    ) -> dict[str, Any]:
        """
        Ingest a file: Save as Resource -> Parse -> Create Document -> Chunk -> Embed -> Store.
        """

        # 1. Read Content
        content_bytes = file_obj.read()
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # 2. Extract Text (Basic implementation for now)
        text_content = ""
        if content_type == "application/pdf" or filename.endswith(".pdf"):
            try:
                from io import BytesIO

                import pypdf

                reader = pypdf.PdfReader(BytesIO(content_bytes))
                text_content = "\n".join([page.extract_text() for page in reader.pages])
            except ImportError:
                # Fallback if pypdf is not installed
                logger.warning("pypdf not installed, storing raw bytes as text placeholder")
                text_content = f"[PDF Content Placeholder for {filename} - pypdf missing]"
            except Exception as e:
                logger.error(f"Failed to parse PDF: {e}")
                text_content = f"[Error parsing PDF: {e}]"
        else:
            # Assume text/code
            try:
                text_content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content_bytes.decode("latin-1")  # Fallback

        if not text_content:
            text_content = "[Empty Content]"

        # 3. DB Transaction
        with get_session() as db:
            # Check for existing resource/document to avoid duplicates (idempotency)
            # Strategy: If content hash exists in tenant, return existing
            existing_doc = (
                db.query(Document)
                .filter(
                    Document.tenant_id == uuid.UUID(workspace_id),
                    Document.content_hash == content_hash,
                )
                .first()
            )

            if existing_doc:
                return {
                    "status": "skipped",
                    "reason": "duplicate",
                    "document_id": str(existing_doc.id),
                    "title": existing_doc.title,
                }

            # Create Resource
            resource_id = uuid.uuid4()
            resource = Resource(
                id=resource_id,
                tenant_id=uuid.UUID(workspace_id),
                file_path=f"uploads/{filename}",  # Virtual path
                file_name=filename,
                resource_type=self._guess_resource_type(filename, content_type),
                content_hash=content_hash,
                size_bytes=len(content_bytes),
                is_active=True,
                is_parsed=True,
            )
            db.add(resource)

            # Create Document
            document_id = uuid.uuid4()
            document = Document(
                id=document_id,
                tenant_id=uuid.UUID(workspace_id),
                source_resource_id=resource_id,
                title=filename,
                content=text_content,
                content_hash=content_hash,
                doc_type=KnowledgeTypeEnum.DOCUMENTATION,
                source="upload",
                security_level=SecurityLevelEnum.INTERNAL,
                is_chunked=False,  # Will mark true after chunking
                is_embedded=False,
            )
            db.add(document)

            # Commit to get IDs
            db.commit()
            db.refresh(document)

            # 4. Chunking
            chunks = self._chunk_text(text_content)

            # 5. Embedding & Storage
            doc_chunks = []
            texts_to_embed = [c["content"] for c in chunks]

            # Batch embed
            embeddings = await self.embedding_service.batch_embed(texts_to_embed)

            for i, chunk_data in enumerate(chunks):
                chunk_id = uuid.uuid4()
                embedding = embeddings[i]

                # Create DB Object
                doc_chunk = DocChunk(
                    id=chunk_id,
                    tenant_id=uuid.UUID(workspace_id),
                    document_id=document_id,
                    content=chunk_data["content"],
                    chunk_index=i,
                    token_count=len(chunk_data["content"]) // 4,  # Rough approx
                    embedding=embedding,
                )
                db.add(doc_chunk)
                doc_chunks.append(doc_chunk)

            # Mark document as processed
            document.is_chunked = True
            document.is_embedded = True
            document.word_count = len(text_content.split())

            # Record Operation
            op = Operation(
                tenant_id=uuid.UUID(workspace_id),
                op_type=OperationTypeEnum.INGEST_DOCUMENT,
                input_hash=content_hash,  # Simplified
                inputs={"filename": filename, "size": len(content_bytes)},
                status=OperationStatusEnum.SUCCESS,
                document_id=document_id,
            )
            db.add(op)

            db.commit()

            return {
                "status": "success",
                "document_id": str(document_id),
                "chunks": len(doc_chunks),
                "title": filename,
            }

    def _guess_resource_type(self, filename: str, content_type: str) -> ResourceTypeEnum:
        if "image" in content_type:
            return ResourceTypeEnum.IMAGE
        if filename.endswith(".py") or filename.endswith(".js") or filename.endswith(".ts"):
            return ResourceTypeEnum.CODE_FILE
        if filename.endswith(".json") or filename.endswith(".yaml"):
            return ResourceTypeEnum.CONFIG
        return ResourceTypeEnum.DOCUMENT

    def _chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 100
    ) -> list[dict[str, Any]]:
        """
        Simple overlapping chunker.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # If we are not at the end, try to break at a newline or space
            if end < text_len:
                # Look for last newline in the window
                last_newline = text.rfind("\n", start, end)
                if last_newline != -1 and last_newline > start + chunk_size // 2:
                    end = last_newline + 1
                else:
                    # Look for last space
                    last_space = text.rfind(" ", start, end)
                    if last_space != -1 and last_space > start + chunk_size // 2:
                        end = last_space + 1

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({"content": chunk_content})

            start = end - overlap
            if start < 0:
                start = 0  # Safety

            # Ensure we advance
            if start >= end:
                start = end

        return chunks
