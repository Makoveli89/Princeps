"""Brain API - FastAPI application with full database integration."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_app():
    """Create FastAPI application."""
    try:
        from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    from ..core.db import get_session
    from ..core.models import DocChunk, Document, Operation, OperationStatusEnum, Tenant
    from ..distillation import DistillationService
    from ..ingestion import IngestService

    app = FastAPI(
        title="Princeps Brain Layer API",
        description="Knowledge management system for AI agents",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models for request/response
    class DocumentResponse(BaseModel):
        id: str
        title: str
        source: str
        content_preview: str
        word_count: int | None = None
        is_chunked: bool = False
        is_embedded: bool = False
        created_at: str | None = None

    class IngestRequest(BaseModel):
        path: str = Field(..., description="Path to file or repository")
        tenant_id: str | None = Field(None, description="Tenant ID (uses default if not provided)")

    class IngestResponse(BaseModel):
        status: str
        operation_id: str
        message: str

    class QueryRequest(BaseModel):
        text: str = Field(..., description="Query text for semantic search")
        limit: int = Field(5, ge=1, le=50, description="Max results to return")
        tenant_id: str | None = None

    class QueryResult(BaseModel):
        chunk_id: str
        document_id: str
        content: str
        score: float
        metadata: dict[str, Any] = {}

    class AgentRunRequest(BaseModel):
        prompt: str = Field(..., description="Prompt for the agent")
        agent_type: str = Field("general", description="Type of agent to use")
        tenant_id: str | None = None

    class WorkspaceResponse(BaseModel):
        id: str
        name: str
        description: str | None = None
        document_count: int = 0
        is_active: bool = True

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "princeps-brain"}

    @app.get("/api/v1/tenants")
    def list_tenants():
        """List all tenants."""
        try:
            with get_session() as session:
                tenants = session.query(Tenant).filter(Tenant.is_active == True).all()
                return {
                    "tenants": [
                        {"id": str(t.id), "name": t.name, "description": t.description}
                        for t in tenants
                    ],
                    "total": len(tenants),
                }
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/documents")
    def list_documents(
        limit: int = Query(default=20, le=100),
        offset: int = Query(default=0, ge=0),
        tenant_id: str | None = None,
    ):
        """List documents with pagination."""
        try:
            with get_session() as session:
                query = session.query(Document)

                if tenant_id:
                    query = query.filter(Document.tenant_id == tenant_id)

                total = query.count()
                documents = (
                    query.order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
                )

                return {
                    "documents": [
                        DocumentResponse(
                            id=str(d.id),
                            title=d.title,
                            source=d.source,
                            content_preview=(
                                d.content[:200] + "..." if len(d.content) > 200 else d.content
                            ),
                            word_count=d.word_count,
                            is_chunked=d.is_chunked,
                            is_embedded=d.is_embedded,
                            created_at=d.created_at.isoformat() if d.created_at else None,
                        ).model_dump()
                        for d in documents
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/documents/{document_id}")
    def get_document(document_id: str):
        """Get a specific document by ID."""
        try:
            with get_session() as session:
                doc = session.query(Document).filter(Document.id == document_id).first()
                if not doc:
                    raise HTTPException(status_code=404, detail="Document not found")

                chunks = session.query(DocChunk).filter(DocChunk.document_id == document_id).all()

                return {
                    "id": str(doc.id),
                    "tenant_id": str(doc.tenant_id),
                    "title": doc.title,
                    "content": doc.content,
                    "source": doc.source,
                    "doc_type": doc.doc_type.value if doc.doc_type else None,
                    "word_count": doc.word_count,
                    "token_count": doc.token_count,
                    "is_chunked": doc.is_chunked,
                    "is_embedded": doc.is_embedded,
                    "is_analyzed": doc.is_analyzed,
                    "chunk_count": len(chunks),
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/ingest/document")
    def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
        """Ingest a document into the knowledge base."""
        try:
            service = IngestService()
            # IngestService uses tenant_name parameter
            result = service.ingest_document(request.path, tenant_name=request.tenant_id)

            return IngestResponse(
                status="completed" if result.success else "failed",
                operation_id=result.operation_id or "unknown",
                message=(
                    f"Created {result.documents_created} documents, {result.chunks_created} chunks"
                    if result.success
                    else f"Failed: {', '.join(result.errors)}"
                ),
            ).model_dump()
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/ingest/repository")
    def ingest_repository(request: IngestRequest, url: str | None = None):
        """Ingest a repository into the knowledge base."""
        try:
            service = IngestService()
            # IngestService uses tenant_name parameter
            result = service.ingest_repository(request.path, tenant_name=request.tenant_id)

            return IngestResponse(
                status="completed" if result.success else "failed",
                operation_id=result.operation_id or "unknown",
                message=(
                    f"Created {result.documents_created} documents, {result.chunks_created} chunks, "
                    f"{result.resources_created} resources"
                    if result.success
                    else f"Failed: {', '.join(result.errors)}"
                ),
            ).model_dump()
        except Exception as e:
            logger.error(f"Error ingesting repository: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/distill/{document_id}")
    def distill_document(document_id: str, tenant_id: str | None = None):
        """Run knowledge distillation on a document."""
        try:
            with get_session() as session:
                doc = session.query(Document).filter(Document.id == document_id).first()
                if not doc:
                    raise HTTPException(status_code=404, detail="Document not found")

            service = DistillationService()
            # DistillationService uses tenant_name parameter
            result = service.distill_document(document_id, tenant_name=tenant_id)

            return {
                "status": "completed" if result.success else "failed",
                "operation_id": result.operation_id,
                "summaries_created": result.summaries_created,
                "entities_created": result.entities_created,
                "topics_created": result.topics_created,
                "concepts_created": result.concepts_created,
                "errors": result.errors,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error distilling document {document_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/query")
    def query(request: QueryRequest):
        """Semantic search across documents."""
        try:

            # For now, return text-based search results
            # Full semantic search requires embedding the query
            with get_session() as session:
                tid = request.tenant_id
                if tid:
                    chunks = (
                        session.query(DocChunk)
                        .filter(
                            DocChunk.tenant_id == tid, DocChunk.content.ilike(f"%{request.text}%")
                        )
                        .limit(request.limit)
                        .all()
                    )
                else:
                    chunks = (
                        session.query(DocChunk)
                        .filter(DocChunk.content.ilike(f"%{request.text}%"))
                        .limit(request.limit)
                        .all()
                    )

                return {
                    "results": [
                        QueryResult(
                            chunk_id=str(c.id),
                            document_id=str(c.document_id),
                            content=c.content,
                            score=1.0,  # Placeholder until vector search
                            metadata=c.metadata or {},
                        ).model_dump()
                        for c in chunks
                    ],
                    "query": request.text,
                    "total": len(chunks),
                }
        except Exception as e:
            logger.error(f"Error querying: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/metrics")
    def get_metrics():
        """Get system metrics."""
        try:
            with get_session() as session:
                total_documents = session.query(Document).count()
                total_chunks = session.query(DocChunk).count()
                total_operations = session.query(Operation).count()
                successful_ops = (
                    session.query(Operation)
                    .filter(Operation.status == OperationStatusEnum.SUCCESS)
                    .count()
                )
                failed_ops = (
                    session.query(Operation)
                    .filter(Operation.status == OperationStatusEnum.FAILED)
                    .count()
                )

                return {
                    "total_documents": total_documents,
                    "total_chunks": total_chunks,
                    "total_operations": total_operations,
                    "successful_operations": successful_ops,
                    "failed_operations": failed_ops,
                    "embedded_chunks": (
                        session.query(DocChunk).filter(DocChunk.embedding.isnot(None)).count()
                        if hasattr(DocChunk, "embedding")
                        else 0
                    ),
                }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Console app endpoints
    @app.get("/api/workspaces")
    def list_workspaces():
        """List workspaces (tenants) for console app."""
        try:
            with get_session() as session:
                tenants = session.query(Tenant).filter(Tenant.is_active == True).all()
                workspaces = []
                for t in tenants:
                    doc_count = session.query(Document).filter(Document.tenant_id == t.id).count()
                    workspaces.append(
                        WorkspaceResponse(
                            id=str(t.id),
                            name=t.name,
                            description=t.description,
                            document_count=doc_count,
                            is_active=t.is_active,
                        ).model_dump()
                    )
                return {"workspaces": workspaces, "total": len(workspaces)}
        except Exception as e:
            logger.error(f"Error listing workspaces: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/workspaces")
    def create_workspace(name: str = Body(...), description: str | None = Body(None)):
        """Create a new workspace (tenant)."""
        try:
            with get_session() as session:
                existing = session.query(Tenant).filter(Tenant.name == name).first()
                if existing:
                    raise HTTPException(
                        status_code=400, detail="Workspace with this name already exists"
                    )

                tenant = Tenant(name=name, description=description)
                session.add(tenant)
                session.commit()

                return WorkspaceResponse(
                    id=str(tenant.id),
                    name=tenant.name,
                    description=tenant.description,
                    document_count=0,
                    is_active=True,
                ).model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating workspace: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/run")
    def run_agent(request: AgentRunRequest):
        """Run an agent task (placeholder for framework integration)."""
        try:
            # This is a placeholder - full implementation requires framework integration
            return {
                "status": "completed",
                "response": f"Agent '{request.agent_type}' processed: {request.prompt[:100]}...",
                "agent_type": request.agent_type,
                "metadata": {"note": "Full agent integration pending framework connection"},
            }
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


class BrainAPI:
    """Wrapper for Brain API."""

    def __init__(self):
        self.app = create_app()

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    api = BrainAPI()
    api.run()
