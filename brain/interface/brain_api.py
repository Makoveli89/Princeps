"""Brain API - FastAPI application."""


def create_app():
    """Create FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

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

    @app.get("/health")
    def health():
        return {"status": "healthy", "service": "princeps-brain"}

    @app.get("/api/v1/documents")
    def list_documents(limit: int = Query(default=20, le=100), offset: int = 0):
        # Stub
        return {"documents": [], "total": 0, "limit": limit, "offset": offset}

    @app.get("/api/v1/documents/{document_id}")
    def get_document(document_id: str):
        # Stub
        raise HTTPException(status_code=404, detail="Document not found")

    @app.post("/api/v1/ingest/document")
    def ingest_document(path: str, tenant: str | None = None):
        # Stub
        return {"status": "accepted", "operation_id": "pending"}

    @app.post("/api/v1/ingest/repository")
    def ingest_repository(path: str, url: str | None = None, tenant: str | None = None):
        # Stub
        return {"status": "accepted", "operation_id": "pending"}

    @app.post("/api/v1/distill/{document_id}")
    def distill_document(document_id: str):
        # Stub
        return {"status": "accepted", "operation_id": "pending"}

    @app.post("/api/v1/query")
    def query(text: str, limit: int = 5):
        # Stub
        return {"results": [], "query": text}

    @app.get("/api/v1/metrics")
    def get_metrics():
        # Stub
        return {"total_documents": 0, "total_chunks": 0}

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
