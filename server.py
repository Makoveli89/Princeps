import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file
import uuid
import datetime
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel
from sqlalchemy import func, desc, text

# Princeps Imports
from brain.core.db import get_engine, init_db, get_session
from brain.core.models import Tenant, AgentRun, Document, KnowledgeNode, Resource, DocChunk, OperationStatusEnum, Operation
from framework.agents.base_agent import BaseAgent, AgentConfig, AgentTask, LLMProvider
from framework.agents.example_agent import SummarizationAgent
from framework.llms.multi_llm_client import MultiLLMClient

# Skills
from framework.skills.resolver import SkillResolver
from framework.skills.registry import get_registry
# Import library to register skills
import framework.skills.library.code_review
import framework.skills.library.research

# New Services
from framework.ingestion.service import IngestionService
from framework.retrieval.vector_search import (
    get_embedding_service,
    query_vector_index,
    PgVectorIndex,
    create_sqlite_index, # Import the factory function
    VectorSearchConfig,
    VectorSearchResult
)

# Initialize Database on Startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check/Init Database
    try:
        engine = get_engine()
        # In a real scenario, we might want to be careful about running init_db
        # unconditionally, but for "easy set up" it's helpful.
        # It handles "IF NOT EXISTS" internally.
        init_db(engine)
        print("✅ Database initialized.")
    except Exception as e:
        print(f"❌ Database initialization failed. Ensure Postgres is running. Error: {e}")

    yield
    # Cleanup if needed

app = FastAPI(title="Princeps Console Backend", version="0.1.0", lifespan=lifespan)

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SlowAPIMiddleware)

# --- Types ---

class WorkspaceDTO(BaseModel):
    id: str
    name: str
    description: str | None
    status: str
    agentCount: int
    lastActive: str
    docCount: int
    chunkCount: int
    runCount: int

class CreateWorkspaceRequest(BaseModel):
    name: str
    description: str

class AgentDTO(BaseModel):
    id: str
    name: str
    role: str
    status: str
    capabilities: List[str]

class RunRequest(BaseModel):
    agentId: str # For now this maps to a hardcoded agent type or ID
    input: str
    workspaceId: str

class SkillRunRequest(BaseModel):
    query: str
    workspaceId: str

class StatsDTO(BaseModel):
    activeAgents: int
    tasksCompleted: int
    uptime: str
    knowledgeNodes: int

class MetricPoint(BaseModel):
    time: str
    success: int
    failure: int

class SearchResultDTO(BaseModel):
    id: str
    score: float
    content: str
    source: str
    chunk_index: int

class RunLogDTO(BaseModel):
    run_id: str
    agent: str
    timestamp: str
    status: str
    input_preview: str
    output_preview: str
    duration_ms: int
    workspace_id: str
    logs: List[str]

# --- Helper: Dependency for DB Session ---
def get_db():
    try:
        with get_session() as session:
            yield session
    except Exception as e:
        # Yield None if DB is down, to allow fallback logic in endpoints
        print(f"DB Connection failed: {e}")
        yield None

# --- Helper: Fake Agent Manager ---
# In a real app, this would be a sophisticated service managing running instances.
class AgentManager:
    def __init__(self):
        # We'll just instantiate a few agents for listing purposes
        self.available_agents = [
            SummarizationAgent(agent_name="Scribe", agent_type="summarization"),
            # We could add more here
        ]
        # And a mock client for now since we don't have keys in env usually
        # But for "Real Data", we should try to use the real client if keys exist.
        self.llm_client = MultiLLMClient()
        self.skill_resolver = SkillResolver(llm_client=self.llm_client)

    def get_agents(self) -> List[AgentDTO]:
        return [
            AgentDTO(
                id=a.agent_id,
                name=a.agent_name,
                role=a.agent_type.capitalize(),
                status="idle",
                capabilities=a.get_capabilities()["capabilities"]
            )
            for a in self.available_agents
        ]

    async def run_agent(self, agent_id: str, prompt: str, workspace_id: str) -> Dict[str, Any]:
        # Find agent by ID or Type
        # Ideally we persist agents in DB, but for now we look up in our list
        # or create a new transient one.
        agent_def = next((a for a in self.available_agents if a.agent_id == agent_id), None)
        if not agent_def:
            # Fallback: create a new one based on ID assuming it is a type
            agent_def = SummarizationAgent(agent_name="TransientScribe", agent_type="summarization")

        # Configure the agent with the real client
        agent_def.llm_client = self.llm_client

        task = agent_def.create_task(
            prompt=prompt,
            tenant_id=workspace_id
        )

        # We need to run this asynchronously
        response = await agent_def.execute_task(task)

        # Log to DB (AgentRun)
        # Assuming BaseAgent.execute_task might handle this in future,
        # but for now we manually ensure it is persisted or use the return object

        return response.to_dict()

    async def run_skill(self, query: str, workspace_id: str) -> Dict[str, Any]:
        """
        Resolve and execute a skill based on natural language query.
        """
        skill_name, params = await self.skill_resolver.resolve(query)

        if not skill_name:
            return {
                "status": "failed",
                "message": "Could not identify a skill for your request.",
                "query": query
            }

        skill_cls = get_registry().get_skill(skill_name)
        if not skill_cls:
             return {
                "status": "failed",
                "message": f"Skill '{skill_name}' was resolved but not found in registry.",
                "query": query
            }

        # Instantiate and execute
        skill_instance = skill_cls(context={"tenant_id": workspace_id})

        try:
            result = await skill_instance.execute(**params)
            return {
                "status": "success",
                "skill_name": skill_name,
                "parameters": params,
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "skill_name": skill_name
            }


agent_manager = AgentManager()
ingestion_service = IngestionService()

# --- Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "princeps-console-backend"}

@app.get("/api/stats", response_model=StatsDTO)
def get_stats(db=Depends(get_db)):
    # Fallback if DB unavailable
    if db is None:
        return StatsDTO(activeAgents=1, tasksCompleted=0, uptime="99.9%", knowledgeNodes=0)

    # Fetch real counts
    try:
        node_count = db.query(KnowledgeNode).count()
        task_count = db.query(AgentRun).filter(AgentRun.success == True).count()
    except:
        node_count = 0
        task_count = 0

    active_agents = len(agent_manager.available_agents)

    return StatsDTO(
        activeAgents=active_agents,
        tasksCompleted=task_count,
        uptime="99.9%",
        knowledgeNodes=node_count
    )

@app.get("/api/workspaces", response_model=List[WorkspaceDTO])
def get_workspaces(db=Depends(get_db)):
    if db is None:
        # Return empty list if DB is down, rather than 500
        return []

    try:
        tenants = db.query(Tenant).filter(Tenant.is_active == True).all()

        result = []
        for t in tenants:
            # Real counts per tenant
            doc_count = db.query(Document).filter(Document.tenant_id == t.id).count()
            chunk_count = db.query(DocChunk).filter(DocChunk.tenant_id == t.id).count()
            run_count = db.query(AgentRun).filter(AgentRun.tenant_id == t.id).count()
            agent_count = 0

            result.append(WorkspaceDTO(
                id=str(t.id),
                name=t.name,
                description=t.description,
                status="active" if t.is_active else "archived",
                agentCount=agent_count,
                lastActive=t.updated_at.isoformat() if t.updated_at else datetime.datetime.utcnow().isoformat(),
                docCount=doc_count,
                chunkCount=chunk_count,
                runCount=run_count
            ))
        return result
    except Exception as e:
        print(f"Error fetching workspaces: {e}")
        return []

@app.post("/api/workspaces", response_model=WorkspaceDTO)
def create_workspace(req: CreateWorkspaceRequest, db=Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Check if exists
    existing = db.query(Tenant).filter(Tenant.name == req.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Workspace name already exists")

    new_tenant = Tenant(
        name=req.name,
        description=req.description,
        is_active=True
    )
    db.add(new_tenant)
    db.commit()
    db.refresh(new_tenant)

    return WorkspaceDTO(
        id=str(new_tenant.id),
        name=new_tenant.name,
        description=new_tenant.description,
        status="active",
        agentCount=0,
        lastActive=new_tenant.created_at.isoformat(),
        docCount=0,
        chunkCount=0,
        runCount=0
    )

@app.get("/api/agents", response_model=List[AgentDTO])
def get_agents():
    return agent_manager.get_agents()

@app.post("/api/run")
@limiter.limit("5/minute")
async def run_agent(request: Request, body: RunRequest):
    # For simplicity in this version, we'll await it (blocking).
    # In production, use background_tasks.add_task or a queue (Celery/Redis).
    # Since BaseAgent calls can be long, we should ideally be async.

    try:
        # Note: This will fail if no API keys are present in env, which is expected.
        # The UI should handle the error.
        result = await agent_manager.run_agent(body.agentId, body.input, body.workspaceId)
        return {
            "status": "success",
            "run_id": result.get("task_id"),
            "output": result.get("response_text", ""),
            "full_result": result
        }
    except Exception as e:
        error_id = str(uuid.uuid4())
        print(f"Agent run failed [ID: {error_id}]: {e}")
        # SENTINEL FIX: Prevent information leakage by hiding internal error details
        raise HTTPException(status_code=500, detail=f"Internal server error. Error ID: {error_id}")

@app.post("/api/skills/execute")
@limiter.limit("10/minute")
async def execute_skill(request: Request, body: SkillRunRequest):
    """
    Execute a natural language skill.
    """
    try:
        result = await agent_manager.run_skill(body.query, body.workspaceId)
        return result
    except Exception as e:
        error_id = str(uuid.uuid4())
        print(f"Skill execution failed [ID: {error_id}]: {e}")
        # SENTINEL FIX: Prevent information leakage by hiding internal error details
        raise HTTPException(status_code=500, detail=f"Internal server error. Error ID: {error_id}")

# --- New Endpoints ---

@app.post("/api/ingest")
@limiter.limit("10/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    workspace_id: str = Form(...)
):
    try:
        result = await ingestion_service.ingest_file(
            file.file,
            file.filename,
            workspace_id,
            content_type=file.content_type
        )
        return result
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search", response_model=List[SearchResultDTO])
@limiter.limit("20/minute")
async def search_knowledge(
    request: Request,
    q: str,
    workspaceId: str,
    limit: int = 10,
    db=Depends(get_db)
):
    """
    Vector search on ingested knowledge.
    """
    try:
        # Get query embedding
        emb_service = get_embedding_service()
        query_vector = await emb_service.embed_text(q)

        # Initialize Vector Index based on Environment
        import os
        db_url = os.getenv("DATABASE_URL", "sqlite:///./princeps.db")

        if db_url.startswith("sqlite"):
            index = create_sqlite_index(connection_string=db_url, table_name="doc_chunks")
        else:
            index = PgVectorIndex(connection_string=db_url, table_name="doc_chunks")

        # Search
        # Filter by tenant
        from framework.retrieval.vector_search import SearchFilter
        filters = SearchFilter(tenant_id=workspaceId)

        results = await query_vector_index(
            query_vector,
            index,
            top_k=limit,
            filters=filters
        )

        # Convert to DTO
        dtos = []
        for r in results:
            dtos.append(SearchResultDTO(
                id=str(r.id),
                score=r.score,
                content=r.content,
                source=r.metadata.get("source") or "unknown",
                chunk_index=r.metadata.get("chunk_index") or 0
            ))

        return dtos

    except Exception as e:
        print(f"Search failed: {e}")
        # logger.error(f"Search failed: {e}")
        # Return empty list on failure gracefully for UI? Or raise?
        # Let's raise for visibility
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics", response_model=List[MetricPoint])
def get_metrics(db=Depends(get_db)):
    """
    Aggregate AgentRun success/failures over last 24h by 4h buckets.
    """
    if db is None:
        return []

    try:
        # Simplified: Get all runs in last 24h
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
        runs = db.query(AgentRun).filter(AgentRun.started_at >= cutoff).all()

        if not runs:
            # Return at least empty structure
            return [
                MetricPoint(time="00:00", success=0, failure=0),
                MetricPoint(time="06:00", success=0, failure=0),
                MetricPoint(time="12:00", success=0, failure=0),
                MetricPoint(time="18:00", success=0, failure=0),
            ]

        # Bucket manually
        buckets = {}
        for r in runs:
            if not r.started_at: continue
            # Bucket by 4 hours
            hour = r.started_at.hour
            bucket_hour = (hour // 4) * 4
            key = f"{bucket_hour:02d}:00"

            if key not in buckets: buckets[key] = {"success": 0, "failure": 0}

            if r.success:
                buckets[key]["success"] += 1
            else:
                buckets[key]["failure"] += 1

        result = []
        # Ensure order
        for h in range(0, 24, 4):
            key = f"{h:02d}:00"
            data = buckets.get(key, {"success": 0, "failure": 0})
            result.append(MetricPoint(time=key, success=data["success"], failure=data["failure"]))

        return result
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return []

@app.get("/api/runs", response_model=List[RunLogDTO])
def get_runs(workspaceId: Optional[str] = None, limit: int = 50, db=Depends(get_db)):
    if db is None:
        return []

    try:
        query = db.query(AgentRun)
        if workspaceId:
            query = query.filter(AgentRun.tenant_id == uuid.UUID(workspaceId))

        runs = query.order_by(desc(AgentRun.started_at)).limit(limit).all()

        dtos = []
        for r in runs:
            # Convert DB model to DTO
            dtos.append(RunLogDTO(
                run_id=str(r.id),
                agent=r.agent_id,
                timestamp=r.started_at.isoformat() if r.started_at else "",
                status="SUCCESS" if r.success else "FAILURE",
                input_preview=r.task[:50] + "..." if r.task else "",
                output_preview=str(r.solution)[:50] + "..." if r.solution else (r.feedback[:50] if r.feedback else ""),
                duration_ms=r.duration_ms or 0,
                workspace_id=str(r.tenant_id),
                logs=[] # Logs not currently stored in AgentRun explicitly as list
            ))
        return dtos
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
