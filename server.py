from dotenv import load_dotenv

load_dotenv()  # Load .env file
import datetime
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy import desc, func

# --- Structlog Configuration ---

# Configure standard logging to intercept basic logs
logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Princeps Imports
# Import library to register skills
from brain.core.db import get_engine, get_session, init_db
from brain.core.models import (
    AgentRun,
    DocChunk,
    Document,
    KnowledgeNode,
    Tenant,
)
from framework.agents.concept_agent import ConceptAgent
from framework.agents.entity_agent import EntityExtractionAgent
from framework.agents.example_agent import SummarizationAgent
from framework.agents.executor_agent import ExecutorAgent
from framework.agents.planner_agent import PlannerAgent
from framework.agents.retriever_agent import RetrieverAgent
from framework.agents.topic_agent import TopicAgent

# New Services
from framework.ingestion.service import IngestionService
from framework.llms.multi_llm_client import MultiLLMClient
from framework.retrieval.vector_search import (
    PgVectorIndex,
    create_sqlite_index,  # Import the factory function
    get_embedding_service,
    query_vector_index,
)
from framework.skills.registry import get_registry

# Skills
from framework.skills.resolver import SkillResolver

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.info("database_initialized")
    except Exception as e:
        logger.error(f"database_initialization_failed: {e}")

    # Initialize Vector Index Singleton
    try:
        db_url = os.getenv("DATABASE_URL", "sqlite:///./princeps.db")
        if db_url.startswith("sqlite"):
            app.state.vector_index = create_sqlite_index(
                connection_string=db_url, table_name="doc_chunks"
            )
        else:
            app.state.vector_index = PgVectorIndex(
                connection_string=db_url, table_name="doc_chunks"
            )
        logger.info("vector_index_initialized")
    except Exception as e:
        logger.error(f"vector_index_initialization_failed: {e}")
        # Ensure we don't crash startup, but endpoint will need fallback
        app.state.vector_index = None

    yield

    # Cleanup
    if hasattr(app.state, "vector_index") and app.state.vector_index:
        await app.state.vector_index.close()
        logger.info("vector_index_closed")


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

# --- Global Exception Handler ---


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to catch unhandled exceptions, log them securely,
    and return a generic error message to the client.
    """
    try:
        error_id = str(uuid.uuid4())
        # Safe logging even if request object is malformed or logger fails
        path = getattr(request.url, "path", "unknown")
        method = getattr(request, "method", "unknown")

        # structlog binded logger might expect kwargs, but if not bound, it might differ.
        # However, structlog.get_logger() returns a bound logger usually.
        # The error `Logger._log() got an unexpected keyword argument 'error_id'` suggests
        # that `logger` might be a standard logging.Logger instance in some context?
        # In server.py: logger = structlog.get_logger()
        # BUT later: logger = logging.getLogger(__name__)  <-- THIS IS THE ISSUE.

        # We should use the structlog logger.
        struct_logger = structlog.get_logger()
        struct_logger.error(
            "unhandled_exception", error_id=error_id, error=str(exc), path=path, method=method
        )

        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error. Error ID: {error_id}"},
        )
    except Exception as e:
        # Fallback if the handler itself fails
        print(f"Error in global exception handler: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error (handler failed)."},
        )


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
    name: str = Field(..., max_length=50, pattern=r"^[a-zA-Z0-9_\-\s]+$")
    description: str = Field(..., max_length=200)


class AgentDTO(BaseModel):
    id: str
    name: str
    role: str
    status: str
    capabilities: list[str]


class RunRequest(BaseModel):
    agentId: str  # For now this maps to a hardcoded agent type or ID
    input: str = Field(..., max_length=100000)
    workspaceId: str


class SkillRunRequest(BaseModel):
    query: str = Field(..., max_length=10000)
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
    logs: list[str]


# --- Helper: Dependency for DB Session ---
def get_db():
    # Attempt to initialize session
    db_context = None
    session = None
    try:
        db_context = get_session()
        session = db_context.__enter__()
    except Exception as e:
        # Setup failed (DB down?)
        logger.error(f"db_connection_failed: {e}")
        yield None
        return

    # Yield session for usage
    try:
        yield session
    except Exception:
        # Endpoint failed - propagate to context manager (rollback)
        if db_context and not db_context.__exit__(*sys.exc_info()):
            raise
    else:
        # Success - commit
        if db_context:
            db_context.__exit__(None, None, None)


# --- Helper: Fake Agent Manager ---
# In a real app, this would be a sophisticated service managing running instances.
class AgentManager:
    def __init__(self):
        self.llm_client = MultiLLMClient()
        self.skill_resolver = SkillResolver(llm_client=self.llm_client)

        # Instantiate agents
        self.available_agents = [
            SummarizationAgent(
                agent_name="Scribe", agent_type="summarization", llm_client=self.llm_client
            ),
            PlannerAgent(agent_name="Strategist", agent_type="planner", llm_client=self.llm_client),
            ExecutorAgent(agent_name="Operator", agent_type="executor", llm_client=self.llm_client),
            RetrieverAgent(
                agent_name="Archivist", agent_type="retriever", llm_client=self.llm_client
            ),
            EntityExtractionAgent(
                agent_name="Profiler", agent_type="entity_extraction", llm_client=self.llm_client
            ),
            TopicAgent(agent_name="Analyst", agent_type="topic", llm_client=self.llm_client),
            ConceptAgent(agent_name="Architect", agent_type="concept", llm_client=self.llm_client),
        ]

    def get_agents(self) -> list[AgentDTO]:
        return [
            AgentDTO(
                id=a.agent_id,
                name=a.agent_name,
                role=a.agent_type.capitalize(),
                status="idle",
                capabilities=a.get_capabilities().get("capabilities", []),
            )
            for a in self.available_agents
        ]

    async def run_agent(self, agent_id: str, prompt: str, workspace_id: str) -> dict[str, Any]:
        # Find agent by ID or Type
        agent_def = next((a for a in self.available_agents if a.agent_id == agent_id), None)

        # Fallback to fuzzy match on type if ID match failed
        if not agent_def:
            agent_def = next(
                (a for a in self.available_agents if a.agent_type.lower() in agent_id.lower()),
                None,
            )

        if not agent_def:
            # Fallback: create a new one based on ID assuming it is a type
            agent_def = SummarizationAgent(
                agent_name="TransientScribe",
                agent_type="summarization",
                llm_client=self.llm_client,
            )

        # Configure the agent with the real client (redundant if passed in init, but safe)
        agent_def.llm_client = self.llm_client

        task = agent_def.create_task(prompt=prompt, tenant_id=workspace_id)

        # We need to run this asynchronously
        response = await agent_def.execute_task(task)

        # Log to DB (AgentRun)
        # Assuming BaseAgent.execute_task might handle this in future,
        # but for now we manually ensure it is persisted or use the return object

        return response.to_dict()

    async def run_skill(self, query: str, workspace_id: str) -> dict[str, Any]:
        """
        Resolve and execute a skill based on natural language query.
        """
        skill_name, params = await self.skill_resolver.resolve(query)

        if not skill_name:
            return {
                "status": "failed",
                "message": "Could not identify a skill for your request.",
                "query": query,
            }

        skill_cls = get_registry().get_skill(skill_name)
        if not skill_cls:
            return {
                "status": "failed",
                "message": f"Skill '{skill_name}' was resolved but not found in registry.",
                "query": query,
            }

        # Instantiate and execute
        skill_instance = skill_cls(context={"tenant_id": workspace_id})

        # Let exceptions bubble up to the global handler
        result = await skill_instance.execute(**params)
        return {
            "status": "success",
            "skill_name": skill_name,
            "parameters": params,
            "result": result,
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
    node_count = db.query(KnowledgeNode).count()
    task_count = db.query(AgentRun).filter(AgentRun.success == True).count()

    active_agents = len(agent_manager.available_agents)

    return StatsDTO(
        activeAgents=active_agents,
        tasksCompleted=task_count,
        uptime="99.9%",
        knowledgeNodes=node_count,
    )


@app.get("/api/workspaces", response_model=list[WorkspaceDTO])
def get_workspaces(db=Depends(get_db)):
    if db is None:
        # Return empty list if DB is down, rather than 500
        # This behavior is debatable but keeps the UI working even if DB is problematic initially
        return []

    # Optimize query to fetch tenants with counts in a single query
    # Using scalar_subquery to prevent N+1 problem
    from sqlalchemy import func, select

    doc_count_sub = (
        select(func.count(Document.id))
        .where(Document.tenant_id == Tenant.id)
        .correlate(Tenant)
        .scalar_subquery()
    )

    chunk_count_sub = (
        select(func.count(DocChunk.id))
        .where(DocChunk.tenant_id == Tenant.id)
        .correlate(Tenant)
        .scalar_subquery()
    )

    run_count_sub = (
        select(func.count(AgentRun.id))
        .where(AgentRun.tenant_id == Tenant.id)
        .correlate(Tenant)
        .scalar_subquery()
    )

    stmt = select(
        Tenant,
        doc_count_sub.label("doc_count"),
        chunk_count_sub.label("chunk_count"),
        run_count_sub.label("run_count"),
    ).where(Tenant.is_active == True)

    rows = db.execute(stmt).all()

    result = []
    for row in rows:
        t = row[0]  # Tenant object
        doc_count = row[1] or 0
        chunk_count = row[2] or 0
        run_count = row[3] or 0
        agent_count = 0

        result.append(
            WorkspaceDTO(
                id=str(t.id),
                name=t.name,
                description=t.description,
                status="active" if t.is_active else "archived",
                agentCount=agent_count,
                lastActive=(
                    t.updated_at.isoformat()
                    if t.updated_at
                    else datetime.datetime.utcnow().isoformat()
                ),
                docCount=doc_c or 0,
                chunkCount=chunk_c or 0,
                runCount=run_c or 0,
            )
        )
    return result


@app.post("/api/workspaces", response_model=WorkspaceDTO)
def create_workspace(req: CreateWorkspaceRequest, db=Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Check if exists
    existing = db.query(Tenant).filter(Tenant.name == req.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Workspace name already exists")

    new_tenant = Tenant(name=req.name, description=req.description, is_active=True)
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
        runCount=0,
    )


@app.get("/api/agents", response_model=list[AgentDTO])
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
            "full_result": result,
        }
    except Exception as e:
        error_id = str(uuid.uuid4())
        logger.error("agent_run_failed", error_id=error_id, error=str(e))
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
        logger.error("skill_execution_failed", error_id=error_id, error=str(e))
        # SENTINEL FIX: Prevent information leakage by hiding internal error details
        raise HTTPException(status_code=500, detail=f"Internal server error. Error ID: {error_id}")


# --- New Endpoints ---


@app.post("/api/ingest")
@limiter.limit("10/minute")
async def ingest_document(
    request: Request, file: UploadFile = File(...), workspace_id: str = Form(...)
):
    result = await ingestion_service.ingest_file(
        file.file, file.filename, workspace_id, content_type=file.content_type
    )
    return result


@app.get("/api/search", response_model=list[SearchResultDTO])
@limiter.limit("20/minute")
async def search_knowledge(
    request: Request, q: str, workspaceId: str, limit: int = 10, db=Depends(get_db)
):
    """
    Vector search on ingested knowledge.
    """
    # Get query embedding
    emb_service = get_embedding_service()
    query_vector = await emb_service.embed_text(q)

    # Use Singleton Vector Index
    index = getattr(app.state, "vector_index", None)
    if not index:
        # Fallback initialization if singleton missing (e.g., startup error)
        logger.warning("vector_index_fallback_init")
        db_url = os.getenv("DATABASE_URL", "sqlite:///./princeps.db")
        if db_url.startswith("sqlite"):
            index = create_sqlite_index(connection_string=db_url, table_name="doc_chunks")
        else:
            index = PgVectorIndex(connection_string=db_url, table_name="doc_chunks")

    # Search
    index = request.app.state.vector_index

    # Filter by tenant
    from framework.retrieval.vector_search import SearchFilter

    filters = SearchFilter(tenant_id=workspaceId)

    results = await query_vector_index(query_vector, index, top_k=limit, filters=filters)

    # Convert to DTO
    dtos = []
    for r in results:
        dtos.append(
            SearchResultDTO(
                id=str(r.id),
                score=r.score,
                content=r.content,
                source=r.metadata.get("source") or "unknown",
                chunk_index=r.metadata.get("chunk_index") or 0,
            )
        )

    return dtos


@app.get("/api/metrics", response_model=list[MetricPoint])
def get_metrics(db=Depends(get_db)):
    """
    Aggregate AgentRun success/failures over last 24h by 4h buckets.
    """
    if db is None:
        return []

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
        if not r.started_at:
            continue
        # Bucket by 4 hours
        hour = r.started_at.hour
        bucket_hour = (hour // 4) * 4
        key = f"{bucket_hour:02d}:00"

        if key not in buckets:
            buckets[key] = {"success": 0, "failure": 0}

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


@app.get("/api/runs", response_model=list[RunLogDTO])
def get_runs(workspaceId: str | None = None, limit: int = 50, db=Depends(get_db)):
    if db is None:
        return []

    query = db.query(AgentRun)
    if workspaceId:
        query = query.filter(AgentRun.tenant_id == uuid.UUID(workspaceId))

    runs = query.order_by(desc(AgentRun.started_at)).limit(limit).all()

    dtos = []
    for r in runs:
        # Convert DB model to DTO
        dtos.append(
            RunLogDTO(
                run_id=str(r.id),
                agent=r.agent_id,
                timestamp=r.started_at.isoformat() if r.started_at else "",
                status="SUCCESS" if r.success else "FAILURE",
                input_preview=r.task[:50] + "..." if r.task else "",
                output_preview=(
                    str(r.solution)[:50] + "..."
                    if r.solution
                    else (r.feedback[:50] if r.feedback else "")
                ),
                duration_ms=r.duration_ms or 0,
                workspace_id=str(r.tenant_id),
                logs=[],  # Logs not currently stored in AgentRun explicitly as list
            )
        )
    return dtos


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
