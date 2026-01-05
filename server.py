import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file
import uuid
import datetime
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Princeps Imports
from brain.core.db import get_engine, init_db, get_session
from brain.core.models import Tenant, AgentRun, Document, KnowledgeNode, Resource
from framework.agents.base_agent import BaseAgent, AgentConfig, AgentTask, LLMProvider
from framework.agents.example_agent import SummarizationAgent
from framework.llms.multi_llm_client import MultiLLMClient

# Skills
from framework.skills.resolver import SkillResolver
from framework.skills.registry import get_registry
# Import library to register skills
import framework.skills.library.code_review
import framework.skills.library.research

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

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Types ---

class WorkspaceDTO(BaseModel):
    id: str
    name: str
    description: str | None
    status: str
    agentCount: int
    lastActive: str

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

# --- Helper: Dependency for DB Session ---
def get_db():
    with get_session() as session:
        yield session

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

# --- Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "princeps-console-backend"}

@app.get("/api/stats", response_model=StatsDTO)
def get_stats(db=Depends(get_db)):
    # Fetch real counts
    node_count = db.query(KnowledgeNode).count()
    # "Tasks completed" could be approximated by AgentRun counts or similar
    # Since AgentRun is not yet populated by this simple console, we might see 0.
    task_count = db.query(AgentRun).filter(AgentRun.success == True).count()

    # Active agents: approximate by unique agents in recent runs or just hardcode based on "available"
    # For now, let's use available agents count
    active_agents = len(agent_manager.available_agents)

    return StatsDTO(
        activeAgents=active_agents,
        tasksCompleted=task_count,
        uptime="99.9%", # Hardcoded for now
        knowledgeNodes=node_count
    )

@app.get("/api/workspaces", response_model=List[WorkspaceDTO])
def get_workspaces(db=Depends(get_db)):
    tenants = db.query(Tenant).filter(Tenant.is_active == True).all()

    result = []
    for t in tenants:
        # Approximate agent count per tenant (e.g. from runs)
        # agent_count = db.query(AgentRun.agent_id).filter(AgentRun.tenant_id == t.id).distinct().count()
        agent_count = 0 # Optimization: skip complex query for list view

        result.append(WorkspaceDTO(
            id=str(t.id),
            name=t.name,
            description=t.description,
            status="active" if t.is_active else "archived",
            agentCount=agent_count,
            lastActive=t.updated_at.isoformat() if t.updated_at else datetime.datetime.utcnow().isoformat()
        ))
    return result

@app.post("/api/workspaces", response_model=WorkspaceDTO)
def create_workspace(req: CreateWorkspaceRequest, db=Depends(get_db)):
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
        lastActive=new_tenant.created_at.isoformat()
    )

@app.get("/api/agents", response_model=List[AgentDTO])
def get_agents():
    return agent_manager.get_agents()

@app.post("/api/run")
async def run_agent(request: RunRequest):
    # For simplicity in this version, we'll await it (blocking).
    # In production, use background_tasks.add_task or a queue (Celery/Redis).
    # Since BaseAgent calls can be long, we should ideally be async.

    try:
        # Note: This will fail if no API keys are present in env, which is expected.
        # The UI should handle the error.
        result = await agent_manager.run_agent(request.agentId, request.input, request.workspaceId)
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
async def execute_skill(request: SkillRunRequest):
    """
    Execute a natural language skill.
    """
    try:
        result = await agent_manager.run_skill(request.query, request.workspaceId)
        return result
    except Exception as e:
        error_id = str(uuid.uuid4())
        print(f"Skill execution failed [ID: {error_id}]: {e}")
        # SENTINEL FIX: Prevent information leakage by hiding internal error details
        raise HTTPException(status_code=500, detail=f"Internal server error. Error ID: {error_id}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

