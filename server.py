import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="Princeps Console Backend", version="0.1.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Types ---

class Workspace(BaseModel):
    id: str
    name: str
    description: str
    status: str
    agentCount: int
    lastActive: str

class RunRequest(BaseModel):
    agentId: str
    input: str

# --- Mock Data ---

MOCK_WORKSPACES = [
    {
        "id": "ws-001",
        "name": "Sanctum Archivum",
        "description": "Knowledge base for ancient artifacts and lore processing.",
        "status": "active",
        "agentCount": 3,
        "lastActive": datetime.utcnow().isoformat(),
    },
    {
        "id": "ws-002",
        "name": "Obsidian Strategos",
        "description": "High-frequency trading and market strategy synthesis.",
        "status": "active",
        "agentCount": 5,
        "lastActive": datetime.utcnow().isoformat(),
    },
    {
        "id": "ws-003",
        "name": "Aetherial Nexus",
        "description": "Experimental communication protocols and linguistic analysis.",
        "status": "error",
        "agentCount": 1,
        "lastActive": datetime.utcnow().isoformat(),
    },
]

# --- Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "princeps-console-backend"}

@app.get("/api/workspaces", response_model=List[Workspace])
def get_workspaces():
    return MOCK_WORKSPACES

@app.post("/api/run")
def run_agent(request: RunRequest):
    # Mock execution
    return {
        "status": "success",
        "run_id": f"run-{int(datetime.utcnow().timestamp())}",
        "message": f"Agent {request.agentId} started successfully."
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
