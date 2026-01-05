# Princeps

A sophisticated AI platform combining persistent knowledge management with multi-agent orchestration.

## Architecture

Princeps consists of two core packages:

### Brain Layer (`brain/`)
Knowledge management system for AI agents with multi-tenant support, semantic search, and knowledge distillation.

```
brain/
├── core/           # Database models, schemas, utilities
├── ingestion/      # Document & repository ingestion
├── distillation/   # Knowledge extraction (NER, topics, concepts)
├── observability/  # Structured logging, metrics, tracing
├── resilience/     # Fault tolerance, retry, idempotency
├── security/       # Access control, PII scanning, tenant isolation
└── interface/      # CLI and REST API
```

### Multi-Agent Framework (`framework/`)
Orchestration system for specialized AI agents with multi-LLM support.

```
framework/
├── agents/         # Base agent + specialists (planner, executor, retriever...)
├── llms/           # Multi-LLM client with automatic fallback
├── tools/          # Tool registry and built-in tools
├── core/           # Task routing and dispatcher
├── retrieval/      # Vector search integration
└── evaluation/     # A/B testing and strategy metrics
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/princeps.git
cd princeps

# Install dependencies
pip install -e ".[all]"

# Set up environment
cp .env.example .env
# Edit .env with your database credentials and API keys
```

### Launch the Platform

To start the full platform (Backend API + Console UI), simply run:

**Linux / macOS:**
```bash
./start.sh
```

**Windows:**
Double-click `start.bat` or run in Command Prompt:
```cmd
start.bat
```

This will:
1. Install necessary Python and Node.js dependencies.
2. Start the FastAPI backend server on port 8000.
3. Start the Princeps Console (React UI) on port 5173.

Access the console at: **http://localhost:5173**

### Brain Layer Usage

```python
from brain import get_session, init_db
from brain.ingestion import IngestService
from brain.distillation import DistillationService

# Initialize database
init_db()

# Ingest a document
ingest = IngestService()
result = ingest.ingest_document("/path/to/document.pdf")

# Distill knowledge
distill = DistillationService()
result = distill.distill_document(document_id)
```

### Framework Usage

```python
from framework.agents import BaseAgent, PlannerAgent, ExecutorAgent
from framework.llms import MultiLLMClient
from framework.core import Dispatcher, Task, TaskType

# Create dispatcher
dispatcher = Dispatcher()

# Create and dispatch a task
task = Task(
    task_type=TaskType.COMPLEX_REASONING,
    payload={"question": "What is the meaning of life?"},
)
result = await dispatcher.dispatch(task)
```

### CLI

```bash
# Initialize brain database
princeps-brain init

# Ingest documents
princeps-brain ingest /path/to/docs --recursive

# Start API server
uvicorn brain.interface.brain_api:create_app --factory --reload
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/princeps

# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Logging
LOG_LEVEL=INFO
```

## Database Schema

The Brain Layer uses PostgreSQL with pgvector for semantic search. Key tables:

- **Tenant** - Multi-tenant isolation
- **Repository** - Git repository tracking
- **Resource** - File/document resources
- **Document** - Ingested content with metadata
- **DocChunk** - Chunked content with embeddings
- **Operation** - Idempotent operation tracking
- **AgentRun** - Agent execution logs
- **KnowledgeNode/Edge** - Knowledge graph

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=brain --cov=framework

# Run specific test file
pytest tests/test_ingestion.py -v
```

## License

MIT
