# GitHub Copilot Instructions for Princeps

## Introduction
These instructions guide GitHub Copilot to follow the coding standards, architecture patterns, and best practices for the Princeps AI platform. Princeps combines persistent knowledge management with multi-agent orchestration.

## Project Overview

Princeps is a sophisticated AI platform with two core packages:

### Brain Layer (`brain/`)
Knowledge management system for AI agents with:
- Multi-tenant support with tenant isolation
- Document and repository ingestion
- Semantic search with pgvector embeddings
- Knowledge distillation (NER, topics, concepts)
- Structured logging and observability
- Security features (PII scanning, access control)

**Key directories:**
- `brain/core/` - Database models (SQLAlchemy), schemas (Pydantic), utilities
- `brain/ingestion/` - Document & repository ingestion pipeline
- `brain/distillation/` - Knowledge extraction
- `brain/observability/` - Structured logging, metrics, tracing
- `brain/resilience/` - Fault tolerance, retry logic, idempotency
- `brain/security/` - Access control, PII scanning, tenant isolation
- `brain/interface/` - CLI and REST API

### Multi-Agent Framework (`framework/`)
Orchestration system for specialized AI agents:
- Multi-LLM support with automatic fallback
- Specialized agents (planner, executor, retriever)
- Tool registry and built-in tools
- Task routing and dispatch
- A/B testing and evaluation

**Key directories:**
- `framework/agents/` - Base agent + specialist implementations
- `framework/llms/` - Multi-LLM client with fallback
- `framework/tools/` - Tool registry and implementations
- `framework/core/` - Task routing and dispatcher
- `framework/retrieval/` - Vector search integration
- `framework/evaluation/` - Metrics and strategy evaluation

## Technology Stack
- **Language:** Python 3.10+
- **Database:** PostgreSQL with pgvector extension
- **ORM:** SQLAlchemy 2.0+
- **Validation:** Pydantic 2.0+
- **API:** FastAPI + Uvicorn
- **ML:** sentence-transformers, spaCy, transformers
- **Testing:** pytest with pytest-asyncio
- **Code Quality:** Black (formatting), Ruff (linting), mypy (type checking)

## Code Style Guidelines

### Python Style
- Follow PEP 8 with modifications specified in `pyproject.toml`
- Use **Black** for code formatting (line length: 100)
- Use **Ruff** for linting
- Use **type hints** for all function parameters and return values
- Prefer f-strings for string formatting
- Use dataclasses or Pydantic models for structured data

### Naming Conventions
- `snake_case` for functions, variables, and module names
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Prefix private methods/attributes with single underscore (`_method_name`)
- Use descriptive names that reflect purpose

### Documentation
- Include docstrings for all modules, classes, and public functions
- Use triple-quoted strings with description and usage examples
- Format: Start with one-line summary, followed by detailed description
- For module-level docstrings, use underline formatting (see example below)
- For class and function docstrings, use standard format without underlines
- Example (module-level):
  ```python
  """
  Module Name - Brief Description
  =================================
  
  Detailed description of functionality.
  
  Usage:
      from module import Class
      obj = Class()
      result = obj.method()
  """
  ```

### Imports
- Group imports in this order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- Use absolute imports from package root
- Avoid wildcard imports (`from module import *`)
- Sort imports within each group (Ruff handles this)

## Coding Practices

### Database Operations
- Always use SQLAlchemy ORM models defined in `brain/core/models.py`
- Use Pydantic schemas from `brain/core/schemas.py` for validation
- Always propagate tenant context for multi-tenant operations
- Use `get_session()` context manager for database sessions
- Commit within the same session scope where changes are made
- Use idempotency patterns from `brain/resilience/` for retry safety

### Error Handling
- Use structured logging from `brain/observability/`
- Catch specific exceptions rather than bare `except:`
- Log errors with context (tenant_id, operation_id, etc.)
- Use retry mechanisms from `brain/resilience/retry_manager.py`
- Return meaningful error messages with appropriate HTTP status codes

### Security
- Never log or expose sensitive data (API keys, passwords, PII)
- Always scan for PII using `brain/security/pii_scanner.py`
- Enforce tenant isolation in all multi-tenant operations
- Validate all user inputs with Pydantic schemas
- Use parameterized queries (SQLAlchemy handles this)

### Testing
- Write tests for all new features
- Place tests in `tests/` directory
- Use pytest fixtures from `tests/conftest.py`
- Test file naming: `test_*.py`
- Test function naming: `test_<functionality>_<condition>`
- Use descriptive test docstrings
- Mock external dependencies (LLM APIs, file systems)
- Test both success and failure paths
- Test edge cases and boundary conditions

### Async/Await
- Use `async/await` for I/O-bound operations
- Mark async tests with `@pytest.mark.asyncio`
- Use `asyncio.gather()` for concurrent operations
- Be aware of async context in FastAPI endpoints

## Build and Test Commands

### Setup
```bash
# Install dependencies
pip install -e ".[all]"

# Set up environment
cp .env.example .env
# Edit .env with your configuration
```

### Development
```bash
# Format code
black brain/ framework/ tests/

# Lint code
ruff check brain/ framework/ tests/

# Type check
mypy brain/ framework/

# Run all tests
pytest

# Run tests with coverage
pytest --cov=brain --cov=framework

# Run specific test file
pytest tests/test_ingestion.py -v
```

### Database
```bash
# Initialize brain database (via CLI)
princeps-brain init

# Start API server
uvicorn brain.interface.brain_api:create_app --factory --reload
```

## Architecture Patterns

### Brain Layer Patterns
- **Idempotency:** Use `get_or_create_operation()` for retry-safe operations
- **Multi-tenancy:** Always pass and validate `tenant_id`
- **Chunking:** Use `TextChunker` from ingestion service for consistent chunking
- **Embeddings:** Generate embeddings in batches for efficiency
- **Logging:** Use structured logging with tenant/operation context

### Framework Patterns
- **Agent Design:** Inherit from `BaseAgent` and implement `execute()` method
- **LLM Calls:** Use `MultiLLMClient` for automatic fallback
- **Tool Registry:** Register tools with proper schemas for validation
- **Task Dispatch:** Route tasks based on `TaskType` enum

## Common Scenarios

### Adding a New Document Type
1. Add file extension to `IngestConfig.include_extensions`
2. Implement extractor in `TextExtractor` if needed
3. Add tests in `tests/test_ingestion.py`

### Creating a New Agent
1. Create agent class in `framework/agents/`
2. Inherit from `BaseAgent`
3. Implement `execute()` method
4. Register agent in `Dispatcher`
5. Add tests for agent functionality

### Adding a New API Endpoint
1. Add endpoint in `brain/interface/brain_api.py`
2. Use Pydantic schemas for request/response validation
3. Add proper error handling and status codes
4. Document endpoint with FastAPI docstrings
5. Add tests for endpoint

## Dependencies

### Required External Services
- PostgreSQL 12+ with pgvector extension
- Optional: Anthropic/OpenAI/Google API keys for LLM features

### Python Version
- Minimum: Python 3.10
- Recommended: Python 3.11 or 3.12

## Additional Guidance

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense (Add, Fix, Update, Remove)
- Reference issue numbers when applicable

### Pull Requests
- Ensure all tests pass before creating PR
- Update documentation if adding new features
- Keep changes focused and minimal
- Provide clear description of changes

### Code Reviews
- Check for proper error handling
- Verify tenant isolation is maintained
- Ensure tests cover new functionality
- Look for security issues (PII exposure, injection risks)
- Verify logging is structured and informative

### Performance Considerations
- Batch database operations when possible
- Use connection pooling (configured in SQLAlchemy)
- Generate embeddings in batches
- Index database queries appropriately
- Profile slow operations and optimize

## Project-Specific Conventions

### Configuration
- Use environment variables for configuration
- Define defaults in code, override with `.env`
- Never commit secrets to repository
- Use `.env.example` as template

### Logging
- Use `logging.getLogger(__name__)` in each module
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Include context: tenant_id, operation_id, user_id
- Use structured logging format (JSON) for production

### Error Messages
- Provide actionable error messages
- Include context for debugging
- Don't expose internal implementation details
- Use appropriate HTTP status codes in API responses

## Examples

### Creating a Database Model
```python
from sqlalchemy import Column, String, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from brain.core.models import Base

class MyModel(Base):
    __tablename__ = "my_model"
    
    id = Column(UUID, primary_key=True, server_default=text("gen_random_uuid()"))
    tenant_id = Column(UUID, ForeignKey("tenant.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
```

### Writing a Test
```python
import pytest

from brain.ingestion import IngestService

class TestIngestService:
    """Tests for IngestService class."""
    
    def test_ingest_document_success(self, tmp_path):
        """Should successfully ingest a valid document."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Test ingestion
        service = IngestService()
        result = service.ingest_document(str(test_file))
        
        assert result.success is True
        assert result.document_id is not None
```

### Using Structured Logging
```python
from brain.observability import get_logger

logger = get_logger(__name__)

def process_document(doc_id: str, tenant_id: str):
    logger.info("Processing document", extra={
        "document_id": doc_id,
        "tenant_id": tenant_id,
        "operation": "process_document"
    })
```

## Resources
- Project README: `/README.md`
- Database Models: `brain/core/models.py`
- Pydantic Schemas: `brain/core/schemas.py`
- Example Configuration: `.env.example`
- Test Examples: `tests/` directory
