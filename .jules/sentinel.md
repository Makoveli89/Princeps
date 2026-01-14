## 2024-05-23 - Information Leakage in Error Handling
**Vulnerability:** The API endpoints `run_agent` and `execute_skill` were catching exceptions and raising `HTTPException` with `detail=str(e)`. This exposed raw exception messages, including database connection strings (with credentials) or internal file paths, to the API client.
**Learning:** Even internal-facing tools (like a console backend) can leak sensitive info if error handling is naive. The assumption that "errors are just strings" is dangerous when exceptions come from lower-level libraries (DB, OS).
**Prevention:** Always mask internal errors in production APIs. Log the full error with a unique ID, and return only that ID to the client. Never pass `str(e)` directly to `detail=` or response body.

## 2026-01-14 - Unbounded Input in API Requests
**Vulnerability:** Core API endpoints (`/api/workspaces`, `/api/run`, etc.) lacked length constraints on string fields. Specifically, `CreateWorkspaceRequest` allowed arbitrarily long names, and `ingest_document` had no file size limit.
**Learning:** Pydantic models default to allowing any string length unless explicitly constrained. This creates trivial Denial of Service (DoS) vectors where a single request can exhaust memory or database storage limits.
**Prevention:** Use `pydantic.Field(..., max_length=N)` for all string inputs exposed to the API. Enforce file size limits at the application level (middleware or endpoint) in addition to any reverse proxy limits.
