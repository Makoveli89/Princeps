## 2024-05-23 - Information Leakage in Error Handling
**Vulnerability:** The API endpoints `run_agent` and `execute_skill` were catching exceptions and raising `HTTPException` with `detail=str(e)`. This exposed raw exception messages, including database connection strings (with credentials) or internal file paths, to the API client.
**Learning:** Even internal-facing tools (like a console backend) can leak sensitive info if error handling is naive. The assumption that "errors are just strings" is dangerous when exceptions come from lower-level libraries (DB, OS).
**Prevention:** Always mask internal errors in production APIs. Log the full error with a unique ID, and return only that ID to the client. Never pass `str(e)` directly to `detail=` or response body.

## 2025-01-10 - Unvalidated Input on Resource Creation
**Vulnerability:** The `CreateWorkspaceRequest` Pydantic model lacked constraints on `name` and `description` fields. This allowed potentially unlimited string lengths (DoS risk) or invalid characters that could bypass downstream logic or cause unhandled database exceptions.
**Learning:** Defining types as `str` in Pydantic is insufficient for security. While the database schema may enforce limits (e.g., `VARCHAR(255)`), failing to validate at the API layer allows bad data to traverse the entire application stack before failing, consuming resources and potentially exposing database errors.
**Prevention:** Always use `Field` with `max_length` and regex patterns for string inputs in Pydantic models. Validate at the edge (API entry point), not just at the storage layer.
