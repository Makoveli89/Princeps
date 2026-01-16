## 2024-05-23 - Information Leakage in Error Handling
**Vulnerability:** The API endpoints `run_agent` and `execute_skill` were catching exceptions and raising `HTTPException` with `detail=str(e)`. This exposed raw exception messages, including database connection strings (with credentials) or internal file paths, to the API client.
**Learning:** Even internal-facing tools (like a console backend) can leak sensitive info if error handling is naive. The assumption that "errors are just strings" is dangerous when exceptions come from lower-level libraries (DB, OS).
**Prevention:** Always mask internal errors in production APIs. Log the full error with a unique ID, and return only that ID to the client. Never pass `str(e)` directly to `detail=` or response body.

## 2025-01-14 - Broken Generator Dependencies on Validation Error
**Vulnerability:** When introducing strict Pydantic validation, the `get_db` dependency crashed with `RuntimeError: generator didn't stop after throw()` instead of returning 422. This masked the validation error with a 500 Internal Server Error.
**Learning:** FastAPI injects exceptions (like validation errors) into dependency generators to trigger cleanup. If the generator catches this exception and tries to `yield` a fallback value (thinking it's a setup error), it violates the generator protocol (yielding twice).
**Prevention:** Structure dependencies to distinguish between "setup" (before yield) and "usage" (yield). Only yield fallback values if setup fails. If usage fails, cleanup and exit without yielding.
