## 2024-05-23 - Information Leakage in Error Handling
**Vulnerability:** The API endpoints `run_agent` and `execute_skill` were catching exceptions and raising `HTTPException` with `detail=str(e)`. This exposed raw exception messages, including database connection strings (with credentials) or internal file paths, to the API client.
**Learning:** Even internal-facing tools (like a console backend) can leak sensitive info if error handling is naive. The assumption that "errors are just strings" is dangerous when exceptions come from lower-level libraries (DB, OS).
**Prevention:** Always mask internal errors in production APIs. Log the full error with a unique ID, and return only that ID to the client. Never pass `str(e)` directly to `detail=` or response body.

## 2024-05-27 - DoS Protection in FastAPI
**Vulnerability:** The `ingest_document` endpoint accepted unlimited file sizes, reading them into memory. The `RunRequest` model accepted unlimited string input. Both allowed potential Denial of Service (DoS) via resource exhaustion.
**Learning:** Pydantic models need explicit `max_length` constraints for text fields exposed to users. FastAPI `UploadFile` should be checked for size before processing, preferably checking headers first then the stream.
**Prevention:** Use `Field(..., max_length=N)` for all public string inputs. Enforce strict file size limits on upload endpoints using `seek(0, 2)` or streaming counters.
