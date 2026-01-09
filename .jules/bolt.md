# Bolt's Journal

## 2024-05-22 - Missing Connection Pooling in Vector Search
**Learning:** The application was re-initializing the Vector Index (and underlying database connection pool) on every search request. This negates the benefits of connection pooling and adds significant overhead (connecting, authenticating) to every request.
**Action:** Always ensure expensive resources like database connection pools are initialized once (e.g., in `lifespan` or as a singleton) and reused across requests.
