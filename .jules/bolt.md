## 2024-05-23 - Singleton Vector Index
**Learning:** Initializing database adapters (like `PgVectorIndex` or `AsyncEngine`) inside request handlers creates a new connection pool for every request, leading to rapid resource exhaustion and high latency.
**Action:** Always initialize heavy resources with connection pools in the application `lifespan` (or `on_event("startup")`) and store them in `app.state` for reuse across requests. Ensure `close()` is called on shutdown.
