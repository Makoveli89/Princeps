# Bolt's Journal

## 2025-05-23 - Vector Search Indexing: HNSW vs IVFFlat
**Learning:** Initial attempt used `ivfflat` index for `pgvector` columns. While valid, `ivfflat` relies on k-means clustering calculated at index creation time. If created on an empty or small table (common in CI/CD or new deployments), the index is ineffective and requires a manual `REINDEX` after data load. `hnsw` is superior as it builds the graph incrementally, requires no training step, and handles dynamic data growth robustly without performance degradation.
**Action:** Prefer `hnsw` indexes for vector columns unless there are specific memory constraints prohibiting it. Always verify index type suitability for the deployment lifecycle (empty -> populated).
