# Bolt's Journal - Critical Learnings

This journal tracks critical performance learnings, anti-patterns, and architectural constraints discovered by Bolt.

## 2026-01-13 - [N+1 Query in Workspace Listing]
**Learning:** The workspace listing endpoint (`/api/workspaces`) was performing 3 extra queries per workspace to fetch counts for documents, chunks, and runs.
**Action:** Replaced loop-based counting with SQLAlchemy `scalar_subquery()` and `.correlate()` to fetch all data in a single optimized query.
