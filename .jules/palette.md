## 2024-05-22 - Interactive Cards Pattern
**Learning:** High-value interactive cards (like Workspaces) are implemented as `div`s with `onClick`, lacking keyboard accessibility.
**Action:** When touching list/grid components, always check for `onClick` on non-button elements and upgrade them to `role="button"` with keyboard handlers.
